"""对手牌型分析器(OppTracker) —— 根据对手的弃牌/碰杠行为反演其暗手分布。

方法: 粒子滤波 + 行为似然。
- 粒子 = 对手暗手的一个候选 counts(28 元组), 带权重。
- 对手摸牌不可见 -> 每次摸牌对所有可行牌分支, 权重 ∝ 观察者视角的剩余张数。
- 观察到对手打出 d -> 行为似然: 学者的出牌与 v10 打分的名次高度相关
  (标定: top1 62.8%, top3 82.3%), 用逐名次经验概率表 DISCARD_RANK_LLK
  作软似然(永不硬杀, 真手始终存活)。打分用 C 原生 score_discards_v10,
  单粒子单分支 ~10µs, 整局过滤毫秒-秒级。
- 碰/杠/不碰 -> 结构硬约束(碰必有对子等) × 标定软因子
  (碰的代理规则"碰后向听严格降"与学者实际一致率 82%, 见 PENGY_* 常数;
  明杠的 decide_gang 是精确确定性规则, 硬约束)。
- 初始粒子: 从观察者未知牌堆均匀无放回抽样(物理发牌等概 -> 正确的超几何先验),
  逐事件按权重剪枝到 beam 上限。

输出(均为后验): 每牌期望持有数/持有概率、向听分布、听牌概率、听口分布、
MAP 手牌。见 summary()。

标定数据来源: tools/perf/probe_scholar_policy.py(改学者策略后需重跑更新)。

注意: 与对局 harness 的约定 —— 若 harness 不向 bot 提供暗杠/补杠选项,
跟踪器也不能使用"没暗杠 => 没四张"这类约束(否则会把真手杀掉)。
当前 eval harness 不提供暗/补杠选项, 跟踪器不施加该约束。
"""

import random
from collections import defaultdict

from ..ai import bot_hv
from ..native import native
from ..rules.ting import waiting_tiles
from ..rules.tiles import tile_name

RED = 27

# 学者实际出牌在 v10 打分降序中的名次分布(probe_scholar_policy, n=2079)。
# 名次 0..6, 7, 8+(8+ 合并)。softmax 温度不拟合, 直接用非参表。
DISCARD_RANK_LLK = (0.6133, 0.1241, 0.0596, 0.0390, 0.0366, 0.0298,
                    0.0255, 0.0197, 0.0524)

# 碰决策的代理规则(碰后最优向听严格降)标定(n=264):
# P(实际碰 | 代理说碰) = 182/187, P(实际碰 | 代理说不碰) = 41/77
PENG_LLK = {True: 0.973, False: 0.532}     # 观察到"碰"
PASS_LLK = {True: 0.027, False: 0.468}     # 观察到"不碰"(键=代理是否说碰)


def _counts_str(counts) -> str:
    return " ".join(tile_name(t) for t in range(28) for _ in range(counts[t]))


class OppTracker:
    def __init__(self, opp_seat: int, hero_hand_counts, n_init: int = 4000,
                 beam: int = 800, policy: bool = True,
                 seed: int = 0, hero_seat: int = 0):
        """opp_seat: 被跟踪的对手座位。hero_hand_counts: 观察者(我方)起手。
        n_init: 初始抽样发牌数; beam: 每事件后保留的最大粒子数。
        policy: False 时退化为纯计数基线(不做行为似然, 用于对照)。
        hero_seat: 观察者座位(默认0; 四座位全跟踪时需要各自实例化)。"""
        self.opp = opp_seat
        self.hero_seat = hero_seat
        self.policy = policy
        self.beam = beam
        self.n_init = n_init
        self.rng = random.Random(seed)
        # ---- 公开信息镜像 ----
        self.discards = [[] for _ in range(4)]
        self.melds = [[] for _ in range(4)]   # (type, tile, kind)
        self.hero = list(hero_hand_counts)     # 我方暗手(观察者已知)
        self.pending_draw = False              # 对手刚摸了一张未知牌
        self.wall_rem = 83                     # 牌墙剩余(公开信息, 由 harness 喂入)
        self.particles: dict[tuple, float] = {}

    # ---------- 公开信息 ----------
    def _public(self) -> list[int]:
        c = [0] * 28
        for s in range(4):
            for t in self.discards[s]:
                c[t] += 1
            for m in self.melds[s]:
                c[m[1]] += 3 if m[0] == "peng" else 4
        return c

    def _unseen_for_opp(self, hand) -> list[int]:
        """学者视角的剩余牌 = 4 - (公开信息 + 自己候选手牌)。"""
        pub = self._public()
        return [max(0, 4 - pub[t] - hand[t]) for t in range(28)]

    def _penged_for_opp(self) -> list[int]:
        """学者视角"被别人碰过的牌"(v10 打分的输入)。"""
        penged = [0] * 28
        for s in range(4):
            if s == self.opp:
                continue
            for m in self.melds[s]:
                if m[0] == "peng":
                    penged[m[1]] = 1
        return penged

    def _draw_pool(self, hand) -> list[tuple[int, int]]:
        """对手摸牌的可行分支: [(t, 剩余张数)]。剩余 = 观察者未知的 t 的拷贝数
        (在墙里或其他两家手里 —— 近似认为等可能摸到)。"""
        pub = self._public()
        out = []
        for t in range(28):
            rem = 4 - pub[t] - self.hero[t] - hand[t]
            if rem > 0:
                out.append((t, rem))
        return out

    # ---------- 粒子集变换 ----------
    def _prune(self):
        ps = {h: w for h, w in self.particles.items() if w > 0}
        if len(ps) > self.beam:
            items = sorted(ps.items(), key=lambda kv: -kv[1])
            keep = dict(items[: self.beam // 2])
            rest = items[self.beam // 2:]
            if rest:
                ks = [h for h, _ in rest]
                ws = [w for _, w in rest]
                for h in self.rng.choices(ks, weights=ws,
                                          k=self.beam - len(keep)):
                    if h not in keep:
                        keep[h] = ps[h]
            ps = keep
        tot = sum(ps.values())
        if tot > 0:
            self.particles = {h: w / tot for h, w in ps.items()}
        else:
            self.particles = ps

    def _recover_if_empty(self, extra_melds: int = 0):
        """粒子集塌缩(beam 早期丢掉真手后, 后续硬约束团灭)时的兜底:
        按当前结构状态(副露数决定暗手张数)重新均匀抽样, 权重退回均匀。
        丢弃了历史行为信息, 但保证输出始终可用。
        extra_melds: 已过滤但尚未计入镜像的副露数(碰/杠过滤场景)。"""
        if self.particles:
            return
        n_melds = len(self.melds[self.opp]) + extra_melds
        n_tiles = 13 - 3 * n_melds
        if n_tiles <= 0:
            return
        pub = self._public()
        pool = []
        for t in range(28):
            pool += [t] * max(0, 4 - pub[t] - self.hero[t])
        if len(pool) < n_tiles:
            return
        cnt = defaultdict(float)
        for _ in range(self.n_init):
            h = [0] * 28
            for t in self.rng.sample(pool, n_tiles):
                h[t] += 1
            cnt[tuple(h)] += 1.0
        self.particles = dict(cnt)
        self._prune()

    def _init_particles(self, n_tiles: int):
        pool = []
        for t in range(28):
            pool += [t] * (4 - self.hero[t])
        cnt = defaultdict(float)
        for _ in range(self.n_init):
            draw = self.rng.sample(pool, n_tiles)
            h = [0] * 28
            for t in draw:
                h[t] += 1
            cnt[tuple(h)] += 1.0
        self.particles = dict(cnt)
        self._prune()

    # ---------- 行为似然 ----------
    def _discard_llk(self, h14, d, unseen, penged, eg) -> float:
        """P(学者打 d | 手牌 h14): v10 名次的经验概率。"""
        rows = native.score_discards_v10(list(h14), unseen, penged, eg)
        rows.sort(key=lambda r: -r["score"])
        for i, r in enumerate(rows):
            if r["tile"] == d:
                return DISCARD_RANK_LLK[min(i, 8)]
        return DISCARD_RANK_LLK[-1]

    @staticmethod
    def _peng_proxy(hand, tile) -> bool:
        """代理: 碰后(打一张最优)向听严格下降。"""
        before = native.shanten(list(hand))
        h2 = list(hand)
        h2[tile] -= 2
        best = 99
        for d in range(28):
            if h2[d] <= 0:
                continue
            h2[d] -= 1
            best = min(best, native.shanten(h2))
            h2[d] += 1
        return best < before

    # ---------- 事件接口(由 harness 按引擎时序调用) ----------
    def notify_deal(self, dealer: int):
        n = 14 if self.opp == dealer else 13
        self._init_particles(n)

    def notify_draw(self, seat: int, tile: int | None, wall_rem: int):
        """seat 摸了一张牌(hero 给真实 tile, 其他人给 None)。"""
        self.wall_rem = wall_rem
        if seat == self.opp:
            self.pending_draw = True
        elif seat == self.hero_seat:
            self.hero[tile] += 1

    def notify_discard(self, seat: int, tile: int, wall_rem: int):
        self.wall_rem = wall_rem
        if seat == self.opp:
            # 学者先决策后打出: 似然用打出前的公开信息
            self._filter_opp_discard(tile)
        if seat == self.hero_seat:
            self.hero[tile] -= 1
        self.discards[seat].append(tile)

    def _filter_opp_discard(self, tile: int):
        drew, self.pending_draw = self.pending_draw, False
        eg = max(0.0, min(1.0, (60 - self.wall_rem) / 60.0))
        penged = self._penged_for_opp() if self.policy else None
        new = defaultdict(float)
        for h, w in self.particles.items():
            if drew:
                branches = self._draw_pool(h)
            else:
                branches = [(None, 1.0)]
            for t, rem in branches:
                if t is None:
                    h14 = h
                    pw = 1.0
                else:
                    h14l = list(h)
                    h14l[t] += 1
                    h14 = tuple(h14l)
                    pw = float(rem)
                if h14[tile] <= 0:
                    continue
                if self.policy:
                    unseen = self._unseen_for_opp(h14)
                    pw *= self._discard_llk(h14, tile, unseen, penged, eg)
                h13 = list(h14)
                h13[tile] -= 1
                new[tuple(h13)] += w * pw
        self.particles = dict(new)
        self._prune()
        self._recover_if_empty()

    def notify_claim(self, seat: int, action: str | None, tile: int,
                     discarder: int):
        """action: 'peng' / 'gang' / None(过)。tile 为被碰/杠的那张。"""
        if seat == self.opp:
            # 学者的碰/杠决策发生在该牌还在桌上、副露未入账时
            self._filter_opp_claim(action, tile)
        if action is not None:
            # 引擎行为: 被碰/杠的牌从打出者弃牌堆移除
            if self.discards[discarder] and \
                    self.discards[discarder][-1] == tile:
                self.discards[discarder].pop()
            kind = "ming" if action == "gang" else None
            self.melds[seat].append((action, tile, kind))
            if seat == self.hero_seat:
                self.hero[tile] -= 2 if action == "peng" else 3

    def _filter_opp_claim(self, action: str | None, tile: int):
        new = {}
        for h, w in self.particles.items():
            if action == "peng":
                if h[tile] < 2:
                    continue
                # harness 先问杠再问碰: 有 3 张且想杠就不会轮到碰(精确硬约束)
                if h[tile] >= 3 and bot_hv.decide_gang(h, tile, "ming"):
                    continue
                if self.policy:
                    w *= PENG_LLK[self._peng_proxy(h, tile)]
                h2 = list(h)
                h2[tile] -= 2
                new[tuple(h2)] = w
            elif action == "gang":
                if h[tile] < 3:
                    continue
                # 明杠判定是精确确定性规则(shanten 口径), 硬约束
                if not bot_hv.decide_gang(h, tile, "ming"):
                    continue
                h2 = list(h)
                h2[tile] -= 3
                new[tuple(h2)] = w
            else:  # 过
                if h[tile] >= 3 and bot_hv.decide_gang(h, tile, "ming"):
                    continue
                if self.policy and h[tile] >= 2:
                    w *= PASS_LLK[self._peng_proxy(h, tile)]
                new[h] = w
        self.particles = new
        self._prune()
        self._recover_if_empty(extra_melds=1 if action in ("peng", "gang")
                               else 0)

    # ---------- 后验输出 ----------
    def expected_counts(self, beta: float = 1.0) -> list[float]:
        """E[持有拷贝数]。beta>1 做温度 sharpening(w^β 归一化), 后验弥散时
        让高置信假设主导, 供下游决策用。"""
        if beta != 1.0:
            items = [(h, w ** beta) for h, w in self.particles.items()]
            tot = sum(w for _, w in items)
            if tot > 0:
                items = [(h, w / tot) for h, w in items]
        else:
            items = self.particles.items()
        out = [0.0] * 28
        for h, w in items:
            for t in range(28):
                if h[t]:
                    out[t] += w * h[t]
        return out

    def hold_probs(self) -> list[float]:
        out = [0.0] * 28
        for h, w in self.particles.items():
            for t in range(28):
                if h[t]:
                    out[t] += w
        return out

    def shanten_dist(self) -> dict[int, float]:
        d = defaultdict(float)
        for h, w in self.particles.items():
            d[native.shanten(list(h))] += w
        return dict(sorted(d.items()))

    def tenpai_prob(self) -> float:
        return sum(w for h, w in self.particles.items()
                   if native.shanten(list(h)) == 0)

    def wait_probs(self) -> dict[int, float]:
        """P(t 是对手的听口) = 听牌粒子中含 t 听口的权重和。"""
        d = defaultdict(float)
        for h, w in self.particles.items():
            if native.shanten(list(h)) != 0:
                continue
            for t in waiting_tiles(list(h)):
                d[t] += w
        return dict(d)

    def map_hand(self):
        if not self.particles:
            return None, 0.0
        return max(self.particles.items(), key=lambda kv: kv[1])

    def summary(self) -> str:
        lines = [f"对手(座{self.opp}) 粒子数={len(self.particles)}"
                 f" 听牌概率={self.tenpai_prob():.1%}"]
        sd = self.shanten_dist()
        lines.append("向听分布: " + " ".join(f"{k}:{v:.0%}" for k, v in sd.items()))
        exp_c = self.expected_counts()
        top = sorted(range(28), key=lambda t: -exp_c[t])[:8]
        lines.append("最可能持有: " + " ".join(
            f"{tile_name(t)}×{exp_c[t]:.2f}" for t in top if exp_c[t] > 0.05))
        mh, mw = self.map_hand()
        if mh:
            lines.append(f"MAP 手牌({mw:.0%}): {_counts_str(mh)}")
        wp = self.wait_probs()
        if wp:
            lines.append("听口分布: " + " ".join(
                f"{tile_name(t)}:{v:.0%}" for t, v in
                sorted(wp.items(), key=lambda kv: -kv[1])))
        return "\n".join(lines)
