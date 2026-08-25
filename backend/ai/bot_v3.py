"""安康159 - 规则AI对手 v3: 蒙特卡罗出牌

核心思想: v1/v2 的出牌评分是单步启发式(-100×向听 + 3×进张 - 风险)。
v3 用蒙特卡罗 rollout 估计每个候选打牌的"到胡牌距离":
  对候选 t: 模拟 K 条后续摸牌序列(从未见牌中按剩余张数加权抽样),
  每条序列贪心摸打(向听最小化), 记录首次听牌/胡牌的巡数。
  候选得分 = 期望(胡牌巡数) 的负值 + 防杠风险惩罚。

成本控制:
  - shanten_cached 模块级 lru_cache, 热 ~0.5us
  - 每条 rollout 最多 sim_turns 巡, 每巡一次 discard 决策
    (用简化规则: 打向听数最大的孤张, 不做完整候选评估)
  - K=32 条 × 14 候选 × ~8 巡 × ~5 次 shanten = ~18k 次查询
  - 缓存命中后单决策 ~30-80ms (v2 单次 choose_discard ~2ms, 慢 20x
    但数据生成可并行, 可接受)

用法与 v1/v2 完全兼容 (Bot(game, seat), choose_discard 等)。
"""

import random

from ..rules.win import shanten_cached
from ..rules.ting import discard_options, waiting_tiles

RED = 27
SIM_ROLLOUTS = 32      # 每候选模拟序列数
SIM_TURNS = 10         # 每条序列最多模拟巡数


def _shanten(counts: list[int]) -> int:
    return shanten_cached(tuple(counts))


def _fast_discard(counts13: list[int]) -> int:
    """rollout 内部快速打牌: 向听最小优先; 平手时打孤张(张数1的)
    ——保留对子/刻子潜力, 拆对子是大忌(实测拆对导致胡牌率崩塌)。"""
    best_t, best_key = None, None
    for t in range(28):
        if counts13[t] <= 0:
            continue
        counts13[t] -= 1
        s = _shanten(counts13)
        counts13[t] += 1
        # 孤张(count=1)优先打出: key 第二项越小越先打
        key = (s, counts13[t])
        if best_key is None or key < best_key:
            best_key, best_t = key, t
    return best_t


class Bot:
    """规则Bot v3: 蒙特卡罗出牌 + v2 的碰杠/防守逻辑"""

    def __init__(self, game, seat: int, sim_rollouts: int = SIM_ROLLOUTS):
        self.game = game
        self.seat = seat
        self.sim_rollouts = sim_rollouts
        self._rng = random.Random(seed := (id(game) * 31 + seat) & 0xffffffff)

    # ---------- 可见信息 ----------

    def _unseen_bag(self) -> list[int]:
        """未见牌袋(按剩余张数展开), 供抽样"""
        visible = [0] * 28
        for q in self.game.players:
            for t in q.discards:
                visible[t] += 1
            for m in q.melds:
                visible[m["tile"]] += 3 if m["type"] == "peng" else 4
        p = self.game.players[self.seat]
        for t in range(28):
            visible[t] += p.hand_counts[t]
        bag = []
        for t in range(28):
            remain = 4 - visible[t]
            if t == RED:
                remain = 0  # 红中按实际 4 张计
            if t == RED:
                remain = max(0, 4 - visible[t])
            bag.extend([t] * max(0, remain))
        return bag

    def _penged_tiles_by_others(self) -> set[int]:
        tiles = set()
        for q in self.game.players:
            if q.seat == self.seat:
                continue
            for m in q.melds:
                if m["type"] == "peng":
                    tiles.add(m["tile"])
        return tiles

    def _endgame_factor(self) -> float:
        wall = self.game.wall_remaining()
        return max(0.0, min(1.0, (60 - wall) / 60.0))

    # ---------- 蒙特卡罗出牌 ----------

    def _rollout_score(self, counts_after: list[int], bag: list[int],
                       turns: int) -> float:
        """从打完牌的手牌(13张)出发, 模拟 turns 巡摸打。
        抽出的牌从袋中移除(模拟真实摸牌消耗); 打出的牌不回袋
        (对手可能拿走/自己再摸概率低, 回袋会稀释进张概率)。"""
        score = 0.0
        counts = list(counts_after)
        rng = self._rng
        for turn in range(turns):
            if not bag:
                break
            i = rng.randrange(len(bag))
            tile = bag[i]
            bag[i] = bag[-1]
            bag.pop()
            counts[tile] += 1
            s = _shanten(counts)
            if s == -1:
                score += 10.0 - turn
                return score
            if s == 0:
                score += 2.0 - 0.1 * turn
            t = _fast_discard(counts)
            counts[t] -= 1
        score -= 1.5 * max(0, _shanten(counts))
        return score

    def choose_discard(self) -> int:
        p = self.game.players[self.seat]
        counts = p.hand_counts
        opts = discard_options(counts)
        if not opts:
            return p.hand[-1]

        # 终局时 MC 无意义(来不及摸牌), 退化为 v2 策略
        eg = self._endgame_factor()
        wall = self.game.wall_remaining()
        if wall < 12:
            return self._heuristic_discard(counts, opts)

        bag = self._unseen_bag()
        penged = self._penged_tiles_by_others()
        turns = min(SIM_TURNS, wall // 4 + 2)

        best_tile, best_score = None, -1e9
        for o in opts:
            t = o["tile"]
            # 向听数主导(与 v2 一致的牌效骨架)
            base = -100.0 * o["shanten"] + 3.0 * o["wait_count"]
            # MC 精细化: 听牌候选估计真实胡牌概率(多面听宽度×剩余张),
            # 非听候选估计到听距离
            c2 = list(counts)
            c2[t] -= 1
            mc = 0.0
            if o["shanten"] <= 1 and len(bag) >= turns:
                n = self.sim_rollouts
                for _ in range(n):
                    mc += self._rollout_score(c2, bag, turns)
                mc /= n
                if o["shanten"] == 0:
                    # 听牌: MC 分数直接反映胡牌概率×速度, 权重放大
                    # (base 对听牌候选无区分度, 全是 -0+3×wait)
                    base = 0.0
                    score_w = 12.0
                else:
                    score_w = 2.0
            else:
                score_w = 0.0
            # 防杠风险
            risk = 0.0
            if t != RED:
                if t in penged:
                    risk = 1.0
                else:
                    remain = sum(1 for x in bag if x == t)
                    base_r = {3: 0.4, 2: 0.2, 1: 0.05, 0: 0.0}.get(
                        remain, 0.4)
                    risk = base_r
            score = base + score_w * mc \
                - 25.0 * (1.0 + 1.5 * eg) * risk
            if score > best_score:
                best_score, best_tile = score, t
        return best_tile

    def _heuristic_discard(self, counts, opts) -> int:
        """v2 同款启发式(MC 退化路径)"""
        visible = [0] * 28
        for q in self.game.players:
            for t in q.discards:
                visible[t] += 1
            for m in q.melds:
                visible[m["tile"]] += 3 if m["type"] == "peng" else 4
        p = self.game.players[self.seat]
        for t in range(28):
            visible[t] += p.hand_counts[t]
        penged = self._penged_tiles_by_others()
        eg = self._endgame_factor()
        best_tile, best_score = None, -1e9
        for o in opts:
            t = o["tile"]
            wr = sum(max(0, 4 - visible[w]) for w in o["waits"])
            risk = 0.0
            if t != RED:
                if t in penged:
                    risk = 1.0
                else:
                    remain = max(0, 4 - visible[t])
                    risk = {3: 0.4, 2: 0.2, 1: 0.05, 0: 0.0}.get(
                        remain, 0.4)
            sh_w = 100.0 * (1.0 - 0.5 * eg)
            risk_w = 25.0 * (1.0 + 1.5 * eg)
            score = -sh_w * o["shanten"] + 3.0 * wr - risk_w * risk
            if score > best_score:
                best_score, best_tile = score, t
        return best_tile

    # ---------- 碰杠: 同 v2 ----------

    def decide_peng(self, tile: int) -> bool:
        p = self.game.players[self.seat]
        counts = p.hand_counts
        before = _shanten(counts)
        c2 = list(counts)
        c2[tile] -= 2
        after = _shanten(c2)
        if after < before:
            if before == 0:
                return after == 0
            return True
        return False

    def decide_gang(self, tile: int, kind: str) -> bool:
        p = self.game.players[self.seat]
        counts = p.hand_counts
        s_before = _shanten(counts)
        if kind == "ming":
            c2 = list(counts)
            c2[tile] -= 3
        elif kind == "an":
            c2 = list(counts)
            c2[tile] -= 4
        else:
            c2 = list(counts)
            c2[tile] -= 1
        s_after = _shanten(c2)
        if s_before == 0 and s_after > 0:
            return False
        return True
