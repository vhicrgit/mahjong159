"""安康159 - 规则Bot v4: 解析骨架 + 同向听候选内的采样搜索精修

设计依据（实测教训）:
- 纯 PIMC 采样失败（15-21%）: shanten/进张是**解析精确值(零方差)**,
  用采样去估同一个量必然更差。32 个世界的 P(胡) 标准误 ~0.09,
  远大于好坏出牌的真实差距 ~0.03。
- v3 混合版有效（29.4%）: 解析骨架定主选 + 采样只做精修。

v4 结构:
  1. 解析层: 算所有候选的 shanten, 只保留最小向听类（主导因素, 精确）
  2. 精修层: 同向听候选间, 采样 W 个可能世界, 每个世界跑 beam search
     估 "k 次摸牌内自摸胡" 的概率与速度 —— 这比 wait_count 更准
     （wait_count 只看一步进张, 搜索看多步序列）
  3. 防守层: 放杠风险惩罚（对手碰过的牌打第4张必被杠）

用法: Bot(game, seat, worlds=48, beam=6, horizon=6)
"""

import random

from ..rules.win import shanten
from ..rules.ting import discard_options
from .bot_oracle import search_first_discard_detail, WIN_DISCOUNT

RED = 27
N_WORLDS = 48
BEAM = 6
HORIZON = 6
RISK_SCALE = 25.0      # 与解析骨架同尺度(shanten 权重 100)
REFINE_SCALE = 30.0    # 精修项尺度: 效用∈(0,1] → 最多 30 分, 不越过向听差(100)


class Bot:
    """v4: 解析骨架 + 同向听内搜索精修 + 放杠防守"""

    def __init__(self, game, seat: int, worlds: int = N_WORLDS,
                 beam: int = BEAM, horizon: int = HORIZON,
                 refine_scale: float = REFINE_SCALE):
        self.game = game
        self.seat = seat
        self.worlds = worlds
        self.beam = beam
        self.horizon = horizon
        self.refine_scale = refine_scale
        self._rng = random.Random((seat * 7919 + 13) & 0xffffffff)

    # ---------- 公开信息 ----------

    def _unseen_counts(self) -> list[int]:
        unseen = [4] * 28
        me = self.game.players[self.seat]
        for t in range(28):
            unseen[t] -= me.hand_counts[t]
        for q in self.game.players:
            for t in q.discards:
                unseen[t] -= 1
            for m in q.melds:
                unseen[m["tile"]] -= 3 if m["type"] == "peng" else 4
        return [max(0, u) for u in unseen]

    def _penged_by_others(self) -> set[int]:
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

    def _sample_future(self, pool, wall_size, opp_total):
        p = pool[:]
        self._rng.shuffle(p)
        wall = p[opp_total:opp_total + wall_size]
        draws, idx = [], 3
        while idx < len(wall) and len(draws) < self.horizon:
            draws.append(wall[idx])
            idx += 4
        return draws

    # ---------- 决策 ----------

    def choose_discard(self) -> int:
        p = self.game.players[self.seat]
        counts14 = list(p.hand_counts)
        opts = discard_options(counts14)
        if not opts:
            return p.hand[-1]

        unseen = self._unseen_counts()
        penged = self._penged_by_others()
        eg = self._endgame_factor()
        risk_w = RISK_SCALE * (1.0 + 1.5 * eg)

        def risk_of(t):
            if t == RED:
                return 0.0
            if t in penged:
                return 1.0
            return {3: 0.4, 2: 0.2, 1: 0.05, 0: 0.0}.get(unseen[t], 0.4)

        # 1) 解析层: 向听 + 进张(精确) —— 主导项
        min_sh = min(o["shanten"] for o in opts)
        base = {}
        for o in opts:
            t = o["tile"]
            wr = sum(unseen[w] for w in o["waits"])
            sh_w = 100.0 * (1.0 - 0.5 * eg)
            base[t] = -sh_w * o["shanten"] + 3.0 * wr - risk_w * risk_of(t)

        # 2) 精修层: 只在最小向听候选间做搜索(其余已被向听差压死)
        cands = [o["tile"] for o in opts if o["shanten"] == min_sh]
        if len(cands) <= 1 or self.worlds <= 0:
            return max(base.items(), key=lambda kv: kv[1])[0]

        pool = []
        for t in range(28):
            pool.extend([t] * unseen[t])
        opp_total = sum(len(self.game.players[(self.seat + r) % 4].hand)
                        for r in (1, 2, 3))
        wall_size = max(0, len(pool) - opp_total)
        if wall_size < 4:
            return max(base.items(), key=lambda kv: kv[1])[0]

        cand_set = set(cands)
        util_sum = {t: 0.0 for t in cands}
        n_worlds = 0
        for _ in range(self.worlds):
            future = self._sample_future(pool, wall_size, opp_total)
            if not future:
                continue
            det = search_first_discard_detail(counts14, future, self.beam,
                                             candidates=cand_set)
            n_worlds += 1
            h = len(future)
            for t, (wd, sh) in det.items():
                # 同一指数尺度: 胡=DISCOUNT^depth, 未胡=按剩余向听悲观外推
                util_sum[t] += (WIN_DISCOUNT ** wd) if wd is not None \
                    else WIN_DISCOUNT ** (h + 2 * max(0, sh) + 1)
        if n_worlds == 0:
            return max(base.items(), key=lambda kv: kv[1])[0]

        best_t, best_v = None, -1e18
        for t in cands:
            v = base[t] + self.refine_scale * (util_sum[t] / n_worlds)
            if v > best_v:
                best_v, best_t = v, t
        return best_t

    # ---------- 碰杠 (v2 逻辑) ----------

    def decide_peng(self, tile: int) -> bool:
        p = self.game.players[self.seat]
        counts = p.hand_counts
        before = shanten(counts)
        c2 = list(counts)
        c2[tile] -= 2
        after = shanten(c2)
        if after < before:
            if before == 0:
                return after == 0
            return True
        return False

    def decide_gang(self, tile: int, kind: str) -> bool:
        p = self.game.players[self.seat]
        counts = p.hand_counts
        s_before = shanten(counts)
        c2 = list(counts)
        if kind == "ming":
            c2[tile] -= 3
        elif kind == "an":
            c2[tile] -= 4
        else:
            c2[tile] -= 1
        s_after = shanten(c2)
        if s_before == 0 and s_after > 0:
            return False
        return True
