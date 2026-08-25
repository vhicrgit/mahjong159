"""安康159 - PIMC Bot: 完美信息蒙特卡罗（不作弊的强搜索）

Bridge/Skat 里的标准强方法, 也是 LuckyJ「不完美信息搜索」的核心思路:
  1. 采样 K 个与公开信息一致的"可能世界"(未见牌如何分配给对手手牌+牌墙顺序)
  2. 每个世界内信息完美 → 用 Oracle 的 beam search 算每个首出牌的效用
  3. 跨世界求平均 → argmax

与 Oracle 的区别: Oracle 直接读真实牌墙(作弊); PIMC 只用公开信息采样,
是真实 AI 可用的方法。Oracle 的 49.8% 是 PIMC 的上界(K→∞ 且世界猜对时)。

用法: Bot(game, seat, worlds=8, beam=8)
"""

import random

from ..rules.win import shanten
from ..rules.ting import discard_options
from .bot_oracle import search_first_discard_detail

RED = 27
N_WORLDS = 32
BEAM = 6
HORIZON = 5    # 只看未来 k 次自己的摸牌: 视野过长会让
               # 所有候选在随机世界里都能胡, 判别信号退化成噪声
RISK_WEIGHT = 0.20     # 放杠风险惩罚(效用尺度 0~1)


class Bot:
    """PIMC: 采样可能世界 + 完美信息搜索 + 跨世界投票"""

    def __init__(self, game, seat: int, worlds: int = N_WORLDS,
                 beam: int = BEAM, risk_weight: float = RISK_WEIGHT,
                 horizon: int = HORIZON):
        self.game = game
        self.seat = seat
        self.worlds = worlds
        self.beam = beam
        self.risk_weight = risk_weight
        self.horizon = horizon
        self._rng = random.Random((seat * 7919 + 13) & 0xffffffff)

    # ---------- 公开信息 ----------

    def _unseen_counts(self) -> list[int]:
        """未见牌计数: 总量4 - 我的手牌 - 所有弃牌 - 所有副露"""
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

    def _opp_hand_total(self) -> int:
        return sum(len(self.game.players[(self.seat + r) % 4].hand)
                   for r in (1, 2, 3))

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

    # ---------- 世界采样 ----------

    def _sample_my_future_draws(self, unseen_pool: list[int],
                                wall_size: int,
                                opp_total: int) -> list[int]:
        """采样一个世界: 打乱未见牌 → 前 opp_total 张给对手(不关心谁),
        其余按顺序构成牌墙 → 我的摸牌在 wall[3], wall[7], ..."""
        pool = unseen_pool[:]
        self._rng.shuffle(pool)
        wall = pool[opp_total:opp_total + wall_size]
        draws = []
        idx = 3
        while idx < len(wall) - 6 and len(draws) < self.horizon:
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
        if len(opts) == 1:
            return opts[0]["tile"]

        unseen = self._unseen_counts()
        pool = []
        for t in range(28):
            pool.extend([t] * unseen[t])
        opp_total = self._opp_hand_total()
        wall_size = max(0, len(pool) - opp_total)

        # 牌墙太浅: 搜索无意义, 退回牌效+防守启发式
        if wall_size < 8:
            return self._heuristic(counts14, opts, unseen)

        agg_win = {}     # tile -> 胡牌世界数
        agg_depth = {}   # tile -> 胡牌世界的深度和
        agg_sh = {}      # tile -> 未胡世界的向听和
        agg_n = {}       # tile -> 出现世界数
        for _ in range(self.worlds):
            future = self._sample_my_future_draws(pool, wall_size, opp_total)
            if not future:
                continue
            det = search_first_discard_detail(counts14, future, self.beam)
            for t, (wd, sh) in det.items():
                agg_n[t] = agg_n.get(t, 0) + 1
                if wd is not None:
                    agg_win[t] = agg_win.get(t, 0) + 1
                    agg_depth[t] = agg_depth.get(t, 0) + wd
                else:
                    agg_sh[t] = agg_sh.get(t, 0) + sh
        if not agg_n:
            return self._heuristic(counts14, opts, unseen)

        penged = self._penged_by_others()
        eg = self._endgame_factor()
        rw = self.risk_weight * (1.0 + 1.5 * eg)

        best_t, best_v = None, -1e18
        for t, n_w in agg_n.items():
            n_win = agg_win.get(t, 0)
            win_rate = n_win / n_w
            mean_depth = (agg_depth.get(t, 0) / n_win) if n_win else 0.0
            n_lose = n_w - n_win
            mean_sh = (agg_sh.get(t, 0) / n_lose) if n_lose else 0.0
            # 主项 P(胡) —— 低方差; 次项打破平手(更快胡 / 未胡时更接近听牌)
            v = win_rate - 0.02 * mean_depth - 0.05 * mean_sh
            # 放杠风险: 对手碰过的牌打第4张必被杠
            risk = 0.0
            if t != RED:
                if t in penged:
                    risk = 1.0
                else:
                    risk = {3: 0.4, 2: 0.2, 1: 0.05,
                            0: 0.0}.get(unseen[t], 0.4)
            v -= rw * risk
            if v > best_v:
                best_v, best_t = v, t
        return best_t

    def _heuristic(self, counts14, opts, unseen) -> int:
        """v2 同款启发式(终局/退化路径)"""
        penged = self._penged_by_others()
        eg = self._endgame_factor()
        best_tile, best_score = None, -1e9
        for o in opts:
            t = o["tile"]
            wr = sum(unseen[w] for w in o["waits"])
            risk = 0.0
            if t != RED:
                if t in penged:
                    risk = 1.0
                else:
                    risk = {3: 0.4, 2: 0.2, 1: 0.05,
                            0: 0.0}.get(unseen[t], 0.4)
            sh_w = 100.0 * (1.0 - 0.5 * eg)
            risk_w = 25.0 * (1.0 + 1.5 * eg)
            score = -sh_w * o["shanten"] + 3.0 * wr - risk_w * risk
            if score > best_score:
                best_score, best_tile = score, t
        return best_tile

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
