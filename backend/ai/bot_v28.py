"""Rule bot v28: v10 skeleton + K-turn win-probability beam DP.

主选同 v10: 最小向听优先。同向听候选间的精修项由 v10 的启发式
continuation 换成**多步胡牌概率估值**:

  V(h) = P(未来 K 次自己摸牌内胡牌) + 叶子启发值(剩余质量的 shanten/ukeire)

其中每步的打牌决策用 v10 自己的启发式(greedy: shanten↓ 优先, 同则 ukeire↑),
即"启发式偏置引导的精确概率搜索"——不做随机采样(区别于 MC), 不做 1 状态
截断(区别于 bot_target 的 FRONTIER_CAP=1), 前沿保留概率最大的 B 个手牌状态。
对手抢胡以危险率 hazard 折现。
"""

import os
from functools import lru_cache

from ..rules.win import is_win, shanten, shanten_cached
from ..rules.ting import discard_options, useful_draws, waiting_tiles

RED = 27


def _add(c: tuple[int, ...], t: int, delta: int) -> tuple[int, ...]:
    a = list(c)
    a[t] += delta
    return tuple(a)


def _shanten(c: tuple[int, ...] | list[int]) -> int:
    return shanten_cached(tuple(c))


@lru_cache(maxsize=500000)
def _ukeire(hand13: tuple[int, ...], unseen: tuple[int, ...]) -> int:
    s = _shanten(hand13)
    if s == 0:
        tiles = waiting_tiles(list(hand13))
    else:
        tiles = list(useful_draws(list(hand13)).keys())
    return sum(unseen[t] for t in tiles)


@lru_cache(maxsize=500000)
def _greedy_next(hand14: tuple[int, ...], unseen: tuple[int, ...]):
    """摸到 hand14 后的最优打牌(v10 启发式: 向听↓优先, 同则进张↑)。
    返回 (打后13张, 向听, 进张数)。"""
    best_key, best_h = None, None
    for disc, cnt in enumerate(hand14):
        if cnt <= 0:
            continue
        h = _add(hand14, disc, -1)
        s = _shanten(h)
        u = _ukeire(h, unseen)
        key = (s, -u, disc)
        if best_key is None or key < best_key:
            best_key, best_h = key, h
    return best_h, best_key[0], best_key[1]


@lru_cache(maxsize=200000)
def _win_prob(hand13: tuple[int, ...], unseen: tuple[int, ...], k: int,
              beam: int) -> float:
    """P(未来 k 次摸牌内胡牌), greedy 策略 + 概率前沿 beam。"""
    total = sum(unseen)
    if k <= 0 or total <= 0:
        return 0.0
    frontier: dict[tuple, float] = {hand13: 1.0}
    win = 0.0
    for _ in range(k):
        nxt: dict[tuple, float] = {}
        for state, mass in frontier.items():
            for draw, n in enumerate(unseen):
                if n <= 0:
                    continue
                p = mass * n / total
                hand14 = _add(state, draw, 1)
                if is_win(list(hand14)):
                    win += p
                    continue
                h, _, _ = _greedy_next(hand14, unseen)
                nxt[h] = nxt.get(h, 0.0) + p
        if not nxt:
            break
        frontier = dict(sorted(nxt.items(), key=lambda kv: -kv[1])[:beam])
    return win


class Bot:
    def __init__(self, game, seat: int):
        self.game = game
        self.seat = seat
        self.k = int(os.environ.get("V28_K", 4))
        self.beam = int(os.environ.get("V28_BEAM", 64))
        self.hazard = float(os.environ.get("V28_HAZARD", 0.008))
        self.dp_weight = float(os.environ.get("V28_DP_W", 60.0))
        self.ukeire_weight = float(os.environ.get("V28_U_W", 1.0))
        self.shanten_weight = float(os.environ.get("V28_SH_W", 100.0))
        self.disc_keep = int(os.environ.get("V28_KEEP", 4))

    def _visible_counts(self) -> list[int]:
        visible = [0] * 28
        for q in self.game.players:
            for t in q.discards:
                visible[t] += 1
            for m in q.melds:
                visible[m["tile"]] += 3 if m["type"] == "peng" else 4
        for t, n in enumerate(self.game.players[self.seat].hand_counts):
            visible[t] += n
        return visible

    def _unseen_counts(self) -> tuple[int, ...]:
        visible = self._visible_counts()
        return tuple(max(0, 4 - visible[t]) for t in range(28))

    def choose_discard(self) -> int:
        p = self.game.players[self.seat]
        counts14 = tuple(p.hand_counts)
        opts = discard_options(list(counts14))
        if not opts:
            return p.hand[-1]
        if len(opts) == 1:
            return opts[0]["tile"]
        unseen = self._unseen_counts()
        min_sh = min(o["shanten"] for o in opts)
        k_eff = min(self.k, max(1, self.game.wall_remaining() // 4))
        cands = []
        for o in opts:
            t = o["tile"]
            h = _add(counts14, t, -1)
            s = o["shanten"]
            if s > min_sh:
                continue
            u = _ukeire(h, unseen)
            winp = _win_prob(h, unseen, k_eff, self.beam)
            score = (self.ukeire_weight * u
                     + self.dp_weight * winp * (1.0 - self.hazard) ** k_eff)
            cands.append((score, t))
        if not cands:
            cands = [(-self.shanten_weight * (1 + o["shanten"]), o["tile"])
                     for o in opts]
        return max(cands)[1]

    def decide_peng(self, tile: int) -> bool:
        p = self.game.players[self.seat]
        before = shanten(p.hand_counts)
        c = list(p.hand_counts)
        c[tile] -= 2
        after = shanten(c)
        if after < before:
            return before != 0 or after == 0
        return False

    def decide_gang(self, tile: int, kind: str) -> bool:
        p = self.game.players[self.seat]
        before = shanten(p.hand_counts)
        c = list(p.hand_counts)
        if kind == "ming":
            c[tile] -= 3
        elif kind == "an":
            c[tile] -= 4
        else:
            c[tile] -= 1
        after = shanten(c)
        return not (before == 0 and after > 0)
