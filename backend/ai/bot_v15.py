"""Rule bot v15: pruned finite-horizon win-probability search."""

import os
from functools import lru_cache

from ..rules.win import is_win, shanten, shanten_cached
from ..rules.ting import discard_options, useful_draws, waiting_tiles

RED = 27


def _add(c: tuple[int, ...], tile: int, delta: int) -> tuple[int, ...]:
    a = list(c)
    a[tile] += delta
    return tuple(a)


def _shanten(c: tuple[int, ...] | list[int]) -> int:
    return shanten_cached(tuple(c))


@lru_cache(maxsize=500000)
def _waits(hand13: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(waiting_tiles(list(hand13))) if _shanten(hand13) == 0 else ()


@lru_cache(maxsize=500000)
def _useful(hand13: tuple[int, ...]) -> tuple[int, ...]:
    if _shanten(hand13) == 0:
        return _waits(hand13)
    return tuple(useful_draws(list(hand13)).keys())


@lru_cache(maxsize=500000)
def _ukeire(hand13: tuple[int, ...], unseen: tuple[int, ...]) -> int:
    return sum(unseen[t] for t in _useful(hand13))


def _rank_discards(hand14: tuple[int, ...], unseen: tuple[int, ...], width: int):
    rows = []
    for discard, cnt in enumerate(hand14):
        if cnt <= 0:
            continue
        h = _add(hand14, discard, -1)
        rows.append((_shanten(h), -_ukeire(h, unseen), h[discard], discard, h))
    rows.sort()
    return rows[:width]


@lru_cache(maxsize=500000)
def _win_prob(hand13: tuple[int, ...], unseen: tuple[int, ...], turns: int,
              width: int) -> float:
    if turns <= 0:
        return 0.0
    total = sum(unseen)
    if total <= 0:
        return 0.0
    win = 0.0
    cont = 0.0
    useful = set(_useful(hand13))
    stay_p = 0.0
    for draw, n in enumerate(unseen):
        if n <= 0:
            continue
        p = n / total
        hand14 = _add(hand13, draw, 1)
        if is_win(list(hand14)):
            win += p
            continue
        if draw not in useful:
            stay_p += p
            continue
        best = 0.0
        for _, _, _, _, h in _rank_discards(hand14, unseen, width):
            best = max(best, _win_prob(h, unseen, turns - 1, width))
        cont += p * best
    return win + cont + stay_p * _win_prob(hand13, unseen, turns - 1, width)


class Bot:
    def __init__(self, game, seat: int):
        self.game = game
        self.seat = seat
        self.horizon = int(os.environ.get("V15_H", 5))
        self.width = int(os.environ.get("V15_WIDTH", 3))
        self.prob_weight = float(os.environ.get("V15_PROB_W", 220.0))
        self.ukeire_weight = float(os.environ.get("V15_UKEIRE_W", 2.0))
        self.risk_weight = float(os.environ.get("V15_RISK_W", 10.0))

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

    def _penged_by_others(self) -> set[int]:
        out = set()
        for q in self.game.players:
            if q.seat == self.seat:
                continue
            for m in q.melds:
                if m["type"] == "peng":
                    out.add(m["tile"])
        return out

    def _endgame_factor(self) -> float:
        return max(0.0, min(1.0, (60 - self.game.wall_remaining()) / 60.0))

    def choose_discard(self) -> int:
        p = self.game.players[self.seat]
        counts14 = tuple(p.hand_counts)
        opts = discard_options(list(counts14))
        if not opts:
            return p.hand[-1]
        unseen = self._unseen_counts()
        penged = self._penged_by_others()
        eg = self._endgame_factor()
        min_sh = min(o["shanten"] for o in opts)
        horizon = min(self.horizon, max(1, self.game.wall_remaining() // 4 + 1))
        best_t, best_score = None, -1e18
        for o in opts:
            t = o["tile"]
            h = _add(counts14, t, -1)
            s = o["shanten"]
            u = _ukeire(h, unseen)
            prob = _win_prob(h, unseen, horizon, self.width) if s <= min_sh + 1 else 0.0
            risk = 0.0
            if t != RED:
                if t in penged:
                    risk = 1.0
                else:
                    risk = {3: 0.4, 2: 0.2, 1: 0.05, 0: 0.0}.get(unseen[t], 0.4)
            score = (-100.0 * s + self.prob_weight * prob
                     + self.ukeire_weight * u - self.risk_weight * (1.0 + 1.5 * eg) * risk)
            if score > best_score:
                best_score, best_t = score, t
        return best_t

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
