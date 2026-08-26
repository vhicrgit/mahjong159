"""Rule bot v11: v2 plus generalized ukeire before tenpai."""

import os
from functools import lru_cache

from ..rules.win import shanten, shanten_cached
from ..rules.ting import discard_options, useful_draws, waiting_tiles

RED = 27


def _add(c: tuple[int, ...], t: int, delta: int) -> tuple[int, ...]:
    a = list(c)
    a[t] += delta
    return tuple(a)


def _shanten(c: tuple[int, ...] | list[int]) -> int:
    return shanten_cached(tuple(c))


@lru_cache(maxsize=300000)
def _ukeire(hand13: tuple[int, ...], unseen: tuple[int, ...]) -> int:
    if _shanten(hand13) == 0:
        tiles = waiting_tiles(list(hand13))
    else:
        tiles = useful_draws(list(hand13)).keys()
    return sum(unseen[t] for t in tiles)


@lru_cache(maxsize=300000)
def _shape_value(hand13: tuple[int, ...]) -> float:
    value = 0.0
    for base in (0, 9, 18):
        suit = hand13[base:base + 9]
        for i, n in enumerate(suit):
            if n >= 2:
                value += 0.6
            if n >= 3:
                value += 1.2
            if i <= 6 and suit[i] and suit[i + 1] and suit[i + 2]:
                value += 1.4
            if i <= 7 and suit[i] and suit[i + 1]:
                value += 0.7
            if i <= 6 and suit[i] and suit[i + 2]:
                value += 0.45
    value += 1.0 * hand13[RED]
    return value


class Bot:
    def __init__(self, game, seat: int):
        self.game = game
        self.seat = seat
        self.ukeire_weight = float(os.environ.get("V11_UKEIRE_W", 2.6))
        self.shape_weight = float(os.environ.get("V11_SHAPE_W", 1.0))
        self.risk_weight = float(os.environ.get("V11_RISK_W", 25.0))

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
        visible = self._visible_counts()
        unseen = tuple(max(0, 4 - visible[t]) for t in range(28))
        penged = self._penged_by_others()
        eg = self._endgame_factor()
        sh_w = 100.0 * (1.0 - 0.5 * eg)
        risk_w = self.risk_weight * (1.0 + 1.5 * eg)
        best_t, best_score = None, -1e18
        for o in opts:
            t = o["tile"]
            h = _add(counts14, t, -1)
            u = _ukeire(h, unseen)
            risk = 0.0
            if t != RED:
                if t in penged:
                    risk = 1.0
                else:
                    risk = {3: 0.4, 2: 0.2, 1: 0.05, 0: 0.0}.get(unseen[t], 0.4)
            score = (-sh_w * o["shanten"] + self.ukeire_weight * u
                     + self.shape_weight * _shape_value(h) - risk_w * risk)
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
