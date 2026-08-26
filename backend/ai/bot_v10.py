"""Rule bot v10: minimum-shanten first, generalized ukeire tie-break."""

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
    s = _shanten(hand13)
    if s == 0:
        tiles = waiting_tiles(list(hand13))
    else:
        tiles = list(useful_draws(list(hand13)).keys())
    return sum(unseen[t] for t in tiles)


@lru_cache(maxsize=300000)
def _second_step_value(hand13: tuple[int, ...], unseen: tuple[int, ...]) -> float:
    total = sum(unseen)
    if total <= 0:
        return 0.0
    base_s = _shanten(hand13)
    v = 0.0
    for draw, n in enumerate(unseen):
        if n <= 0:
            continue
        hand14 = _add(hand13, draw, 1)
        if base_s == 0:
            from ..rules.win import is_win
            if is_win(list(hand14)):
                v += n / total * 50.0
                continue
        best_s, best_u = 99, 0
        for discard, cnt in enumerate(hand14):
            if cnt <= 0:
                continue
            h = _add(hand14, discard, -1)
            s = _shanten(h)
            u = _ukeire(h, unseen)
            if s < best_s or (s == best_s and u > best_u):
                best_s, best_u = s, u
        v += n / total * (20.0 * max(0, base_s - best_s) + 0.15 * best_u)
    return v


class Bot:
    def __init__(self, game, seat: int):
        self.game = game
        self.seat = seat
        self.shanten_weight = float(os.environ.get("V10_SHANTEN_W", 100.0))
        self.ukeire_weight = float(os.environ.get("V10_UKEIRE_W", 1.0))
        self.cont_weight = float(os.environ.get("V10_CONT_W", 0.5))
        self.risk_weight = float(os.environ.get("V10_RISK_W", 0.0))
        self.cont_max_shanten = int(os.environ.get("V10_CONT_MAX_SH", 2))

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
        best_t, best_score = None, -1e18
        for o in opts:
            t = o["tile"]
            h = _add(counts14, t, -1)
            s = o["shanten"]
            if s > min_sh:
                score = -10.0 * self.shanten_weight - self.shanten_weight * s
            else:
                u = _ukeire(h, unseen)
                cont = _second_step_value(h, unseen) if s <= self.cont_max_shanten else 0.0
                score = self.ukeire_weight * u + self.cont_weight * cont
            if t != RED:
                if t in penged:
                    risk = 1.0
                else:
                    risk = {3: 0.4, 2: 0.2, 1: 0.05, 0: 0.0}.get(unseen[t], 0.4)
                score -= self.risk_weight * (1.0 + 1.5 * eg) * risk
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
