"""Rule bot v6: generalized ukeire and one-step continuation value."""

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
def _best_after_useful_draw(hand13: tuple[int, ...], unseen: tuple[int, ...], draw: int) -> tuple[int, int]:
    hand14 = _add(hand13, draw, 1)
    unseen2 = _add(unseen, draw, -1) if unseen[draw] > 0 else unseen
    best = (99, 0)
    for discard, cnt in enumerate(hand14):
        if cnt <= 0:
            continue
        next13 = _add(hand14, discard, -1)
        s = _shanten(next13)
        u = _ukeire(next13, unseen2)
        cand = (s, -u)
        if cand < (best[0], -best[1]):
            best = (s, u)
    return best


@lru_cache(maxsize=200000)
def _continuation(hand13: tuple[int, ...], unseen: tuple[int, ...]) -> float:
    total = sum(unseen)
    if total <= 0:
        return 0.0
    base_s = _shanten(hand13)
    acc = 0.0
    for draw, n in enumerate(unseen):
        if n <= 0:
            continue
        s, u = _best_after_useful_draw(hand13, unseen, draw)
        gain = max(0, base_s - s)
        acc += n / total * (36.0 * gain + 0.25 * u)
    return acc


class Bot:
    def __init__(self, game, seat: int):
        self.game = game
        self.seat = seat

    def _visible_counts(self) -> list[int]:
        visible = [0] * 28
        for q in self.game.players:
            for t in q.discards:
                visible[t] += 1
            for m in q.melds:
                visible[m["tile"]] += 3 if m["type"] == "peng" else 4
        p = self.game.players[self.seat]
        for t, n in enumerate(p.hand_counts):
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
        if len(opts) == 1:
            return opts[0]["tile"]

        unseen = self._unseen_counts()
        penged = self._penged_by_others()
        eg = self._endgame_factor()
        sh_w = 100.0 * (1.0 - 0.5 * eg)
        risk_w = 25.0 * (1.0 + 1.5 * eg)
        best_t, best_score = None, -1e18
        for o in opts:
            t = o["tile"]
            h = _add(counts14, t, -1)
            s = o["shanten"]
            u = _ukeire(h, unseen)
            cont = _continuation(h, unseen) if s <= 2 else 0.0
            risk = 0.0
            if t != RED:
                if t in penged:
                    risk = 1.0
                else:
                    risk = {3: 0.4, 2: 0.2, 1: 0.05, 0: 0.0}.get(unseen[t], 0.4)
            score = -sh_w * s + 2.2 * u + cont - risk_w * risk
            if score > best_score:
                best_score, best_t = score, t
        return best_t

    def decide_peng(self, tile: int) -> bool:
        p = self.game.players[self.seat]
        counts = p.hand_counts
        before = shanten(counts)
        c2 = list(counts)
        c2[tile] -= 2
        after = shanten(c2)
        if after < before:
            return before != 0 or after == 0
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
        return not (s_before == 0 and s_after > 0)
