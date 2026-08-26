"""Rule bot v19: min-shanten with greedy finite-horizon probability."""

import os
from functools import lru_cache

from ..rules.win import is_win, shanten, shanten_cached
from ..rules.ting import discard_options, waiting_tiles

RED = 27


def _add(c: tuple[int, ...], tile: int, delta: int) -> tuple[int, ...]:
    a = list(c)
    a[tile] += delta
    return tuple(a)


def _shanten(c: tuple[int, ...] | list[int]) -> int:
    return shanten_cached(tuple(c))


@lru_cache(maxsize=500000)
def _wait_count(hand13: tuple[int, ...], unseen: tuple[int, ...]) -> int:
    if _shanten(hand13) != 0:
        return 0
    return sum(unseen[w] for w in waiting_tiles(list(hand13)))


@lru_cache(maxsize=500000)
def _best_discard_state(hand14: tuple[int, ...], unseen: tuple[int, ...]) -> tuple[int, ...]:
    best_key, best_h = None, None
    for discard, cnt in enumerate(hand14):
        if cnt <= 0:
            continue
        h = _add(hand14, discard, -1)
        key = (_shanten(h), -_wait_count(h, unseen), h[discard], discard)
        if best_key is None or key < best_key:
            best_key, best_h = key, h
    return best_h


@lru_cache(maxsize=500000)
def _win_value(hand13: tuple[int, ...], unseen: tuple[int, ...], turns: int) -> float:
    if turns <= 0:
        return 0.0
    total = sum(unseen)
    if total <= 0:
        return 0.0
    acc = 0.0
    for draw, n in enumerate(unseen):
        if n <= 0:
            continue
        hand14 = _add(hand13, draw, 1)
        p = n / total
        if is_win(list(hand14)):
            acc += p
        else:
            next13 = _best_discard_state(hand14, unseen)
            acc += p * _win_value(next13, unseen, turns - 1)
    return acc


class Bot:
    def __init__(self, game, seat: int):
        self.game = game
        self.seat = seat
        self.horizon = int(os.environ.get("V19_H", 5))
        self.prob_weight = float(os.environ.get("V19_PROB_W", 180.0))
        self.ukeire_weight = float(os.environ.get("V19_UKEIRE_W", 2.0))
        self.risk_weight = float(os.environ.get("V19_RISK_W", 0.0))

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

    def choose_discard(self) -> int:
        p = self.game.players[self.seat]
        counts14 = tuple(p.hand_counts)
        opts = discard_options(list(counts14))
        if not opts:
            return p.hand[-1]
        unseen = self._unseen_counts()
        penged = self._penged_by_others()
        min_sh = min(o["shanten"] for o in opts)
        turns = min(self.horizon, max(1, self.game.wall_remaining() // 4 + 1))
        best_t, best_score = None, -1e18
        for o in opts:
            t = o["tile"]
            h = _add(counts14, t, -1)
            s = o["shanten"]
            prob = _win_value(h, unseen, turns) if s <= min_sh + 1 else 0.0
            u = _wait_count(h, unseen)
            risk = 0.0
            if t != RED:
                if t in penged:
                    risk = 1.0
                else:
                    risk = {3: 0.4, 2: 0.2, 1: 0.05, 0: 0.0}.get(unseen[t], 0.4)
            score = -100.0 * s + self.prob_weight * prob + self.ukeire_weight * u - self.risk_weight * risk
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
