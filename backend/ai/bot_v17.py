"""Rule bot v17: draw-then-discard effective ukeire."""

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
def _best_after_draw(hand13: tuple[int, ...], unseen: tuple[int, ...], draw: int) -> tuple[int, int, tuple[int, ...]]:
    hand14 = _add(hand13, draw, 1)
    if is_win(list(hand14)):
        return -1, 0, hand13
    best_s, best_waits, best_h = 99, 0, hand13
    unseen2 = _add(unseen, draw, -1) if unseen[draw] > 0 else unseen
    for discard, cnt in enumerate(hand14):
        if cnt <= 0:
            continue
        h = _add(hand14, discard, -1)
        s = _shanten(h)
        waits = _wait_count(h, unseen2)
        if s < best_s or (s == best_s and waits > best_waits):
            best_s, best_waits, best_h = s, waits, h
    return best_s, best_waits, best_h


@lru_cache(maxsize=500000)
def _effective_stats(hand13: tuple[int, ...], unseen: tuple[int, ...]) -> tuple[int, float, float]:
    base_s = _shanten(hand13)
    total = sum(unseen)
    eff = 0
    exp_gain = 0.0
    exp_waits = 0.0
    if total <= 0:
        return 0, 0.0, 0.0
    for draw, n in enumerate(unseen):
        if n <= 0:
            continue
        best_s, best_waits, _ = _best_after_draw(hand13, unseen, draw)
        if best_s < base_s:
            eff += n
            exp_gain += n / total * (base_s - best_s)
        exp_waits += n / total * best_waits
    return eff, exp_gain, exp_waits


@lru_cache(maxsize=300000)
def _two_step_value(hand13: tuple[int, ...], unseen: tuple[int, ...]) -> float:
    total = sum(unseen)
    if total <= 0:
        return 0.0
    base_s = _shanten(hand13)
    value = 0.0
    for draw, n in enumerate(unseen):
        if n <= 0:
            continue
        best_s, best_waits, best_h = _best_after_draw(hand13, unseen, draw)
        if best_s < 0:
            value += n / total * 80.0
        elif best_s < base_s:
            eff2, gain2, waits2 = _effective_stats(best_h, unseen)
            value += n / total * (28.0 * (base_s - best_s) + 0.30 * eff2 + 8.0 * gain2 + 0.05 * waits2)
        elif best_s == 0:
            value += n / total * (0.20 * best_waits)
    return value


@lru_cache(maxsize=300000)
def _shape_value(hand13: tuple[int, ...]) -> float:
    value = 0.0
    for base in (0, 9, 18):
        suit = hand13[base:base + 9]
        for i, n in enumerate(suit):
            if n >= 2:
                value += 0.35
            if n >= 3:
                value += 0.8
            if i <= 6 and suit[i] and suit[i + 1] and suit[i + 2]:
                value += 0.8
            if i <= 7 and suit[i] and suit[i + 1]:
                value += 0.35
            if i <= 6 and suit[i] and suit[i + 2]:
                value += 0.2
    value += 0.5 * hand13[RED]
    return value


class Bot:
    def __init__(self, game, seat: int):
        self.game = game
        self.seat = seat
        self.ukeire_weight = float(os.environ.get("V17_UKEIRE_W", 1.6))
        self.two_step_weight = float(os.environ.get("V17_TWO_W", 1.0))
        self.shape_weight = float(os.environ.get("V17_SHAPE_W", 0.5))
        self.risk_weight = float(os.environ.get("V17_RISK_W", 20.0))

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
        sh_w = 100.0 * (1.0 - 0.5 * eg)
        risk_w = self.risk_weight * (1.0 + 1.5 * eg)
        best_t, best_score = None, -1e18
        for o in opts:
            t = o["tile"]
            h = _add(counts14, t, -1)
            s = o["shanten"]
            eff, gain, waits = _effective_stats(h, unseen)
            cont = _two_step_value(h, unseen) if s <= 2 else 0.0
            risk = 0.0
            if t != RED:
                if t in penged:
                    risk = 1.0
                else:
                    risk = {3: 0.4, 2: 0.2, 1: 0.05, 0: 0.0}.get(unseen[t], 0.4)
            score = (-sh_w * s + self.ukeire_weight * eff + 10.0 * gain
                     + self.two_step_weight * cont + 0.2 * waits
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
