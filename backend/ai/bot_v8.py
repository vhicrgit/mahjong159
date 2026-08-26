"""Rule bot v8: open-hand-aware shanten and ukeire scoring."""

from functools import lru_cache

from ..rules.win import _dfs_cached, shanten
from ..rules.ting import waiting_tiles

RED = 27


def open_shanten(counts: list[int] | tuple[int, ...], exposed_melds: int) -> int:
    need = max(0, 4 - exposed_melds)
    red = counts[RED]
    m, t, p = _dfs_cached(tuple(counts[:27]), red)
    m = min(m, need)
    t = min(t, need - m)
    p = min(p, 1)
    return 2 * need - 2 * m - t - p


@lru_cache(maxsize=300000)
def _open_shanten_cached(counts: tuple[int, ...], exposed_melds: int) -> int:
    return open_shanten(counts, exposed_melds)


@lru_cache(maxsize=300000)
def _waiting_tiles_open(counts: tuple[int, ...], exposed_melds: int) -> tuple[int, ...]:
    waits = []
    for tile in range(28):
        if counts[tile] >= 4:
            continue
        c = list(counts)
        c[tile] += 1
        if open_shanten(c, exposed_melds) == -1:
            waits.append(tile)
    return tuple(waits)


@lru_cache(maxsize=300000)
def _useful_draws_open(counts: tuple[int, ...], exposed_melds: int) -> tuple[int, ...]:
    base = open_shanten(counts, exposed_melds)
    out = []
    for tile in range(28):
        if counts[tile] >= 4:
            continue
        c = list(counts)
        c[tile] += 1
        if open_shanten(c, exposed_melds) < base:
            out.append(tile)
    return tuple(out)


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

    def _score_hand13(self, counts: tuple[int, ...], visible: list[int],
                      unseen: tuple[int, ...], exposed: int) -> tuple[int, int, int]:
        s = _open_shanten_cached(counts, exposed)
        if s == 0:
            useful = _waiting_tiles_open(counts, exposed)
        else:
            useful = _useful_draws_open(counts, exposed)
        remain = sum(unseen[t] for t in useful)
        raw_types = len(useful)
        return s, remain, raw_types

    def choose_discard(self) -> int:
        p = self.game.players[self.seat]
        counts14 = p.hand_counts
        visible = self._visible_counts()
        unseen = self._unseen_counts()
        exposed = len(p.melds)
        penged = self._penged_by_others()
        eg = self._endgame_factor()
        sh_w = 100.0 * (1.0 - 0.5 * eg)
        risk_w = 25.0 * (1.0 + 1.5 * eg)
        best_t, best_score = None, -1e18
        for t in range(28):
            if counts14[t] <= 0:
                continue
            c = list(counts14)
            c[t] -= 1
            s, remain, raw_types = self._score_hand13(tuple(c), visible, unseen, exposed)
            risk = 0.0
            if t != RED:
                if t in penged:
                    risk = 1.0
                else:
                    risk = {3: 0.4, 2: 0.2, 1: 0.05, 0: 0.0}.get(unseen[t], 0.4)
            score = -sh_w * s + 2.5 * remain + 0.5 * raw_types - risk_w * risk
            if score > best_score:
                best_score, best_t = score, t
        return best_t if best_t is not None else p.hand[-1]

    def decide_peng(self, tile: int) -> bool:
        p = self.game.players[self.seat]
        exposed = len(p.melds)
        before = open_shanten(p.hand_counts, exposed)
        before_waits = len(_waiting_tiles_open(tuple(p.hand_counts), exposed)) if before == 0 else 0
        c = list(p.hand_counts)
        c[tile] -= 2
        best_after, best_waits = 99, 0
        for discard, cnt in enumerate(c):
            if cnt <= 0:
                continue
            c[discard] -= 1
            after = open_shanten(c, exposed + 1)
            waits = len(_waiting_tiles_open(tuple(c), exposed + 1)) if after == 0 else 0
            if after < best_after or (after == best_after and waits > best_waits):
                best_after, best_waits = after, waits
            c[discard] += 1
        if best_after < before:
            return True
        return before == 0 and best_after == 0 and best_waits >= before_waits

    def decide_gang(self, tile: int, kind: str) -> bool:
        p = self.game.players[self.seat]
        exposed = len(p.melds)
        before = open_shanten(p.hand_counts, exposed)
        c = list(p.hand_counts)
        if kind == "ming":
            c[tile] -= 3
            exposed += 1
        elif kind == "an":
            c[tile] -= 4
            exposed += 1
        else:
            c[tile] -= 1
        after = open_shanten(c, exposed)
        return not (before == 0 and after > 0)
