"""Rule bot v22: strict min-shanten plus draw-then-discard effective ukeire."""

import os
from .bot_v17 import _add, _effective_stats
from ..rules.win import shanten
from ..rules.ting import discard_options

RED = 27


class Bot:
    def __init__(self, game, seat: int):
        self.game = game
        self.seat = seat
        self.eff_weight = float(os.environ.get("V22_EFF_W", 2.0))
        self.gain_weight = float(os.environ.get("V22_GAIN_W", 20.0))
        self.wait_weight = float(os.environ.get("V22_WAIT_W", 0.2))
        self.risk_weight = float(os.environ.get("V22_RISK_W", 0.0))

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
        best_t, best_score = None, -1e18
        for o in opts:
            t = o["tile"]
            h = _add(counts14, t, -1)
            if o["shanten"] > min_sh:
                score = -1000.0 - 100.0 * o["shanten"]
            else:
                eff, gain, waits = _effective_stats(h, unseen)
                score = self.eff_weight * eff + self.gain_weight * gain + self.wait_weight * waits
            if t != RED:
                if t in penged:
                    risk = 1.0
                else:
                    risk = {3: 0.4, 2: 0.2, 1: 0.05, 0: 0.0}.get(unseen[t], 0.4)
                score -= self.risk_weight * risk
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
