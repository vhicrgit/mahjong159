"""Rule bot v24: v10 plus shape and red-preservation tie-breaks."""

import os
from functools import lru_cache

from .bot_v10 import Bot as BotV10, _add, _ukeire, _second_step_value

RED = 27


@lru_cache(maxsize=300000)
def _shape_value(hand13: tuple[int, ...]) -> float:
    value = 0.0
    for base in (0, 9, 18):
        suit = hand13[base:base + 9]
        for i, n in enumerate(suit):
            if n >= 2:
                value += 0.30
            if n >= 3:
                value += 0.60
            if i <= 6 and suit[i] and suit[i + 1] and suit[i + 2]:
                value += 0.80
            if i <= 7 and suit[i] and suit[i + 1]:
                value += 0.40
            if i <= 6 and suit[i] and suit[i + 2]:
                value += 0.25
            if n == 1 and (i == 0 or i == 8):
                value -= 0.08
    value += 0.80 * hand13[RED]
    return value


class Bot(BotV10):
    def __init__(self, game, seat: int):
        super().__init__(game, seat)
        self.shape_weight = float(os.environ.get("V24_SHAPE_W", 1.0))
        self.red_discard_penalty = float(os.environ.get("V24_RED_PENALTY", 4.0))

    def choose_discard(self) -> int:
        p = self.game.players[self.seat]
        counts14 = tuple(p.hand_counts)
        from ..rules.ting import discard_options
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
                score = self.ukeire_weight * u + self.cont_weight * cont + self.shape_weight * _shape_value(h)
            if t == RED:
                score -= self.red_discard_penalty
            else:
                if t in penged:
                    risk = 1.0
                else:
                    risk = {3: 0.4, 2: 0.2, 1: 0.05, 0: 0.0}.get(unseen[t], 0.4)
                score -= self.risk_weight * (1.0 + 1.5 * eg) * risk
            if score > best_score:
                best_score, best_t = score, t
        return best_t
