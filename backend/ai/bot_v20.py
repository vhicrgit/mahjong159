"""Rule bot v20: soft-shanten generalized-ukeire scoring."""

import os
from .bot_v10 import Bot as BotV10, _add, _ukeire, _second_step_value

RED = 27


class Bot(BotV10):
    def __init__(self, game, seat: int):
        super().__init__(game, seat)
        self.soft_shanten_weight = float(os.environ.get("V20_SHANTEN_W", 80.0))
        self.soft_ukeire_weight = float(os.environ.get("V20_UKEIRE_W", 1.0))
        self.soft_cont_weight = float(os.environ.get("V20_CONT_W", 0.5))
        self.soft_risk_weight = float(os.environ.get("V20_RISK_W", 0.0))

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
        best_t, best_score = None, -1e18
        for o in opts:
            t = o["tile"]
            h = _add(counts14, t, -1)
            s = o["shanten"]
            u = _ukeire(h, unseen)
            cont = _second_step_value(h, unseen) if s <= 2 else 0.0
            risk = 0.0
            if t != RED:
                if t in penged:
                    risk = 1.0
                else:
                    risk = {3: 0.4, 2: 0.2, 1: 0.05, 0: 0.0}.get(unseen[t], 0.4)
            score = -self.soft_shanten_weight * (1.0 - 0.5 * eg) * s + self.soft_ukeire_weight * u + self.soft_cont_weight * cont - self.soft_risk_weight * (1.0 + 1.5 * eg) * risk
            if score > best_score:
                best_score, best_t = score, t
        return best_t
