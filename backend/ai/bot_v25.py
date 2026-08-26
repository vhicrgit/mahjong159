"""Rule bot v25: v10 plus opponent-call lure tie-break."""

import os
from .bot_v10 import Bot as BotV10, _add, _ukeire, _second_step_value

RED = 27


class Bot(BotV10):
    def __init__(self, game, seat: int):
        super().__init__(game, seat)
        self.lure_weight = float(os.environ.get("V25_LURE_W", 3.0))

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
                score = self.ukeire_weight * u + self.cont_weight * cont
            if t != RED:
                if t in penged:
                    risk = 1.0
                    lure = 0.0
                else:
                    risk = {3: 0.4, 2: 0.2, 1: 0.05, 0: 0.0}.get(unseen[t], 0.4)
                    lure = unseen[t]
                score += self.lure_weight * lure
                score -= self.risk_weight * (1.0 + 1.5 * eg) * risk
            if score > best_score:
                best_score, best_t = score, t
        return best_t
