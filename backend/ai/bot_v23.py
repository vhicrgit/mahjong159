"""Rule bot v23: v10 with open-aware discard only after melding."""

from .bot_v10 import Bot as BotV10
from .bot_v8 import open_shanten, _waiting_tiles_open, _useful_draws_open

RED = 27


class Bot(BotV10):
    def _choose_open_discard(self) -> int:
        p = self.game.players[self.seat]
        counts14 = tuple(p.hand_counts)
        visible = self._visible_counts()
        unseen = tuple(max(0, 4 - visible[t]) for t in range(28))
        exposed = len(p.melds)
        penged = self._penged_by_others()
        eg = self._endgame_factor()
        sh_w = self.shanten_weight * (1.0 - 0.5 * eg)
        risk_w = self.risk_weight * (1.0 + 1.5 * eg)
        best_t, best_score = None, -1e18
        for t, cnt in enumerate(counts14):
            if cnt <= 0:
                continue
            h = list(counts14)
            h[t] -= 1
            s = open_shanten(h, exposed)
            useful = _waiting_tiles_open(tuple(h), exposed) if s == 0 else _useful_draws_open(tuple(h), exposed)
            u = sum(unseen[x] for x in useful)
            risk = 0.0
            if t != RED:
                if t in penged:
                    risk = 1.0
                else:
                    risk = {3: 0.4, 2: 0.2, 1: 0.05, 0: 0.0}.get(unseen[t], 0.4)
            score = -sh_w * s + self.ukeire_weight * u - risk_w * risk
            if score > best_score:
                best_score, best_t = score, t
        return best_t if best_t is not None else p.hand[-1]

    def choose_discard(self) -> int:
        if self.game.players[self.seat].melds:
            return self._choose_open_discard()
        return super().choose_discard()
