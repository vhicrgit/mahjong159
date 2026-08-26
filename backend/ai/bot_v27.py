"""Rule bot v27: v10 with turn-steal reaction calls that preserve open shanten."""

from .bot_v10 import Bot as BotV10
from .bot_v8 import open_shanten, _waiting_tiles_open, _useful_draws_open
from ..rules.win import shanten


class Bot(BotV10):
    def _open_ukeire(self, counts, exposed):
        useful = _waiting_tiles_open(tuple(counts), exposed) if open_shanten(counts, exposed) == 0 else _useful_draws_open(tuple(counts), exposed)
        unseen = self._unseen_counts()
        return sum(unseen[t] for t in useful)

    def _best_after_peng(self, tile: int):
        p = self.game.players[self.seat]
        exposed = len(p.melds) + 1
        c = list(p.hand_counts)
        c[tile] -= 2
        best_s, best_u = 99, -1
        for discard, cnt in enumerate(c):
            if cnt <= 0:
                continue
            c[discard] -= 1
            s = open_shanten(c, exposed)
            u = self._open_ukeire(tuple(c), exposed)
            if s < best_s or (s == best_s and u > best_u):
                best_s, best_u = s, u
            c[discard] += 1
        return best_s, best_u

    def decide_peng(self, tile: int) -> bool:
        p = self.game.players[self.seat]
        exposed = len(p.melds)
        before_s = shanten(p.hand_counts) if exposed == 0 else open_shanten(p.hand_counts, exposed)
        before_u = self._open_ukeire(tuple(p.hand_counts), exposed)
        after_s, after_u = self._best_after_peng(tile)
        if after_s < before_s:
            return True
        if after_s == before_s and before_s <= 2:
            skipped = (self.seat - self.game.last_discarder) % 4 - 1
            return skipped > 0 and after_u + 2 >= before_u
        return False
