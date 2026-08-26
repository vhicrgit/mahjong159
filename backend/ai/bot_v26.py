"""Rule bot v26: v18 with conservative equal-shanten call search."""

import os

from .bot_v18 import Bot as BotV18
from .bot_v8 import open_shanten, _waiting_tiles_open, _useful_draws_open
from ..rules.win import shanten


class Bot(BotV18):
    def __init__(self, game, seat: int):
        super().__init__(game, seat)
        self.equal_call_min_waits = int(os.environ.get("V26_EQUAL_MIN_WAITS", 8))
        self.equal_call_min_delta = int(os.environ.get("V26_EQUAL_MIN_DELTA", 4))

    def _open_ukeire_for_counts(self, counts, exposed):
        if open_shanten(counts, exposed) == 0:
            useful = _waiting_tiles_open(tuple(counts), exposed)
        else:
            useful = _useful_draws_open(tuple(counts), exposed)
        unseen = self._unseen_counts()
        return sum(unseen[t] for t in useful)

    def decide_peng(self, tile: int) -> bool:
        p = self.game.players[self.seat]
        exposed = len(p.melds)
        before_s = shanten(p.hand_counts) if exposed == 0 else open_shanten(p.hand_counts, exposed)
        before_u = self._open_ukeire_for_counts(tuple(p.hand_counts), exposed)
        c = list(p.hand_counts)
        c[tile] -= 2
        best_after, best_u = 99, 0
        for discard, cnt in enumerate(c):
            if cnt <= 0:
                continue
            c[discard] -= 1
            after = open_shanten(c, exposed + 1)
            u = self._open_ukeire_for_counts(tuple(c), exposed + 1)
            if after < best_after or (after == best_after and u > best_u):
                best_after, best_u = after, u
            c[discard] += 1
        if best_after < before_s:
            return True
        if best_after == before_s and before_s <= 1:
            return best_u >= self.equal_call_min_waits and best_u >= before_u + self.equal_call_min_delta
        return False
