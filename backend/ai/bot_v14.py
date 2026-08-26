"""Rule bot v14: v10 discard with conservative open-call handling."""

import os

from .bot_v10 import Bot as BotV10
from .bot_v8 import open_shanten, _waiting_tiles_open
from ..rules.win import shanten


class Bot(BotV10):
    def __init__(self, game, seat: int):
        super().__init__(game, seat)
        self.min_waits_after_peng = int(os.environ.get("V14_MIN_WAITS", 4))
        self.allow_equal_shanten = os.environ.get("V14_ALLOW_EQUAL", "0") == "1"

    def decide_peng(self, tile: int) -> bool:
        p = self.game.players[self.seat]
        exposed = len(p.melds)
        before = shanten(p.hand_counts) if exposed == 0 else open_shanten(p.hand_counts, exposed)
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
        if not self.allow_equal_shanten:
            return False
        return best_after == 0 and before <= 1 and best_waits >= self.min_waits_after_peng
