"""Rule bot v13: v2 discard policy with aggressive open-call pressure."""

from .bot_v2 import Bot as BotV2
from ..rules.win import shanten


class Bot(BotV2):
    def decide_peng(self, tile: int) -> bool:
        p = self.game.players[self.seat]
        before = shanten(p.hand_counts)
        c = list(p.hand_counts)
        c[tile] -= 2
        after = shanten(c)
        return after <= before + 1

    def decide_gang(self, tile: int, kind: str) -> bool:
        return True
