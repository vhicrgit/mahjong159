"""Rule bot v7: v2 discard policy, but never opens."""

from .bot_v2 import Bot as BotV2


class Bot(BotV2):
    def decide_peng(self, tile: int) -> bool:
        return False

    def decide_gang(self, tile: int, kind: str) -> bool:
        return False
