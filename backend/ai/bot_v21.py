"""Rule bot v21: v10 discard policy with no calls."""

from .bot_v10 import Bot as BotV10


class Bot(BotV10):
    def decide_peng(self, tile: int) -> bool:
        return False

    def decide_gang(self, tile: int, kind: str) -> bool:
        return False
