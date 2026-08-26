"""Rule bot v9: target-probability discard policy, but never opens."""

from .bot_target import Bot as TargetBot


class Bot(TargetBot):
    def decide_peng(self, tile: int) -> bool:
        return False

    def decide_gang(self, tile: int, kind: str) -> bool:
        return False
