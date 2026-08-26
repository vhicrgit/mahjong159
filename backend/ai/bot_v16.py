"""Rule bot v16: aggressive calls with open-hand-aware discards."""

from .bot_v8 import Bot as OpenAwareBot, open_shanten, _waiting_tiles_open


class Bot(OpenAwareBot):
    def decide_peng(self, tile: int) -> bool:
        p = self.game.players[self.seat]
        exposed = len(p.melds)
        before = open_shanten(p.hand_counts, exposed)
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
        return best_after <= before

    def decide_gang(self, tile: int, kind: str) -> bool:
        return True
