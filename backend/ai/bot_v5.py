"""Rule bot v5: v2 analytic scoring plus target-probability tie breaking."""

from .bot_v2 import Bot as BotV2
from .bot_target import _target_chain, _chain_win_probability, _tuple_add, _wait_count
from ..rules.ting import discard_options

RED = 27


class Bot(BotV2):
    def __init__(self, game, seat: int, margin: float = 6.0):
        super().__init__(game, seat)
        self.margin = margin

    def choose_discard(self) -> int:
        p = self.game.players[self.seat]
        counts = p.hand_counts
        opts = discard_options(counts)
        if not opts:
            return p.hand[-1]

        visible = self._visible_counts()
        unseen = tuple(max(0, 4 - visible[t]) for t in range(28))
        penged = self._penged_tiles_by_others()
        eg = self._endgame_factor()

        base = {}
        best_base = -1e18
        for o in opts:
            t = o["tile"]
            wr = sum(max(0, 4 - visible[w]) for w in o["waits"])
            risk = 0.0
            if t != RED:
                if t in penged:
                    risk = 1.0
                else:
                    remain = max(0, 4 - visible[t])
                    risk = {3: 0.4, 2: 0.2, 1: 0.05, 0: 0.0}.get(remain, 0.4)
            sh_w = 100.0 * (1.0 - 0.5 * eg)
            risk_w = 25.0 * (1.0 + 1.5 * eg)
            score = -sh_w * o["shanten"] + 3.0 * wr - risk_w * risk
            base[t] = score
            if score > best_base:
                best_base = score

        candidates = [o for o in opts if base[o["tile"]] >= best_base - self.margin]
        if len(candidates) == 1:
            return candidates[0]["tile"]

        long_h = min(5, max(1, self.game.wall_remaining() // 4 + 1))
        short_h = min(3, long_h)
        best_t, best_score = None, -1e18
        for o in candidates:
            t = o["tile"]
            hand13 = _tuple_add(tuple(counts), t, -1)
            chain = _target_chain(hand13, unseen, 3, 1)
            p_short = _chain_win_probability(chain, short_h)
            p_long = _chain_win_probability(chain, long_h)
            progress = sum((i + 1) * u for i, (_, u) in enumerate(chain))
            score = base[t] + 180.0 * p_short + 60.0 * p_long + 8.0 * progress
            if score > best_score:
                best_score, best_t = score, t
        return best_t
