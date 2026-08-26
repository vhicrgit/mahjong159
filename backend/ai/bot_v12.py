"""Rule bot v12: belief-sampled full-game rollout against v1 opponents."""

import os
import random
from copy import deepcopy

from ..rules.win import shanten
from ..rules.ting import discard_options
from .bot_v1 import Bot as BotV1
from .bot_v10 import Bot as BotV10
from .bot_v2 import Bot as BotV2

RED = 27


class Bot:
    def __init__(self, game, seat: int, worlds: int | None = None,
                 margin: float | None = None):
        self.game = game
        self.seat = seat
        self.worlds = worlds if worlds is not None else int(os.environ.get("BELIEF_WORLDS", 6))
        self.margin = margin if margin is not None else float(os.environ.get("BELIEF_MARGIN", 8.0))
        self.rng = random.Random((id(game) * 31 + seat * 9973) & 0xffffffff)
        self.fallback = BotV2(game, seat)

    def _visible_counts(self) -> list[int]:
        visible = [0] * 28
        me = self.game.players[self.seat]
        for t, n in enumerate(me.hand_counts):
            visible[t] += n
        for q in self.game.players:
            for t in q.discards:
                visible[t] += 1
            for m in q.melds:
                visible[m["tile"]] += 3 if m["type"] == "peng" else 4
        return visible

    def _candidate_discards(self):
        p = self.game.players[self.seat]
        counts = p.hand_counts
        opts = discard_options(counts)
        if not opts:
            return [p.hand[-1]]
        visible = self._visible_counts()
        penged = set()
        for q in self.game.players:
            if q.seat == self.seat:
                continue
            for m in q.melds:
                if m["type"] == "peng":
                    penged.add(m["tile"])
        eg = max(0.0, min(1.0, (60 - self.game.wall_remaining()) / 60.0))
        scored = []
        best = -1e18
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
            score = -100.0 * (1.0 - 0.5 * eg) * o["shanten"] + 3.0 * wr - 25.0 * (1.0 + 1.5 * eg) * risk
            scored.append((score, t))
            best = max(best, score)
        return [t for score, t in scored if score >= best - self.margin]

    def _sample_world(self):
        g = deepcopy(self.game)
        visible = [0] * 28
        me = g.players[self.seat]
        for t, n in enumerate(me.hand_counts):
            visible[t] += n
        for q in g.players:
            for t in q.discards:
                visible[t] += 1
            for m in q.melds:
                visible[m["tile"]] += 3 if m["type"] == "peng" else 4
        pool = []
        for t in range(28):
            pool.extend([t] * max(0, 4 - visible[t]))
        self.rng.shuffle(pool)
        pos = 0
        for q in g.players:
            if q.seat == self.seat:
                continue
            n = len(q.hand)
            q.hand = sorted(pool[pos:pos + n])
            pos += n
        g.wall = pool[pos:pos + len(g.wall)]
        return g

    def _simulate(self, game) -> float:
        bots = {i: BotV1(game, i) for i in range(4) if i != self.seat}
        me = BotV10(game, self.seat)
        guard = 0
        while game.phase != "game_over" and guard < 500:
            guard += 1
            if game.phase == "discard_wait":
                if game.turn == self.seat:
                    tile = me.choose_discard()
                else:
                    tile = bots[game.turn].choose_discard()
                game.action_discard(game.turn, tile)
            elif game.phase == "react_wait":
                s = list(game.pending_actions.keys())[0]
                if s == self.seat:
                    b = me
                else:
                    b = bots[s]
                if game.pending_actions[s].get("gang") and b.decide_gang(game.last_discard, "ming"):
                    game.action_gang(s)
                elif game.pending_actions[s].get("peng") and b.decide_peng(game.last_discard):
                    game.action_peng(s)
                else:
                    game.action_pass(s)
        if game.winner == self.seat:
            return 100.0 + game.players[self.seat].score_delta
        if game.winner is None:
            return game.players[self.seat].score_delta
        return -100.0 + game.players[self.seat].score_delta

    def choose_discard(self) -> int:
        p = self.game.players[self.seat]
        candidates = self._candidate_discards()
        if len(candidates) == 1:
            return candidates[0]
        scores = {t: 0.0 for t in candidates}
        for _ in range(self.worlds):
            sampled = self._sample_world()
            for t in candidates:
                g2 = deepcopy(sampled)
                try:
                    g2.action_discard(self.seat, t)
                except Exception:
                    scores[t] -= 1e9
                    continue
                scores[t] += self._simulate(g2)
        return max(scores.items(), key=lambda kv: kv[1])[0]

    def decide_peng(self, tile: int) -> bool:
        return self.fallback.decide_peng(tile)

    def decide_gang(self, tile: int, kind: str) -> bool:
        return self.fallback.decide_gang(tile, kind)
