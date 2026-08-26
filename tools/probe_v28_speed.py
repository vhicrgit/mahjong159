"""Seat0-only timing probe: v28 (various k/beam) or v10 at seat 0, 3x v1 elsewhere."""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.game.engine import Game
from backend.ai.bot_v1 import Bot as V1


def make_seat0(name, game):
    if name == "v10":
        from backend.ai.bot_v10 import Bot
        return Bot(game, 0)
    if name.startswith("v28"):
        from backend.ai.bot_v28 import Bot
        b = Bot(game, 0)
        _, k, beam = name.split("-")
        b.k, b.beam = int(k), int(beam)
        return b
    if name.startswith("v29"):
        from backend.ai.bot_v29 import Bot
        b = Bot(game, 0)
        _, kcap, beam = name.split("-")
        b.k_cap, b.beam = int(kcap), int(beam)
        return b
    raise ValueError(name)


def probe(name, n_games=8, max_decisions=60):
    t_sum, n_dec = 0.0, 0
    for seed in range(5000, 5000 + n_games):
        g = Game(seed=seed, human_seat=-1)
        bots = {0: make_seat0(name, g), 1: V1(g, 1), 2: V1(g, 2), 3: V1(g, 3)}
        guard = 0
        while g.phase != "game_over" and guard < 500:
            guard += 1
            if g.phase == "discard_wait":
                if g.turn == 0:
                    t0 = time.time()
                    t = bots[0].choose_discard()
                    t_sum += time.time() - t0
                    n_dec += 1
                else:
                    t = bots[g.turn].choose_discard()
                g.action_discard(g.turn, t)
            elif g.phase == "react_wait":
                s = list(g.pending_actions.keys())[0]
                b = bots[s]
                if g.pending_actions[s].get("gang") and b.decide_gang(g.last_discard, "ming"):
                    g.action_gang(s)
                elif g.pending_actions[s].get("peng") and b.decide_peng(g.last_discard):
                    g.action_peng(s)
                else:
                    g.action_pass(s)
        if n_dec >= max_decisions:
            break
    print(f"{name}: {n_dec} decisions, {t_sum/max(n_dec,1)*1000:.1f} ms/decision", flush=True)


if __name__ == "__main__":
    for name in sys.argv[1:] or ["v10", "v28-2-16", "v28-3-32", "v28-4-64"]:
        probe(name)
