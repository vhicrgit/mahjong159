"""Count peng-decision divergence between v10 rule and v30 winp rule at seat0 (parallel)."""

import multiprocessing as mp
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.game.engine import Game
from backend.ai.bot_v1 import Bot as V1
from backend.ai.bot_v10 import Bot as V10
from backend.ai.bot_v30 import Bot as V30


def one(seed):
    g = Game(seed=seed, human_seat=-1)
    bots = {0: V10(g, 0), 1: V1(g, 1), 2: V1(g, 2), 3: V1(g, 3)}
    b30 = V30(g, 0)
    n_opp = n_div = n_v10_yes = n_v30_yes = 0
    guard = 0
    while g.phase != "game_over" and guard < 500:
        guard += 1
        if g.phase == "discard_wait":
            g.action_discard(g.turn, bots[g.turn].choose_discard())
        elif g.phase == "react_wait":
            s = list(g.pending_actions.keys())[0]
            b = bots[s]
            if s == 0 and g.pending_actions[s].get("peng") and not g.pending_actions[s].get("gang"):
                d10 = b.decide_peng(g.last_discard)
                d30 = b30.decide_peng(g.last_discard)
                n_opp += 1
                n_v10_yes += d10
                n_v30_yes += d30
                n_div += d10 != d30
            if g.pending_actions[s].get("gang") and b.decide_gang(g.last_discard, "ming"):
                g.action_gang(s)
            elif g.pending_actions[s].get("peng") and b.decide_peng(g.last_discard):
                g.action_peng(s)
            else:
                g.action_pass(s)
    return n_opp, n_div, n_v10_yes, n_v30_yes


if __name__ == "__main__":
    games = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    procs = int(sys.argv[2]) if len(sys.argv) > 2 else 64
    with mp.Pool(procs) as pool:
        rs = pool.map(one, range(7000, 7000 + games), chunksize=1)
    n_opp = sum(r[0] for r in rs)
    n_div = sum(r[1] for r in rs)
    y10 = sum(r[2] for r in rs)
    y30 = sum(r[3] for r in rs)
    print(f"games {games}, peng opportunities: {n_opp}, v10 yes: {y10}, "
          f"v30 yes: {y30}, divergence: {n_div} ({n_div/max(n_opp,1):.1%})")
