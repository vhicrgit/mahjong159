import sys
import time

sys.path.insert(0, "/home/zuofengrui.zfr/mahjong159")

from backend.game.engine import Game
from backend.ai.bot_v28 import Bot, _win_prob, _ukeire, _add
from backend.rules.win import shanten_cached


def bench(k: int, n_games: int = 3, max_decisions: int = 12):
    t0 = time.time()
    n_dec = 0
    for seed in range(1000, 1000 + n_games):
        g = Game(seed=seed, human_seat=-1)
        bots = {i: Bot(g, i) for i in range(4)}
        bots[0].k = k
        guard = 0
        while g.phase != "game_over" and guard < 500:
            guard += 1
            if g.phase == "discard_wait":
                if g.turn == 0:
                    n_dec += 1
                    if n_dec > max_decisions:
                        break
                g.action_discard(g.turn, bots[g.turn].choose_discard())
            elif g.phase == "react_wait":
                s = list(g.pending_actions.keys())[0]
                b = bots[s]
                if g.pending_actions[s].get("gang") and b.decide_gang(g.last_discard, "ming"):
                    g.action_gang(s)
                elif g.pending_actions[s].get("peng") and b.decide_peng(g.last_discard):
                    g.action_peng(s)
                else:
                    g.action_pass(s)
        if n_dec > max_decisions:
            break
    dt = time.time() - t0
    print(f"k={k}: {n_dec} decisions in {dt:.1f}s -> {dt/max(n_dec,1)*1000:.0f} ms/decision")


if __name__ == "__main__":
    for k in (2, 3, 4):
        bench(k)
