"""安康159 - Bot v1 vs v2 对战评估

用法: python -m backend.ai.bot_battle --games 4000 --v2-seats 0,1
v2 占 2 座位时, 若 v2 平均得分 > v1 且胜率份额 > 50% → v2 更强
"""

import argparse
import multiprocessing as mp

import numpy as np

from ..game.engine import Game
from .bot_v1 import Bot as BotV1
from .bot_v2 import Bot as BotV2
from .bot_v3 import Bot as BotV3


def play_one(args):
    seed, v2_seats, v2_bot = args
    B2 = {'v2': BotV2, 'v3': BotV3}[v2_bot]
    g = Game(seed=seed, human_seat=-1)
    bots = {i: (B2(g, i) if i in v2_seats else BotV1(g, i))
            for i in range(4)}
    guard = 0
    while g.phase != "game_over" and guard < 500:
        guard += 1
        if g.phase == "discard_wait":
            g.action_discard(g.turn, bots[g.turn].choose_discard())
        elif g.phase == "react_wait":
            s = list(g.pending_actions.keys())[0]
            b = bots[s]
            if g.pending_actions[s].get("gang") and \
                    b.decide_gang(g.last_discard, "ming"):
                g.action_gang(s)
            elif g.pending_actions[s].get("peng") and \
                    b.decide_peng(g.last_discard):
                g.action_peng(s)
            else:
                g.action_pass(s)
    return ([p.score_delta for p in g.players], g.winner)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=4000)
    ap.add_argument("--procs", type=int, default=100)
    ap.add_argument("--v2-seats", type=str, default="0,1",
                    help="v2 占据的座位, 逗号分隔; 2v2 消除座位偏差")
    ap.add_argument("--bot", type=str, default="v2",
                    choices=["v2", "v3"])
    args = ap.parse_args()

    v2_seats = {int(x) for x in args.v2_seats.split(",")}
    tasks = [(seed, v2_seats, args.bot) for seed in range(args.games)]

    with mp.Pool(args.procs) as pool:
        results = pool.map(play_one, tasks, chunksize=16)

    v2_score, v1_score, v2_wins, v1_wins, draws = 0.0, 0.0, 0, 0, 0
    for scores, winner in results:
        for seat in range(4):
            if seat in v2_seats:
                v2_score += scores[seat]
            else:
                v1_score += scores[seat]
        if winner is None:
            draws += 1
        elif winner in v2_seats:
            v2_wins += 1
        else:
            v1_wins += 1

    n = len(results)
    n_v2 = len(v2_seats)
    print(f"{n}局 2v2 (v2座位 {sorted(v2_seats)}):")
    print(f"  v2 场均 {v2_score/(n*n_v2):+.3f}  vs  "
          f"v1 场均 {v1_score/(n*(4-n_v2)):+.3f}")
    print(f"  v2 胜局份额 {v2_wins/(n-draws):.1%} vs "
          f"v1 {v1_wins/(n-draws):.1%} (流局 {draws})")
    print(f"  v2 胡牌率/座 {v2_wins/n/n_v2:.1%} vs "
          f"v1 {v1_wins/n/(4-n_v2):.1%}")


if __name__ == "__main__":
    main()
