"""安康159 - 单座位强度评估: 任意 Bot 在座位0 vs 3 个 v1 规则Bot

与神经网络评估(evaluate_vec)同口径: 1 个测试Bot + 3 个 v1 规则Bot,
胜率基准 25%。用于测 Bot v2 / v3 / Oracle 的绝对强度。

用法:
  python -m backend.ai.bot_eval --bot oracle --games 2000 --procs 100
"""

import argparse
import multiprocessing as mp

import numpy as np

from ..game.engine import Game
from .bot_v1 import Bot as BotV1


def _make_bot(kind, game, seat, param=0):
    if kind == "v1":
        return BotV1(game, seat)
    if kind == "v2":
        from .bot_v2 import Bot as B
        return B(game, seat)
    if kind == "v3":
        from .bot_v3 import Bot as B
        return B(game, seat, sim_rollouts=param or 16)
    if kind == "v4":
        from .bot_v4 import Bot as B
        import os
        return B(game, seat, worlds=param or 48,
                 horizon=int(os.environ.get("V4_H", 6)),
                 refine_scale=float(os.environ.get("V4_RS", 30)))
    if kind == "pimc":
        from .bot_pimc import Bot as B
        return B(game, seat, worlds=param or 32)
    if kind == "oracle":
        from .bot_oracle import Bot as B
        return B(game, seat, beam=param or 12)
    raise ValueError(kind)


def play_one(args):
    seed, kind, test_seat, param = args
    g = Game(seed=seed, human_seat=-1)
    bots = {i: (_make_bot(kind, g, i, param) if i == test_seat
                else BotV1(g, i)) for i in range(4)}
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
    return (g.players[test_seat].score_delta, g.winner == test_seat,
            g.winner is None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bot", type=str, default="v2",
                    choices=["v1", "v2", "v3", "v4", "oracle", "pimc"])
    ap.add_argument("--games", type=int, default=2000)
    ap.add_argument("--procs", type=int, default=100)
    ap.add_argument("--seat", type=int, default=0)
    ap.add_argument("--seed0", type=int, default=300000)
    ap.add_argument("--param", type=int, default=0,
                    help="oracle: beam 宽度; v3: rollout 数")
    args = ap.parse_args()

    tasks = [(args.seed0 + i, args.bot, args.seat, args.param)
             for i in range(args.games)]
    with mp.Pool(args.procs) as pool:
        results = pool.map(play_one, tasks, chunksize=8)

    scores = np.array([r[0] for r in results], dtype=np.float64)
    wins = sum(1 for r in results if r[1])
    draws = sum(1 for r in results if r[2])
    n = len(results)
    se = 1.96 * np.sqrt(wins / n * (1 - wins / n) / n)
    print(f"{args.bot} @座位{args.seat} vs 3×v1规则Bot, {n} 局:")
    print(f"  胜率 {wins/n:.1%} ±{se:.1%}(95%CI)  (基准 25%)")
    print(f"  场均 {scores.mean():+.3f}  流局 {draws/n:.1%}")


if __name__ == "__main__":
    main()
