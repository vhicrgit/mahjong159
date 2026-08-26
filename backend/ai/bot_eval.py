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
    if kind == "v5":
        from .bot_v5 import Bot as B
        return B(game, seat, margin=param or 6)
    if kind == "v6":
        from .bot_v6 import Bot as B
        return B(game, seat)
    if kind == "v7":
        from .bot_v7 import Bot as B
        return B(game, seat)
    if kind == "v8":
        from .bot_v8 import Bot as B
        return B(game, seat)
    if kind == "v9":
        from .bot_v9 import Bot as B
        return B(game, seat)
    if kind == "v10":
        from .bot_v10 import Bot as B
        return B(game, seat)
    if kind == "v11":
        from .bot_v11 import Bot as B
        return B(game, seat)
    if kind == "v12":
        from .bot_v12 import Bot as B
        return B(game, seat, worlds=param or None)
    if kind == "v13":
        from .bot_v13 import Bot as B
        return B(game, seat)
    if kind == "v14":
        from .bot_v14 import Bot as B
        return B(game, seat)
    if kind == "v15":
        from .bot_v15 import Bot as B
        return B(game, seat)
    if kind == "v16":
        from .bot_v16 import Bot as B
        return B(game, seat)
    if kind == "v17":
        from .bot_v17 import Bot as B
        return B(game, seat)
    if kind == "v18":
        from .bot_v18 import Bot as B
        return B(game, seat)
    if kind == "v19":
        from .bot_v19 import Bot as B
        return B(game, seat)
    if kind == "v20":
        from .bot_v20 import Bot as B
        return B(game, seat)
    if kind == "v21":
        from .bot_v21 import Bot as B
        return B(game, seat)
    if kind == "v22":
        from .bot_v22 import Bot as B
        return B(game, seat)
    if kind == "v23":
        from .bot_v23 import Bot as B
        return B(game, seat)
    if kind == "v24":
        from .bot_v24 import Bot as B
        return B(game, seat)
    if kind == "v25":
        from .bot_v25 import Bot as B
        return B(game, seat)
    if kind == "v26":
        from .bot_v26 import Bot as B
        return B(game, seat)
    if kind == "v27":
        from .bot_v27 import Bot as B
        return B(game, seat)
    if kind == "v28":
        from .bot_v28 import Bot as B
        return B(game, seat)
    if kind == "v29":
        from .bot_v29 import Bot as B
        return B(game, seat)
    if kind == "v30":
        from .bot_v30 import Bot as B
        return B(game, seat)
    if kind == "v31":
        from .bot_v31 import Bot as B
        return B(game, seat)
    if kind == "target":
        from .bot_target import Bot as B
        return B(game, seat)
    if kind == "pimc":
        from .bot_pimc import Bot as B
        return B(game, seat, worlds=param or 32)
    if kind == "oracle":
        from .bot_oracle import Bot as B
        return B(game, seat, beam=param or 12)
    if kind == "cheat_full":
        from .bot_cheat import Bot as B
        return B(game, seat, wall_lookahead=-1, see_opponents=True,
                 beam=param or 4, rollout=True)
    if kind == "cheat_wall":
        from .bot_cheat import Bot as B
        return B(game, seat, wall_lookahead=param or 32,
                 see_opponents=False, beam=12, rollout=False)
    if kind == "cheat_opp":
        from .bot_cheat import Bot as B
        return B(game, seat, wall_lookahead=param or 32,
                 see_opponents=True, beam=12, rollout=False)
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
                    choices=["v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31", "target", "oracle", "pimc",
                             "cheat_full", "cheat_wall", "cheat_opp"])
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
