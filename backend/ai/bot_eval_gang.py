"""Bot evaluation harness that also lets bots take legal concealed/add-gang actions."""

import argparse
import multiprocessing as mp

import numpy as np

from ..game.engine import Game
from .bot_eval import _make_bot
from .bot_v1 import Bot as BotV1


def _gang_kind(game, seat: int, tile: int) -> str:
    p = game.players[seat]
    if p.hand.count(tile) == 4:
        return "an"
    return "bu"


def _take_self_gang_if_wanted(game, bot) -> bool:
    opts = game._gang_options(game.turn)
    for tile in opts:
        if bot.decide_gang(tile, _gang_kind(game, game.turn, tile)):
            game.action_gang(game.turn, tile)
            return True
    return False


def play_one(args):
    seed, kind, test_seat, param = args
    g = Game(seed=seed, human_seat=-1)
    bots = {i: (_make_bot(kind, g, i, param) if i == test_seat
                else BotV1(g, i)) for i in range(4)}
    guard = 0
    while g.phase != "game_over" and guard < 700:
        guard += 1
        if g.phase == "discard_wait":
            if _take_self_gang_if_wanted(g, bots[g.turn]):
                continue
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
    return (g.players[test_seat].score_delta, g.winner == test_seat,
            g.winner is None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bot", type=str, default="v2",
                    choices=["v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v17", "v18", "v19",
                             "target", "oracle", "pimc",
                             "cheat_full", "cheat_wall", "cheat_opp"])
    ap.add_argument("--games", type=int, default=2000)
    ap.add_argument("--procs", type=int, default=100)
    ap.add_argument("--seat", type=int, default=0)
    ap.add_argument("--seed0", type=int, default=300000)
    ap.add_argument("--param", type=int, default=0)
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
    print(f"{args.bot} @座位{args.seat} vs 3×v1规则Bot, {n} 局 (self-gang enabled):")
    print(f"  胜率 {wins/n:.1%} ±{se:.1%}(95%CI)  (基准 25%)")
    print(f"  场均 {scores.mean():+.3f}  流局 {draws/n:.1%}")


if __name__ == "__main__":
    main()
