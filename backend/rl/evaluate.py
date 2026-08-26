"""安康159 - 模型评估: NetBot vs 规则Bot

用法: python -m backend.rl.evaluate --model model_tiny.pt --games 200
"""

import argparse
import torch

from ..game.engine import Game
from ..ai.bot_v1 import Bot
from .net_bot import NetBot


def play_eval_game(seed: int, model_path: str, net_seat: int = 0):
    g = Game(seed=seed, human_seat=-1)
    net = NetBot(g, net_seat, model_path)
    rule_bots = {i: Bot(g, i) for i in range(4) if i != net_seat}

    def get_bot(s):
        return net if s == net_seat else rule_bots[s]

    guard = 0
    while g.phase != "game_over" and guard < 500:
        guard += 1
        if g.phase == "discard_wait":
            seat = g.turn
            g.action_discard(seat, get_bot(seat).choose_discard())
        elif g.phase == "react_wait":
            s = list(g.pending_actions.keys())[0]
            b = get_bot(s)
            if g.pending_actions[s].get("gang") and \
                    b.decide_gang(g.last_discard, "ming"):
                g.action_gang(s)
            elif g.pending_actions[s].get("peng") and \
                    b.decide_peng(g.last_discard):
                g.action_peng(s)
            else:
                g.action_pass(s)
    return g.players[net_seat].score_delta, g.winner == net_seat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--games", type=int, default=200)
    args = ap.parse_args()

    wins, total_score, huang = 0, 0.0, 0
    for i in range(args.games):
        g_seed = 100000 + i
        sd, won = play_eval_game(g_seed, args.model)
        total_score += sd
        if won:
            wins += 1
    print(f"模型 vs 3个规则Bot, {args.games}局:")
    print(f"  胜率: {wins}/{args.games} = {wins / args.games:.1%} "
          f"(随机基准约25%)")
    print(f"  场均得分: {total_score / args.games:+.2f}")


if __name__ == "__main__":
    main()
