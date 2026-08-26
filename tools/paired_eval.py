"""Paired same-seed evaluation: bot A vs bot B, each at seat0 vs 3x v1.

Deterministic engine + deterministic bots -> per-seed paired binary outcomes.
Reports win rates, McNemar exact test on discordant pairs, and (optionally)
decision divergence of bot B vs shadow bot A inside bot B's games.
"""

import argparse
import multiprocessing as mp
import os
import sys
from math import comb

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.game.engine import Game
from backend.ai.bot_v1 import Bot as BotV1
from backend.ai.bot_eval import _make_bot

ARGS = None


def _play(seed, kind, param, shadow_kind=None, self_gang=False):
    g = Game(seed=seed, human_seat=-1)
    bots = {i: (_make_bot(kind, g, i, param) if i == 0 else BotV1(g, i))
            for i in range(4)}
    shadow = _make_bot(shadow_kind, g, 0, 0) if shadow_kind else None
    n_dec = n_div = 0
    guard = 0
    while g.phase != "game_over" and guard < 500:
        guard += 1
        if g.phase == "discard_wait":
            if self_gang and g.turn == 0:
                gang_taken = False
                for gt in g._gang_options(0):
                    kind_g = "an" if g.players[0].hand.count(gt) == 4 else "bu"
                    if bots[0].decide_gang(gt, kind_g):
                        g.action_gang(0, gt)
                        gang_taken = True
                        break
                if gang_taken:
                    continue
            t = bots[g.turn].choose_discard()
            if g.turn == 0 and shadow is not None:
                n_dec += 1
                if shadow.choose_discard() != t:
                    n_div += 1
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
    return g.winner == 0, g.players[0].score_delta, n_dec, n_div


def play_pair(seed):
    a = ARGS
    wa, sa, _, _ = _play(seed, a.bot_a, a.param_a, self_gang=a.self_gang)
    wb, sb, nd, nv = _play(seed, a.bot_b, a.param_b,
                           shadow_kind=(a.bot_a if a.shadow else None),
                           self_gang=a.self_gang)
    return wa, sa, wb, sb, nd, nv


def _init(args):
    global ARGS
    ARGS = args


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bot-a", default="v10")
    ap.add_argument("--bot-b", default="v28")
    ap.add_argument("--param-a", type=int, default=0)
    ap.add_argument("--param-b", type=int, default=0)
    ap.add_argument("--games", type=int, default=300)
    ap.add_argument("--procs", type=int, default=9)
    ap.add_argument("--seed0", type=int, default=980000)
    ap.add_argument("--shadow", action="store_true",
                    help="count bot-a decision divergence inside bot-b games")
    ap.add_argument("--self-gang", action="store_true",
                    help="allow seat0 an/bu gang via decide_gang (web-like protocol)")
    args = ap.parse_args()

    seeds = [args.seed0 + i for i in range(args.games)]
    with mp.Pool(args.procs, initializer=_init, initargs=(args,)) as pool:
        res = pool.map(play_pair, seeds, chunksize=1)

    n = len(res)
    wa = sum(r[0] for r in res)
    wb = sum(r[2] for r in res)
    sa = sum(r[1] for r in res) / n
    sb = sum(r[3] for r in res) / n
    n01 = sum(1 for r in res if (not r[0]) and r[2])   # B wins, A loses
    n10 = sum(1 for r in res if r[0] and (not r[2]))   # A wins, B loses
    m = n01 + n10
    if m:
        k = min(n01, n10)
        p = sum(comb(m, i) for i in range(k + 1)) / 2 ** m * 2
        p = min(1.0, p)
    else:
        p = 1.0
    print(f"paired {n} seeds (seed0={args.seed0}):")
    print(f"  A={args.bot_a}: win {wa/n:.1%}  avg {sa:+.3f}")
    print(f"  B={args.bot_b}: win {wb/n:.1%}  avg {sb:+.3f}")
    print(f"  discordant: B+A- {n01}, A+B- {n10}, McNemar exact p={p:.3f}")
    nd = sum(r[4] for r in res)
    nv = sum(r[5] for r in res)
    if nd:
        print(f"  divergence (B vs shadow A, in B games): {nv}/{nd} = {nv/nd:.1%}")


if __name__ == "__main__":
    main()
