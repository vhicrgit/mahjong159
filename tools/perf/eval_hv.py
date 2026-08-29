"""评估 HV(牌型价值) bot: vs 3×v1n 和 vs 3×v31n, 并以同种子 v31n 作配对参照。

每个 seed 打三局(同一牌墙):
  A: HV@座0 vs 3×v1n
  B: v31n@座0 vs 3×v1n   (参照, 同种子配对)
  C: HV@座0 vs 3×v31n
报告: 胜率/场均/配对 McNemar 精确检验(A vs B 的胜负差异)。

用法: python -m tools.perf.eval_hv --games 1500 --procs 8
"""

import argparse
import math
import multiprocessing as mp

import numpy as np

from backend.ai.bot_hv import Bot as HVBot
from backend.ai.bot_native import NativeV1, NativeV31
from backend.game.engine import Game


def play(seed, hero_cls, opp_cls):
    g = Game(seed=seed, human_seat=-1)
    bots = {i: (hero_cls(g, i) if i == 0 else opp_cls(g, i))
            for i in range(4)}
    guard = 0
    while g.phase != "game_over" and guard < 500:
        guard += 1
        if g.phase == "discard_wait":
            g.action_discard(g.turn, bots[g.turn].choose_discard())
        else:
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
    return (1 if g.winner == 0 else 0, g.players[0].score_delta)


def work(seed):
    a = play(seed, HVBot, NativeV1)
    b = play(seed, NativeV31, NativeV1)
    c = play(seed, HVBot, NativeV31)
    return a, b, c


def mcnemar(wins_a, wins_b):
    """精确双侧 McNemar: b=A胜B负, c=B胜A负。"""
    b = sum(1 for x, y in zip(wins_a, wins_b) if x == 1 and y == 0)
    c = sum(1 for x, y in zip(wins_a, wins_b) if x == 0 and y == 1)
    n = b + c
    if n == 0:
        return b, c, 1.0
    k = min(b, c)
    p = 2 * min(1.0, sum(math.comb(n, i) for i in range(0, k + 1))
                / 2 ** n)
    return b, c, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=1500)
    ap.add_argument("--seed0", type=int, default=970000)
    ap.add_argument("--procs", type=int, default=8)
    args = ap.parse_args()

    seeds = [args.seed0 + i for i in range(args.games)]
    with mp.Pool(args.procs) as pool:
        rets = pool.map(work, seeds, chunksize=4)

    A = [r[0] for r in rets]   # HV vs v1n
    B = [r[1] for r in rets]   # v31n vs v1n
    C = [r[2] for r in rets]   # HV vs v31n

    def stat(rows):
        w = np.array([r[0] for r in rows], dtype=float)
        s = np.array([r[1] for r in rows], dtype=float)
        n = len(w)
        return (w.mean() * 100, 1.96 * math.sqrt(w.mean() * (1 - w.mean())
                / n) * 100, s.mean())

    for name, rows in (("HV  vs 3×v1 ", A), ("v31 vs 3×v1 (参照)", B),
                       ("HV  vs 3×v31", C)):
        wr, ci, sc = stat(rows)
        print(f"{name:20s} 胜率 {wr:.1f}% (±{ci:.1f})  场均 {sc:+.2f}")

    b_, c_, p_ = mcnemar([r[0] for r in A], [r[0] for r in B])
    print(f"\n配对检验 HV vs v31(同为对3×v1): 分歧对 b={b_} c={c_}, "
          f"McNemar p={p_:.4f}")


if __name__ == "__main__":
    main()
