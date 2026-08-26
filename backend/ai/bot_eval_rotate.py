"""Evaluate a bot across all four seats against v1 opponents."""

import argparse
import multiprocessing as mp

import numpy as np

from .bot_eval import play_one


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bot", type=str, required=True)
    ap.add_argument("--games", type=int, default=400)
    ap.add_argument("--procs", type=int, default=32)
    ap.add_argument("--seed0", type=int, default=900000)
    ap.add_argument("--param", type=int, default=0)
    args = ap.parse_args()

    tasks = []
    for i in range(args.games):
        tasks.append((args.seed0 + i, args.bot, i % 4, args.param))
    with mp.Pool(args.procs) as pool:
        results = pool.map(play_one, tasks, chunksize=4)

    scores = np.array([r[0] for r in results], dtype=np.float64)
    wins = sum(1 for r in results if r[1])
    draws = sum(1 for r in results if r[2])
    n = len(results)
    se = 1.96 * np.sqrt(wins / n * (1 - wins / n) / n)
    print(f"{args.bot} vs 3×v1规则Bot, rotating seats, {n} 局:")
    print(f"  胜率 {wins/n:.1%} ±{se:.1%}(95%CI)")
    print(f"  场均 {scores.mean():+.3f}  流局 {draws/n:.1%}")


if __name__ == "__main__":
    main()
