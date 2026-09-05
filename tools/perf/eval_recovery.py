"""Evaluate recovery checkpoints against a frozen champion on fresh first-win games.

Saves raw (seed, model, seat, metric) returns for clustered reanalysis. Model A/B
each face three NativeV31 opponents; this is a paired benchmark, not direct A/B play.
"""

import argparse
import json
import multiprocessing as mp
from pathlib import Path
import time

import numpy as np
import torch

from backend.ai.bot_native import NativeV31
from backend.rl import eval_crn
from tools.perf.eval_ckpt import make_factory

_FACTORIES = None


def initialize(paths):
    global _FACTORIES
    torch.set_num_threads(1)
    _FACTORIES = [make_factory(p, "hv")[0] for p in paths]


def evaluate_seed(seed):
    out = np.zeros((len(_FACTORIES), 4, 4), dtype=np.float64)
    for i, factory in enumerate(_FACTORIES):
        for s in range(4):
            factories = {seat: NativeV31 for seat in range(4)}
            factories[s] = factory
            g = eval_crn._play(seed, False, factories)
            out[i, s] = [g.players[s].score_delta, g.rank_rewards()[s],
                         float(g.winner == s), float(g.huangzhuang)]
    return seed, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True, help="First model is frozen reference")
    ap.add_argument("--seed0", type=int, required=True)
    ap.add_argument("--seeds", type=int, default=300)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.seeds < 2:
        raise ValueError("Need at least two independent seeds")
    start = time.time()
    values = []
    with mp.get_context("spawn").Pool(args.workers, initialize, (args.models,)) as pool:
        for i, row in enumerate(pool.imap(evaluate_seed, range(args.seed0, args.seed0 + args.seeds)), 1):
            values.append(row)
            if i % 25 == 0:
                print(f"{i}/{args.seeds} seeds, {time.time()-start:.1f}s", flush=True)
    seeds = np.array([s for s, _ in values])
    returns = np.stack([v for _, v in values])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(out) + ".npz", seeds=seeds, returns=returns,
                        models=np.array(args.models),
                        metrics=np.array(["raw_score", "rank", "win", "draw"]))
    results = {}
    for m, path in enumerate(args.models):
        results[path] = {metric: eval_crn._stat(returns[:, m, :, k] - returns[:, 0, :, k])
                         for k, metric in enumerate(["raw_score", "rank", "win", "draw"])}
    summary = {"protocol": eval_crn.PROTOCOL, "opponents": "3xNativeV31", "claim": "hv",
               "seed0": args.seed0, "seeds": args.seeds, "seconds": time.time()-start,
               "reference": args.models[0], "differences_vs_reference": results}
    out.with_suffix(".json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
