"""Cross-fit action selection across disjoint hidden worlds, grouped by source game.

This estimates a single-action change under the specified simulated world model;
it does not establish improvement in real complete games.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from backend.rl.eval_crn import _stat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data", nargs="+")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rows = []
    switches = []
    optimistic = []
    for path in args.data:
        d = np.load(path)
        if str(d["world_mode"]) != "resample":
            raise ValueError("Expected independently resampled worlds")
        returns = d["rollout_returns"]
        groups = d["game_id"]
        candidates = d["candidates"]
        n = returns.shape[-1]
        if n < 4:
            raise ValueError("Need at least 4 worlds for independent selection/evaluation")
        for i, group in enumerate(groups):
            valid = candidates[i] >= 0
            rs = returns[i, valid]
            assert np.isfinite(rs).all()
            a, b = rs[:, :n // 2].mean(1), rs[:, n // 2:].mean(1)
            ia, ib = int(a.argmax()), int(b.argmax())
            gain = ((b[ia] - b[0]) + (a[ib] - a[0])) / 2
            rows.append((int(group), float(gain)))
            switches.append(((ia != 0) + (ib != 0)) / 2)
            optimistic.append(((a[ia] - a[0]) + (b[ib] - b[0])) / 2)
    by_game = {}
    for group, gain in rows:
        by_game.setdefault(group, []).append(gain)
    result = {"purpose": "cross-fit one-action value under uniform hidden-world model; not match strength",
              "states": len(rows), "source_games": len(by_game),
              "crossfit_gain_per_game": _stat(np.array([np.mean(v) for v in by_game.values()])),
              "crossfit_gain_per_state": float(np.mean([v for _, v in rows])),
              "in_sample_selected_gain_per_state": float(np.mean(optimistic)),
              "switch_fraction": float(np.mean(switches)), "sources": args.data}
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
