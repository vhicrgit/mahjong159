"""Read-only diagnostics for the current CRN evaluator (not a strength benchmark).

  OMP_NUM_THREADS=1 python -m tools.perf.audit_eval_protocol
Counts available self-kong choices and actual callbacks, then compares standard
errors from treating seats as independent versus treating seeds as independent.
"""

import json

import numpy as np

from backend.ai.bot_native import NativeV1, NativeV31
from backend.rl import eval_crn


def main():
    counts = {"self_kong_options_at_discard": 0, "self_kong_callbacks": 0,
              "ming_kong_callbacks": 0, "games": 120}

    class ProbeBot(NativeV31):
        def choose_discard(self):
            counts["self_kong_options_at_discard"] += len(self.game._gang_options(self.seat))
            return super().choose_discard()

        def decide_gang(self, tile, kind):
            counts["ming_kong_callbacks" if kind == "ming" else "self_kong_callbacks"] += 1
            return super().decide_gang(tile, kind)

    seeds = list(range(202609050, 202609170))
    for seed in seeds:
        g = eval_crn._play(seed, False, {s: ProbeBot for s in range(4)})
        assert g.phase == "game_over"
    a, _ = eval_crn.rotate_arm(NativeV1, NativeV31, seeds, False)
    b, _ = eval_crn.baseline_arm(NativeV31, seeds, False)
    delta = a - b
    by_seed = delta.mean(axis=1)
    result = {"self_kong": counts, "v1_vs_v31_first_win_rank": {
        "mean": float(delta.mean()),
        "old_se": float(delta.std(ddof=1) / np.sqrt(delta.size)),
        "cluster_se": float(by_seed.std(ddof=1) / np.sqrt(len(seeds))),
        "independent_seeds": len(seeds), "seat_observations": delta.size,
        "purpose": "protocol diagnostic only; does not re-evaluate the NN champion"}}
    assert counts["self_kong_callbacks"] > 0
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
