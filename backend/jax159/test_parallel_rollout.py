"""多卡 rollout 正确性/速度验证: 单卡 vs 双卡 结果一致性。"""

import copy
import sys
import time

sys.path.insert(0, ".")

import numpy as np
import torch

from backend.game.engine import Game
from backend.ai.bot_v10 import Bot as V10
from backend.jax159.fast_inject import build_world_states
from backend.rl.world_grpo import _v10_scores


def main():
    g = Game(seed=9, human_seat=-1)
    bots = {i: V10(g, i) for i in range(4)}
    snaps = []
    guard = 0
    while g.phase != "game_over" and guard < 500 and len(snaps) < 8:
        guard += 1
        if g.phase == "discard_wait" and len(snaps) < 8:
            snaps.append((copy.deepcopy(g), g.turn))
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
    cands = []
    for snap, hero in snaps:
        sc = _v10_scores(snap, hero)
        cands.append([t for t, _ in sorted(sc.items(), key=lambda kv: -kv[1])[:4]])
    sts, meta, _ = build_world_states(snaps, cands, 16, 71)
    print("B =", sts.hands.shape[0])

    ckpt = torch.load("models/dqn_shaped_100k_best.pt", map_location="cpu",
                      weights_only=True)
    params = {k: v.detach().cpu().numpy() for k, v in ckpt["model"].items()}

    from backend.jax159.jax_net import JaxNet
    from backend.jax159.rollout import rollout_jax
    from backend.jax159.env159 import load_win_tables
    from backend.jax159.shanten import load_front_table
    load_win_tables()
    load_front_table()
    net = JaxNet.from_dict(params)
    t0 = time.time()
    r_single = rollout_jax(sts, meta, net, 0.02)
    t_single = time.time() - t0
    print(f"单卡 rollout: {t_single:.1f}s")

    from backend.jax159.parallel_rollout import (start_workers,
                                                 rollout_parallel,
                                                 stop_workers)
    in_q, out_q, procs = start_workers(2, "")
    t0 = time.time()
    r_multi = rollout_parallel(sts, meta, params, 0.02, in_q, out_q, 2)
    t_multi = time.time() - t0
    print(f"双卡 rollout: {t_multi:.1f}s")
    stop_workers(in_q, procs)

    bad = 0
    for si in r_single:
        if si in r_multi:
            for t in r_single[si]:
                if abs(r_single[si][t] - r_multi[si][t]) > 1e-3:
                    bad += 1
                    print(f"diff si={si} t={t}: {r_single[si][t]:.3f} vs {r_multi[si][t]:.3f}")
    print(f"一致性: {len(r_single)} 局面 x 4 候选, 不一致 {bad}")


if __name__ == "__main__":
    main()
