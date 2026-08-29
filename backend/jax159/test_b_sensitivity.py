"""验证 B 大小对 rollout 结果的浮点敏感性, 以及 FP32 matmul 是否消除。

同局面: B=768 全量 rollout vs 分两块 B=384 rollout 拼接。
TF32(默认) 与 FP32 各测一次。
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import numpy as np
import jax

from backend.jax159.test_pmap_rollout import _rand_games
from backend.jax159.rollout import build_world_states, rollout_jax
from backend.jax159.jax_net import JaxNet


def run_once(precision):
    if precision == "float32":
        jax.config.update("jax_default_matmul_precision", "float32")
    snaps = _rand_games(14, seed=7)
    cand_lists = []
    for g, hero in snaps:
        hc = g.players[hero].hand_counts
        cand_lists.append([t for t in range(28) if hc[t] > 0][:4])
    net = JaxNet("models/dqn_shaped_100k_best_jax.npz")
    sts, meta, _ = build_world_states(snaps, cand_lists, 16, world_seed=99)
    B = sts.hands.shape[0]
    # 全量
    r_full = rollout_jax(sts, meta, net, step_penalty=0.02)
    # 分两块
    from backend.jax159.convert import slice_state
    r_split = {}
    for lo, hi in [(0, B // 2), (B // 2, B)]:
        sts_i = slice_state(sts, lo, hi)
        meta_i = meta[lo:hi]
        r_i = rollout_jax(sts_i, meta_i, net, step_penalty=0.02)
        for si, d in r_i.items():
            r_split.setdefault(si, {}).update(d)
    bad = sum(1 for si in r_full for t in r_full[si]
              if abs(r_full[si][t] - r_split[si][t]) > 1e-4)
    tot = sum(len(d) for d in r_full.values())
    print(f"[{precision}] B={B} 全量 vs 分块: 不一致 {bad}/{tot}", flush=True)
    return bad


if __name__ == "__main__":
    prec = os.environ.get("PREC", "default")
    run_once(prec)
