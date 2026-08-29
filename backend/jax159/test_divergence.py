"""定位 rollout 的 B 敏感分叉点: 同一局面集, B=768 全量 vs B=384 分块,
复刻 rollout_jax 循环, 每步比较 acts, 找第一个分叉。

FP32 matmul。只跑到第一个分叉或 60 步。
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_default_matmul_precision", "float32")

from backend.jax159.test_pmap_rollout import _rand_games
from backend.jax159.rollout import build_world_states, _peng_gang_ok
from backend.jax159.convert import slice_state
from backend.jax159.env159 import (State, step_jit, legal_jit,
                                   load_win_tables, P_OVER)
from backend.jax159.features import encode_obs
from backend.jax159.jax_net import JaxNet
from backend.jax159.shanten import load_front_table


def run_steps(sts, net, n_steps, tag):
    """复刻 rollout_jax 循环, 返回每步 acts 历史。"""
    obs_j = jax.jit(encode_obs)
    legal_v = jax.vmap(jax.jit(legal_jit))
    step_v = jax.vmap(jax.jit(step_jit))
    B = sts.hands.shape[0]
    done = jnp.zeros(B, dtype=jnp.bool_)
    hist = []
    for step in range(n_steps):
        if bool(done.all()):
            break
        feats = obs_j(sts, sts.turn.astype(jnp.int8))
        legal = legal_v(sts)[:, :28]
        q = net.q_values(feats)
        q = jnp.where(legal, q, -1e9)
        acts = q.argmax(axis=-1).astype(jnp.int32)
        react_mask = np.asarray(sts.phase) == 1
        if react_mask.any():
            acts_np = np.asarray(acts).copy()
            hands_np = np.asarray(sts.hands)
            pend_p = np.asarray(sts.pend_peng)
            pend_g = np.asarray(sts.pend_gang)
            ld_np = np.asarray(sts.last_discard)
            for i in np.nonzero(react_mask)[0]:
                ps = int(np.argmax((pend_p[i] | pend_g[i]).astype(np.int8)))
                hand = list(hands_np[i, ps])
                t = int(ld_np[i])
                dg = pend_g[i, ps] and _peng_gang_ok(hand, t, 3)
                dp = pend_p[i, ps] and _peng_gang_ok(hand, t, 2)
                acts_np[i] = 2 if dg else (1 if dp else 0)
            acts = jnp.asarray(acts_np, dtype=jnp.int32)
        hist.append(np.asarray(acts))
        sts = step_v(sts, acts)
        done = done | (np.asarray(sts.phase) == P_OVER)
    return hist, sts


def main():
    load_win_tables(); load_front_table()
    snaps = _rand_games(14, seed=7)
    cand_lists = []
    for g, hero in snaps:
        hc = g.players[hero].hand_counts
        cand_lists.append([t for t in range(28) if hc[t] > 0][:4])
    net = JaxNet("models/dqn_shaped_100k_best_jax.npz")
    sts, meta, _ = build_world_states(snaps, cand_lists, 16, world_seed=99)
    B = sts.hands.shape[0]
    print(f"B={B}", flush=True)

    hist_full, sf = run_steps(sts, net, 60, "full")
    print(f"全量 {len(hist_full)} 步", flush=True)
    # 分块
    hist_a, sa = run_steps(slice_state(sts, 0, B // 2), net, 60, "a")
    hist_b, sb = run_steps(slice_state(sts, B // 2, B), net, 60, "b")
    print(f"分块 {len(hist_a)}+{len(hist_b)} 步", flush=True)

    # 找第一个分叉
    n = min(len(hist_full), len(hist_a), len(hist_b))
    for s in range(n):
        a_full = hist_full[s]
        a_split = np.concatenate([hist_a[s], hist_b[s]])
        diff = np.nonzero(a_full != a_split)[0]
        if len(diff):
            print(f"第 {s} 步分叉: {len(diff)} 局面 acts 不同, 前5: {diff[:5].tolist()}",
                  flush=True)
            i = int(diff[0])
            print(f"  局面{i}: full act={a_full[i]} split act={a_split[i]}",
                  flush=True)
            break
    else:
        print(f"前 {n} 步无分叉", flush=True)


if __name__ == "__main__":
    main()
