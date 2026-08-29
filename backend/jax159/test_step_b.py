"""测 step 对真实 NN 动作的 B 一致性: 第0步 argmax acts, step 后比较所有字段。"""
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
from backend.jax159.env159 import step_jit, legal_jit, load_win_tables
from backend.jax159.features import encode_obs
from backend.jax159.jax_net import JaxNet
from backend.jax159.shanten import load_front_table


def step0_acts(sts, net):
    """复刻 rollout 第0步: obs->forward->argmax->react覆盖"""
    obs_j = jax.jit(encode_obs)
    legal_v = jax.vmap(jax.jit(legal_jit))
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
    return acts


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
    step_v = jax.vmap(jax.jit(step_jit))

    acts = step0_acts(sts, net)  # 完整 B 的第0步动作(确定性)
    # full step
    s_full = step_v(sts, acts)
    # split step: 用相同的 acts 切片
    s_a = step_v(slice_state(sts, 0, B//2), acts[:B//2])
    s_b = step_v(slice_state(sts, B//2, B), acts[B//2:])
    nbad = 0
    for f in s_full._fields:
        a = np.asarray(getattr(s_full, f))
        b = np.concatenate([np.asarray(getattr(s_a, f)),
                            np.asarray(getattr(s_b, f))], axis=0)
        d = np.abs(a.astype(np.int64) - b.astype(np.int64))
        if d.max() != 0:
            nbad += 1
            nidx = np.nonzero(d.reshape(B, -1).any(axis=-1))[0]
            print(f"字段 {f}: {len(nidx)} 局面不同, 前5={nidx[:5].tolist()}, "
                  f"最大差={d.max()}", flush=True)
    print(f"不一致字段数: {nbad}", flush=True)
    print("PASS" if nbad == 0 else "FAIL", flush=True)


if __name__ == "__main__":
    main()
