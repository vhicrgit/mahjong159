"""诊断 build_world_states 局面的第1步: B=768 vs B=384 的 obs/q/argmax。"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_default_matmul_precision", "float32")

from backend.jax159.test_pmap_rollout import _rand_games
from backend.jax159.rollout import build_world_states
from backend.jax159.convert import slice_state
from backend.jax159.env159 import legal_jit, load_win_tables
from backend.jax159.features import encode_obs
from backend.jax159.jax_net import JaxNet
from backend.jax159.shanten import load_front_table


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
    obs_j = jax.jit(encode_obs)
    legal_v = jax.vmap(jax.jit(legal_jit))
    seats = sts.turn.astype(jnp.int8)

    # 全量
    f_full = obs_j(sts, seats)
    q_full = np.asarray(net.q_values(f_full))
    l_full = np.asarray(legal_v(sts))[:, :28]
    a_full = np.argmax(np.where(l_full, q_full, -1e9), axis=-1)

    # 分块
    f_sp, q_sp, l_sp, a_sp = [], [], [], []
    for lo, hi in [(0, B // 2), (B // 2, B)]:
        s_i = slice_state(sts, lo, hi)
        f_i = obs_j(s_i, seats[lo:hi])
        q_i = np.asarray(net.q_values(f_i))
        l_i = np.asarray(legal_v(s_i))[:, :28]
        f_sp.append(np.asarray(f_i)); q_sp.append(q_i); l_sp.append(l_i)
        a_sp.append(np.argmax(np.where(l_i, q_i, -1e9), axis=-1))
    f_sp = np.concatenate(f_sp); q_sp = np.concatenate(q_sp)
    l_sp = np.concatenate(l_sp); a_sp = np.concatenate(a_sp)

    print(f"obs 最大差: {np.abs(np.asarray(f_full)-f_sp).max():.3e}", flush=True)
    print(f"q 最大差: {np.abs(q_full-q_sp).max():.3e}", flush=True)
    print(f"legal 不一致: {(l_full!=l_sp).sum()}", flush=True)
    print(f"np.argmax 不一致: {(a_full!=a_sp).sum()}/{B}", flush=True)
    # jnp.argmax (GPU) 对 B 的敏感性 —— rollout 里用的是这个
    qj_full = jnp.where(jnp.asarray(l_full), jnp.asarray(q_full), -1e9)
    ja_full = np.asarray(qj_full.argmax(axis=-1))
    ja_sp = np.concatenate([
        np.asarray(jnp.where(jnp.asarray(l_sp[:B//2]), jnp.asarray(q_sp[:B//2]), -1e9).argmax(axis=-1)),
        np.asarray(jnp.where(jnp.asarray(l_sp[B//2:]), jnp.asarray(q_sp[B//2:]), -1e9).argmax(axis=-1))])
    print(f"jnp.argmax(GPU) 不一致: {(ja_full!=ja_sp).sum()}/{B}", flush=True)
    # q 值并列(tie)统计: 最大值出现多次的局面数
    qmax = q_full.max(axis=-1, keepdims=True)
    n_tie = ((q_full == qmax).sum(axis=-1) > 1)
    print(f"q 有 tie 的局面: {n_tie.sum()}/{B}", flush=True)
    return
    # 找一个不一致局面, 看 q 差异
    diff = np.nonzero(a_full != a_sp)[0]
    if len(diff):
        i = int(diff[0])
        print(f"局面{i}: full_act={a_full[i]} split_act={a_sp[i]}", flush=True)
        print(f"  full q[top5]: {np.sort(q_full[i])[-5:]}", flush=True)
        print(f"  split q[top5]: {np.sort(q_sp[i])[-5:]}", flush=True)
        print(f"  full obs==split obs: {np.allclose(np.asarray(f_full)[i], f_sp[i])}", flush=True)
        print(f"  phase={np.asarray(sts.phase)[i]}", flush=True)


if __name__ == "__main__":
    main()
