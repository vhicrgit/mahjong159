"""多卡 pmap 推演: State 按设备分片, obs/forward/step pmap 并行。

流程与 rollout_jax 完全一致(Python 驱动循环 + react 规则决策),
区别是每步的 GPU 计算(obs/forward/step)分片到 N 卡并行。

用法: CUDA_VISIBLE_DEVICES 包含多卡, 单进程 pmap 自动用全部可见设备。
react 决策用批量 jax(peng_gang_ok_batch), 避免逐局 Python DFS。
"""

import jax
import jax.numpy as jnp
import numpy as np

from .env159 import State, step_jit, legal_jit, load_win_tables, P_OVER
from .features import encode_obs
from .jax_net import q_forward
from .shanten import load_front_table, shanten_batch
from .rollout import _shaped_from_final


def shard_state(sts, ndev):
    """(B,...) State -> (ndev, B//ndev, ...) State。B 必须整除 ndev。"""
    per = sts.hands.shape[0] // ndev
    return State(**{f: getattr(sts, f).reshape(
        ndev, per, *getattr(sts, f).shape[1:]) for f in State._fields})


def pad_state(sts, target_b):
    """B -> target_b, 用第 0 局面填充(pad 局面不参与结果聚合)。"""
    B = sts.hands.shape[0]
    if B >= target_b:
        return sts
    pad = target_b - B
    return State(**{f: jnp.concatenate(
        [getattr(sts, f),
         jnp.concatenate([getattr(sts, f)[:1]] * pad, axis=0)],
        axis=0) for f in State._fields})


def peng_gang_ok_batch(hands, tiles, n):
    """v1 语义批量版: 碰/杠后最优弃牌向听 < 当前向听。
    hands: (R,28) 计数; tiles: (R,); n: 2(碰)/3(杠)。返回 (R,) bool。"""
    R = hands.shape[0]
    idx = jnp.arange(R)
    c11 = hands.at[idx, tiles].add(-n).astype(jnp.int32)
    eye = jnp.eye(28, dtype=jnp.int32)
    c10 = jnp.clip(c11[:, None, :] - eye[None, :, :], 0, 4)
    valid = c11 > 0
    s = shanten_batch(c10.reshape(-1, 28)).reshape(R, 28)
    after = jnp.where(valid, s, 99).min(axis=1)
    before = shanten_batch(hands.astype(jnp.int32))
    return after < before


def _obs_step(sts_sh, params):
    """单卡分片内的一步: 特征+合法掩码+Q+argmax。返回 acts。"""
    feats = encode_obs(sts_sh, sts_sh.turn.astype(jnp.int8))
    legal = jax.vmap(legal_jit)(sts_sh)[:, :28]
    q = q_forward(params, feats)
    q = jnp.where(legal, q, -1e9)
    return q.argmax(axis=-1).astype(jnp.int32)


def _apply(sts_sh, acts_sh):
    return jax.vmap(step_jit)(sts_sh, acts_sh)


def rollout_jax_pmap(sts: State, meta, net, step_penalty: float,
                     max_steps: int = 200):
    """多卡分片推演。net: JaxNet。返回与 rollout_jax 相同的 r_map。"""
    load_win_tables()
    load_front_table()
    ndev = jax.local_device_count()
    B0 = sts.hands.shape[0]
    tgt = (B0 + ndev - 1) // ndev * ndev
    sts = pad_state(sts, tgt)
    B = sts.hands.shape[0]
    per = B // ndev
    sts_sh = shard_state(sts, ndev)

    devices = jax.local_devices()
    params_rep = jax.device_put_replicated(net.params, devices)
    compute_acts = jax.pmap(_obs_step)
    apply_step = jax.pmap(_apply)

    hero_arr = np.array([h for _, _, h in meta] + [0] * (B - B0),
                        dtype=np.int32)
    done = np.zeros(B, dtype=bool)
    for _ in range(max_steps):
        if done.all():
            break
        acts_sh = compute_acts(sts_sh, params_rep)  # (ndev, per)
        acts = np.asarray(acts_sh).reshape(-1).copy()
        # react 决策(批量 jax, 只对 react 局面)
        phase = np.asarray(sts_sh.phase).reshape(-1)
        react_mask = phase == 1
        if react_mask.any():
            pend_p = np.asarray(sts_sh.pend_peng).reshape(B, 4)
            pend_g = np.asarray(sts_sh.pend_gang).reshape(B, 4)
            hands_np = np.asarray(sts_sh.hands).reshape(B, 4, 28)
            ld = np.asarray(sts_sh.last_discard).reshape(B)
            ridx = np.nonzero(react_mask)[0]
            ps = np.argmax((pend_p[ridx] | pend_g[ridx]).astype(np.int8),
                           axis=1)
            hands_r = hands_np[ridx, ps]
            tiles_r = ld[ridx]
            dg = np.asarray(peng_gang_ok_batch(
                jnp.asarray(hands_r), jnp.asarray(tiles_r), 3)) & \
                pend_g[ridx, ps]
            dp = np.asarray(peng_gang_ok_batch(
                jnp.asarray(hands_r), jnp.asarray(tiles_r), 2)) & \
                pend_p[ridx, ps]
            acts[ridx] = np.where(dg, 2, np.where(dp, 1, 0))
        acts_sh = jnp.asarray(acts, dtype=jnp.int32).reshape(ndev, per)
        sts_sh = apply_step(sts_sh, acts_sh)
        done = done | (np.asarray(sts_sh.phase).reshape(-1) == P_OVER)

    sts = State(**{f: getattr(sts_sh, f).reshape(B, *getattr(sts_sh, f).shape[2:])
                   for f in State._fields})
    winner = sts.winner.astype(jnp.int32)
    n159 = sts.n_159.astype(jnp.int32)
    draws = sts.draws.astype(jnp.float32)
    shaped = _shaped_from_final(sts, winner, n159, draws, step_penalty,
                                hero_arr)
    shaped = np.asarray(shaped)[:B0]
    r_map = {}
    buf = {}
    for (si, tile, hero), r in zip(meta, shaped.tolist()):
        buf.setdefault(si, {}).setdefault(tile, []).append(r)
    for si, d in buf.items():
        r_map[si] = {t: float(np.mean(v)) for t, v in d.items()}
    return r_map
