"""JAX 查表向听: 三花色前沿合并 (MahJax 移植的关键算子)

shanten_batch(hands) -> (B,) int32, 与 backend.rules.win.shanten 全等。
代价: 35 种红中分配 × (B, K³) 三元组合求最小。K 由表决定(当前 20)。
"""

import os
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

_POW5 = jnp.array([5 ** i for i in range(9)], dtype=jnp.int32)
FRONT = None
K = 0


def load_front_table(path=None):
    global FRONT, K
    if FRONT is None:
        p = path or os.environ.get("SUIT_TABLE", "models/suit_front_table.npz")
        z = np.load(p)
        FRONT = jnp.asarray(z["table"])   # (5^9, 5, K, 4) uint8
        K = FRONT.shape[2]
    return FRONT


ALL_SPLITS = [(r0, r1, r2)
              for r0 in range(5) for r1 in range(5) for r2 in range(5)
              if r0 + r1 + r2 <= 4]


def _merge_score(F0, F1, F2, need, red, ok):
    """三花色前沿合并, 返回每个三元组的最终向听分数 (B,K,K,K) int32。
    F*: (B, K, 4) 每行 (m, t, p, r_used), 无效项 m==255。"""
    PAD = 255
    m0, t0, p0, r0 = F0[..., 0], F0[..., 1], F0[..., 2], F0[..., 3]
    m1, t1, p1, r1 = F1[..., 0], F1[..., 1], F1[..., 2], F1[..., 3]
    m2, t2, p2, r2 = F2[..., 0], F2[..., 1], F2[..., 2], F2[..., 3]

    m = (m0[:, :, None, None] + m1[:, None, :, None] +
         m2[:, None, None, :]).astype(jnp.int32)
    t = (t0[:, :, None, None] + t1[:, None, :, None] +
         t2[:, None, None, :]).astype(jnp.int32)
    p = (p0[:, :, None, None] + p1[:, None, :, None] +
         p2[:, None, None, :]).astype(jnp.int32)
    r = (r0[:, :, None, None] + r1[:, None, :, None] +
         r2[:, None, None, :]).astype(jnp.int32)

    valid = ((m0 != PAD)[:, :, None, None] &
             (m1 != PAD)[:, None, :, None] &
             (m2 != PAD)[:, None, None, :] &
             ok[:, None, None, None])

    left = red[:, None, None, None] - r
    q = left // 3
    rem = left % 3
    mc = jnp.minimum(m + q, need[:, None, None, None])
    tA = t + (rem == 2)
    sA = (2 * need[:, None, None, None] - 2 * mc -
          jnp.minimum(tA, need[:, None, None, None] - mc) -
          jnp.minimum(p, 1))
    pB = jnp.minimum(p + (rem == 2), 1)
    sB = (2 * need[:, None, None, None] - 2 * mc -
          jnp.minimum(t, need[:, None, None, None] - mc) - pB)
    s = jnp.minimum(sA, sB)
    return jnp.where(valid, s, 99)


def shanten_batch(hands) -> jax.Array:
    """hands: (B, 28) int8/32 计数(任意张数) -> (B,) int32 向听数"""
    load_front_table()
    c = hands.astype(jnp.int32)
    c0 = (c[:, 0:9] * _POW5).sum(1)
    c1 = (c[:, 9:18] * _POW5).sum(1)
    c2 = (c[:, 18:27] * _POW5).sum(1)
    red = c[:, 27]
    total = c.sum(1)
    need = jnp.clip((total - 1) // 3, 1, 4)

    best = jnp.full((hands.shape[0],), 99, dtype=jnp.int32)
    for r0, r1, r2 in ALL_SPLITS:
        ok = (r0 + r1 + r2) <= red
        F0 = FRONT[c0, r0]
        F1 = FRONT[c1, r1]
        F2 = FRONT[c2, r2]
        s = _merge_score(F0, F1, F2, need, red, ok)
        best = jnp.minimum(best, s.min(axis=(1, 2, 3)))
    return best
