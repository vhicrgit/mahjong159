"""决定性测试: 同一局面填充整个 batch, B=768 vs B=384 的 obs 是否一致。
若不同, 则 obs 对 B 敏感(与输入局面无关, 纯 kernel/编译问题)。"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import numpy as np
import jax
import jax.numpy as jnp
import random

from backend.game.engine import Game
from backend.jax159.convert import state_from_game, batch_states, slice_state
from backend.jax159.env159 import load_win_tables
from backend.jax159.features import encode_obs
from backend.jax159.shanten import load_front_table


def main():
    load_win_tables(); load_front_table()
    # 用 build_world_states 的注入局面(与 divergence 同源)
    from backend.jax159.test_pmap_rollout import _rand_games
    from backend.jax159.rollout import build_world_states
    from backend.jax159.jax_net import JaxNet
    snaps = _rand_games(14, seed=7)
    cand_lists = []
    for g, hero in snaps:
        hc = g.players[hero].hand_counts
        cand_lists.append([t for t in range(28) if hc[t] > 0][:4])
    net = JaxNet("models/dqn_shaped_100k_best_jax.npz")
    sts_w, meta, _ = build_world_states(snaps, cand_lists, 16, world_seed=99)
    # 取局面0(divergence 的分叉局面)
    s0 = slice_state(sts_w, 0, 1)
    print(f"局面0: phase={np.asarray(s0.phase)} turn={np.asarray(s0.turn)}",
          flush=True)
    # 同一局面填充 B=768 和 B=384 (s0 是 B=1 批量, 直接 tile 不要 batch_states)
    from backend.jax159.env159 import State
    results = {}
    for B in (768, 384, 256):
        sts = State(**{f: jnp.concatenate([getattr(s0, f)] * B, axis=0)
                       for f in State._fields})
        seats = sts.turn.astype(jnp.int8)
        f = np.asarray(jax.jit(encode_obs)(sts, seats))
        row_diff = np.abs(f - f[0:1]).max()
        results[B] = f[0]
        print(f"B={B}: 行内不一致={row_diff:.3e}, 维度507={f[0,507]:.4f}, "
              f"维度569={f[0,569]:.4f}, 维度597={f[0,597]:.4f}", flush=True)
    f768, f384 = results[768], results[384]
    d = np.abs(f768 - f384)
    print(f"\nB=768 vs B=384 注入局面0 obs 最大差={d.max():.3e}, "
          f"不一致维数={(d>1e-6).sum()}", flush=True)
    if d.max() > 1e-6:
        didx = np.nonzero(d > 1e-6)[0]
        print(f"差异维度={didx[:15].tolist()}", flush=True)
        for v in didx[:8]:
            print(f"  dim{v}: 768={f768[v]:.4f} 384={f384[v]:.4f}", flush=True)


if __name__ == "__main__":
    main()
