"""安康159 - Cheat JAX: 用 GPU 全 jit 推演评估候选, 替代 Python beam search.

原理: cheat_full(神挂) 每步做 beam search + Python rollout, 慢在 Python 推演。
cheat_full_jax 对全部候选牌**一次批量**构建共享世界 State,
用 rollout.py 的 _rollout_jit(全 jit while_loop, NN 决策) 在 GPU 上推演到底,
取 n_worlds 世界均值塑形回报, 选最优。

性能关键: 候选 pad 到 MAX_CANDS, 世界数固定 N_WORLDS -> 输入形状恒定,
XLA 只编译一次(约 2 分钟), 之后每次决策 ~2s。

用法(在 grpo_train 中):
  --rollout-mode cheat_full_jax
"""
import copy

from backend.rules.ting import discard_options
from backend.game.engine import Game

MAX_CANDS = 16
N_WORLDS = 32


def choose_discard_jax(game: Game, seat: int, net,
                       n_worlds: int = N_WORLDS,
                       step_penalty: float = 0.02) -> int:
    """批量评估全部合法候选, 选回报最高的。候选 pad 到 MAX_CANDS 固定形状。"""
    import numpy as np
    import jax.numpy as jnp
    from backend.jax159.fast_inject import build_world_states
    from backend.jax159.rollout import _rollout_jit, _shaped_from_final
    from backend.jax159.env159 import load_win_tables
    from backend.jax159.shanten import load_front_table

    cnt = list(game.players[seat].hand_counts)
    opts = discard_options(list(cnt))
    tiles = [o["tile"] for o in opts]
    if not tiles:
        return -1
    if len(tiles) == 1:
        return tiles[0]
    if len(tiles) > MAX_CANDS:
        tiles = tiles[:MAX_CANDS]

    load_win_tables()
    load_front_table()

    snap = copy.deepcopy(game)
    snap.log = []
    world_seed = hash((id(game), seat)) & 0x7FFFFFFF
    # pad 到 MAX_CANDS(重复最后一张), 保证形状恒定 -> jit 只编译一次
    cands_pad = tiles + [tiles[-1]] * (MAX_CANDS - len(tiles))
    sts, meta, _ = build_world_states([(snap, seat)], [cands_pad],
                                      n_worlds, world_seed)
    final_sts, _ = _rollout_jit(sts, net.params, step_penalty, 200)

    winner = final_sts.winner.astype(jnp.int32)
    n159 = final_sts.n_159.astype(jnp.int32)
    draws = final_sts.draws.astype(jnp.float32)
    hero_arr = np.array([h for _, _, h in meta], dtype=np.int32)
    shaped = np.asarray(_shaped_from_final(final_sts, winner, n159, draws,
                                           step_penalty, hero_arr))
    means = {}
    for k, (si, tile, hero) in enumerate(meta):
        means.setdefault(tile, []).append(shaped[k])
    # 只比较真实候选
    real = {t: float(np.mean(means[t])) for t in tiles}
    return int(max(real, key=real.get))