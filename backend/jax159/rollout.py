"""JAX 世界推演驱动器: GRPO 的 (候选 × 世界) 批量推演全走 JAX 环境 + torch NN。

流程:
1. 输入: 局面快照列表(python Game) + 每局面候选弃牌 + 世界采样(对手手牌+牌墙)
2. 注入: 每个 (快照, 候选, 世界) 构造一个 JAX State, 首动作即候选弃牌
3. 循环: observe(JAX特征) -> torch NN 贪心选动作(四家) -> env step, 直到全部终局
4. 塑形奖励: 从终局 State 计算 E[n|wall] 替换实际翻牌 + 步数惩罚

用法(在 grpo_train 中以 --rollout jax 调用)。
"""

import random

import jax
import jax.numpy as jnp
import numpy as np

from .convert import state_from_game, batch_states, slice_state
from .env159 import State, step_jit, legal_jit, load_win_tables, P_OVER, P_REACT
from .features import encode_obs
from .shanten import load_front_table

IS159 = jnp.array([1 if (t < 27 and t % 9 in (0, 4, 8)) else 0
                   for t in range(28)], dtype=jnp.int32)

_obs_jit = None


def _get_obs_jit():
    global _obs_jit
    if _obs_jit is None:
        _obs_jit = jax.jit(encode_obs)
    return _obs_jit


def _get_step_jit():
    return jax.vmap(step_jit)  # 批量: vmap 单状态 step


def build_world_states(snaps, cand_lists, n_worlds, world_seed):
    """snaps: [(Game快照, hero_seat)]; cand_lists: [每局面候选列表]
    返回 (states(batched), meta[(si, tile, hero)])"""
    meta = []
    games = []
    rng = random.Random(world_seed)
    for si, (snap, hero) in enumerate(snaps):
        worlds = []
        for _ in range(n_worlds):
            worlds.append(_sample_world(snap, rng, hero))
        for tile in cand_lists[si]:
            for hands, wall in worlds:
                g = _inject(snap, hands, wall)
                g.action_discard(hero, tile)
                games.append(g)
                meta.append((si, tile, hero))
    sts = batch_states([state_from_game(g) for g in games])
    return sts, meta, games


def _sample_world(snap, rng, hero):
    """重洗未见牌: 对手暗牌(尺寸不变) + 剩余牌墙 (与 world_grpo 同语义)"""
    from backend.rl.world_grpo import sample_world
    return sample_world(snap, rng, hero_seat=hero)


def _inject(snap, hands, wall):
    import copy
    g = copy.deepcopy(snap)
    for seat, h in hands.items():
        g.players[seat].hand = list(h)
    g.wall = list(wall)
    return g


def rollout_jax(sts: State, meta, net, step_penalty: float,
                max_steps: int = 200):
    """全 jit 推演: 一次 while_loop 跑到终局, 返回 per-(si,tile) 平均塑形得分。
    net: JaxNet(其 params 作为显式参数, 更新不触发重编译)。"""
    load_win_tables()
    load_front_table()
    # 分块调用 jit 推演(每块形状静态, 峰值内存可控; 编译一次)
    B = sts.hands.shape[0]
    CH = 2048
    parts = []
    for c0 in range(0, B, CH):
        c1 = min(c0 + CH, B)
        part, _ = _rollout_jit(slice_state(sts, c0, c1), net.params,
                               step_penalty, max_steps)
        parts.append(part)
    if len(parts) == 1:
        sts = parts[0]
    else:
        sts = State(**{f: jnp.concatenate([getattr(p, f) for p in parts],
                                          axis=0)
                       for f in State._fields})
    winner = sts.winner.astype(jnp.int32)
    n159 = sts.n_159.astype(jnp.int32)
    draws = sts.draws.astype(jnp.float32)
    shaped = _shaped_from_final(sts, winner, n159, draws, step_penalty)
    r_map = {}
    buf = {}
    for (si, tile, hero), r in zip(meta, shaped.tolist()):
        buf.setdefault(si, {})[tile] = r
    for si, d in buf.items():
        r_map[si] = {t: v for t, v in d.items()}
    return r_map


def _rollout_scan_fn(sts, params, step_penalty, max_steps):
    """单次 jit 的整局推演: while_loop 包住 特征->推理->step 循环。"""
    from .env159 import P_OVER as _P_OVER, P_REACT as _P_REACT
    from .env159 import legal_jit as _legal_jit, step_jit as _step_jit
    from .features import encode_obs as _encode_obs
    from .jax_net import q_forward
    B = sts.hands.shape[0]

    def body(carry):
        sts, done, it = carry
        feats = _encode_obs(sts, sts.turn.astype(jnp.int8))
        legal = jax.vmap(_legal_jit)(sts)[:, :28]
        q = q_forward(params, feats)
        q = jnp.where(legal, q, -1e9)
        acts = q.argmax(axis=-1).astype(jnp.int32)
        acts = jnp.where(sts.phase == _P_REACT, 0, acts)
        sts = jax.vmap(_step_jit)(sts, acts)
        done = done | (sts.phase == _P_OVER)
        return sts, done, it + 1

    def cond(carry):
        sts, done, it = carry
        return (it < max_steps) & (~jnp.all(done))

    done0 = jnp.zeros(B, dtype=jnp.bool_)
    sts, done, it = jax.lax.while_loop(cond, body, (sts, done0, jnp.int32(0)))
    return sts, it


_rollout_jit = jax.jit(_rollout_scan_fn, static_argnames=("max_steps",))


def _shaped_from_final(sts, winner, n159, draws, step_penalty):
    """(B,) 塑形得分: scores = 杠分 + E[n|wall]分 - δ×draws。
    直接用最终 State 的 scores 字段(含实际翻牌分), 用 E[n] 替换翻牌部分。"""
    B = sts.hands.shape[0]
    wall_pos = sts.wall_pos.astype(jnp.int32)
    wall_tail = sts.wall_tail.astype(jnp.int32)
    wall_rem = (wall_tail - wall_pos)
    # E[n|wall] = 6 × 剩余墙中159密度; 墙不足6张时与引擎一致按0
    idxm = jnp.arange(112)[None, :]
    mask = (idxm >= wall_pos[:, None]) & (idxm < wall_tail[:, None])
    cnt159 = (IS159[sts.wall].astype(jnp.float32) * mask).sum(-1)
    n_exp = jnp.where(wall_rem >= 6,
                      6.0 * cnt159 / jnp.maximum(wall_rem, 1), 0.0)
    # 实际翻牌 n (只有 winner>=0 有意义)
    has_win = winner >= 0
    per_act = (n159 + 1).astype(jnp.float32)
    per_exp = n_exp + 1.0
    win_act = jnp.where(has_win[:, None] & (jnp.arange(4)[None, :] ==
                                            winner[:, None]),
                        3.0 * per_act[:, None], -per_act[:, None])
    win_act = jnp.where(has_win[:, None], win_act, 0.0)
    win_exp = jnp.where(has_win[:, None] & (jnp.arange(4)[None, :] ==
                                            winner[:, None]),
                        3.0 * per_exp[:, None], -per_exp[:, None])
    win_exp = jnp.where(has_win[:, None], win_exp, 0.0)
    # scores(最终) 含 杠分+实际翻牌分; 替换翻牌部分
    shaped_all = sts.scores.astype(jnp.float32) - win_act + win_exp
    return shaped_all.sum(-1) - step_penalty * draws
