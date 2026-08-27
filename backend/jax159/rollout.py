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
from .env159 import State, step_jit, legal_jit, load_win_tables, P_OVER
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
    return jax.jit(step_jit)


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
                max_steps: int = 200, feat_chunk: int = 512):
    """驱动批量状态到终局。net: JaxNet(特征 628)。
    返回 per-(si,tile) 的平均塑形得分 dict。"""
    load_win_tables()
    load_front_table()
    obs_jit = _get_obs_jit()
    step_j = _get_step_jit()
    legal_j = jax.jit(legal_jit)  # 单状态版; 批量用 vmap

    B = sts.hands.shape[0]
    done = jnp.zeros(B, dtype=jnp.bool_)
    draws = jnp.zeros(B, dtype=jnp.float32)
    n159 = jnp.zeros(B, dtype=jnp.int32)
    winner = jnp.full(B, -1, dtype=jnp.int32)

    for _ in range(max_steps):
        if bool(done.all()):
            break
        # 特征 + 合法掩码 + JAX 推理(全部留在 GPU; 特征分块控编译图规模)
        B = sts.hands.shape[0]
        feat_parts = []
        for c0 in range(0, B, feat_chunk):
            c1 = min(c0 + feat_chunk, B)
            feat_parts.append(obs_jit(slice_state(sts, c0, c1),
                                      sts.turn.astype(jnp.int8)[c0:c1]))
        feats = jnp.concatenate(feat_parts, axis=0)
        legal = jax.vmap(legal_j)(sts)[:, :28]  # 推演只用弃牌动作(28维)
        q = net.q_values(feats)
        q = jnp.where(legal, q, -1e9)
        acts = q.argmax(axis=-1).astype(jnp.int32)
        # react_wait 阶段: 暂不鸣牌, 一律过 (动作0=pass)
        acts = jnp.where(sts.phase == P_REACT, 0, acts)
        sts = step_j(sts, acts)
        # 记录终局
        over = sts.phase == P_OVER
        nw = over & ~done
        if nw.any():
            winner = winner.at[nw].set(sts.winner.astype(jnp.int32)[nw])
            n159 = n159.at[nw].set(sts.n_159.astype(jnp.int32)[nw])
            draws = draws.at[nw].set(sts.draws.astype(jnp.float32)[nw])
            done = done | nw
        if done.all():
            break
    # 塑形奖励: E[n|墙] 替换实际 n
    shaped = _shaped_from_final(sts, winner, n159, draws, step_penalty)
    r_map = {}
    buf = {}
    for (si, tile, hero), r in zip(meta, shaped.tolist()):
        buf.setdefault(si, {})[tile] = r
    for si, d in buf.items():
        r_map[si] = {t: v for t, v in d.items()}
    return r_map


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
    cnt159 = (IS159[sts.wall[idxm]].astype(jnp.float32) * mask).sum(-1)
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
