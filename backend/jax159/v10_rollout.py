"""安康159 - v10 规则 JAX 化: 用 GPU 全 jit 推演替代 Python v10 规则推演.

v10(BotV10) 的决策 = 向听 + 进张(ukeire) 启发式评分, 无搜索。
本模块把评分写成 JAX 函数(v10_scores), 整局推演用 jax.lax.while_loop
(与 _rollout_scan_fn 同构), 使规则推演也能 GPU 批量加速。

评分公式(对应 bot_v10._v10_scores, 默认权重 shanten_w=100, ukeire_w=1,
cont_w=0.5, risk_w=0.0; 当前省略两步价值 cont 与风险项):
  score[t] = -10*100 - 100*sh[t]          若 sh[t] > min_sh(退向听)
           = 1.0 * ukeire(hand-t, unseen) 否则
  决策 = argmax score(合法)

用法(在 grpo_train 中):
  --rollout-mode v10_jax
"""
import jax
import jax.numpy as jnp

from .env159 import P_OVER, P_REACT, legal_jit, step_jit
from .features import shanten_batch, _is_win_hand

RED = 27
SH_W = 100.0
UK_W = 1.0


def v10_scores(sts):
    """(B,28) v10 风格出牌评分。sts: State, 决策者 = sts.turn。"""
    B = sts.hands.shape[0]
    turn = sts.turn.astype(jnp.int32)
    hand14 = sts.hands[jnp.arange(B), turn]                      # (B,28)
    eye = jnp.eye(28, dtype=jnp.int32)
    h13 = hand14[:, None, :] - eye[None, :, :]                   # (B,28,28)
    in_hand = hand14 > 0                                         # (B,28)
    sh_all = shanten_batch(h13.reshape(-1, 28)).reshape(B, 28)
    sh = jnp.where(in_hand, sh_all, 99)                          # (B,28)
    min_sh = jnp.min(jnp.where(in_hand, sh, 99), axis=-1)        # (B,)
    # 可见牌: 仅当前决策者手牌 + 所有家弃牌 + 副露 (有限信息, 与 Python v10 一致)
    hero_hand = sts.hands[jnp.arange(B), turn]                   # (B,28)
    visible = hero_hand + sts.discards.sum(1)
    mt = sts.melds_tile.astype(jnp.int32)                        # (B,4,4)
    mk = sts.melds_kind.astype(jnp.int32)                        # (B,4,4)
    meld_vis = jnp.zeros((B, 28), dtype=jnp.float32)
    for slot in range(4):
        t = mt[:, :, slot]                                       # (B,4)
        k = mk[:, :, slot]
        n = jnp.where(k == 1, 3.0, jnp.where(k == 2, 4.0, 0.0))
        meld_vis = meld_vis.at[jnp.arange(B)[:, None], t].add(n)
    visible = visible + meld_vis
    unseen = jnp.maximum(4.0 - visible, 0.0)                     # (B,28)
    # ukeire 网格: 每候选打后, 枚举 28 种进张
    H = h13[:, :, None, :] + eye[None, None, :, :]               # (B,28,28,28)
    sh_draw = shanten_batch(H.reshape(-1, 28)).reshape(B, 28, 28)  # (B,28,28)
    win_all = _is_win_hand(H.reshape(-1, 28).astype(jnp.int8)).reshape(B, 28, 28)
    useful = (sh_draw < sh[:, :, None]) & in_hand[:, :, None]
    wait = jnp.where(sh[:, :, None] == 0, win_all, useful)
    ukeire = (wait * unseen[:, None, :]).sum(-1)                 # (B,28)
    score = jnp.where(sh > min_sh[:, None],
                      -10.0 * SH_W - SH_W * sh,
                      UK_W * ukeire)
    return jnp.where(in_hand, score, -1e9)


def v10_rollout_scan_fn(sts, step_penalty, max_steps):
    """单次 jit 的整局推演: 决策 = v10 启发式评分 argmax。"""
    from .env159 import step_jit as _step_jit, legal_jit as _legal_jit
    B = sts.hands.shape[0]

    def body(carry):
        sts, done, it = carry
        score = v10_scores(sts)
        legal = jax.vmap(_legal_jit)(sts)[:, :28]
        acts = jnp.argmax(jnp.where(legal, score, -1e9), axis=-1).astype(jnp.int32)
        acts = jnp.where(sts.phase == P_REACT, 0, acts)
        sts = jax.vmap(_step_jit)(sts, acts)
        done = done | (sts.phase == P_OVER)
        return sts, done, it + 1

    def cond(carry):
        sts, done, it = carry
        return (it < max_steps) & (~jnp.all(done))

    done0 = jnp.zeros(B, dtype=jnp.bool_)
    sts, done, it = jax.lax.while_loop(cond, body, (sts, done0, jnp.int32(0)))
    return sts, it


v10_rollout_jit = jax.jit(v10_rollout_scan_fn, static_argnames=("max_steps",))

def rollout_v10_jax(sts, meta, step_penalty, max_steps=200, feat_chunk=512):
    """v10 规则全 jit 推演, 返回 per-(si,tile) 平均塑形得分(接口同 rollout_jax)。
    分块调用 v10_rollout_jit, 控制编译图大小(每块形状静态)。"""
    import numpy as np
    from .rollout import _shaped_from_final, slice_state
    from .env159 import State, load_win_tables
    from .shanten import load_front_table
    load_win_tables()
    load_front_table()
    B = sts.hands.shape[0]
    hero_arr = np.array([h for _, _, h in meta], dtype=np.int32)
    parts = []
    for c0 in range(0, B, feat_chunk):
        c1 = min(c0 + feat_chunk, B)
        part, _ = v10_rollout_jit(slice_state(sts, c0, c1), step_penalty,
                                  max_steps)
        parts.append(part)
    final_sts = parts[0] if len(parts) == 1 else State(**{
        f: jnp.concatenate([getattr(p, f) for p in parts], axis=0)
        for f in State._fields})
    winner = final_sts.winner.astype(jnp.int32)
    n159 = final_sts.n_159.astype(jnp.int32)
    draws = final_sts.draws.astype(jnp.float32)
    shaped = np.asarray(_shaped_from_final(final_sts, winner, n159, draws,
                                           step_penalty, hero_arr))
    buf = {}
    for k, (si, tile, hero) in enumerate(meta):
        buf.setdefault(si, {}).setdefault(tile, []).append(shaped[k])
    r_map = {si: {t: float(np.mean(v)) for t, v in d.items()}
             for si, d in buf.items()}
    return r_map
