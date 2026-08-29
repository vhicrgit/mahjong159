"""JAX 特征编码: 从 env159.State 批量编码 v2 特征 (628维, 与 features_v2 全等)。

布局与 backend/rl/features_v2.py 一致:
- v1 基础 512: 手牌oh4(112) 自己副露oh4(112) 对手副露168 弃牌计数112 全局8
- v2 派生 116: 打后向听28 打后进张28 放杠风险28 可见剩余28 + 全局4
"""

import jax
import jax.numpy as jnp

from .env159 import RED, State, _is_win_hand, load_win_tables
from .shanten import shanten_batch, load_front_table

POW5 = jnp.array([5 ** i for i in range(9)], dtype=jnp.int32)
IS159 = jnp.array([1 if (t < 27 and t % 9 in (0, 4, 8)) else 0
                   for t in range(28)], dtype=jnp.int32)

FEAT_DIM = 628


def _oh4(n: jax.Array) -> jax.Array:
    """(...,) 张数0-4 -> (...,4) one-hot; n==4 映射到 3"""
    idx = jnp.minimum(n.astype(jnp.int32), 3)
    return jax.nn.one_hot(idx, 4, dtype=jnp.float32)


def _meld_counts(melds_tile, melds_kind):
    """(B,4) 副露槽 -> (B,28) 每牌副露张数 (碰3, 杠4)。

    [bug修复] scatter 索引曾用 arange(B)[:,None,None] (B,1,1), 与
    melds_tile (B,4) 广播成 (B,B,4) → 副露计数跨局面串台(污染量随B增大,
    正是多卡rollout浮点分叉的根因)。应为 arange(B)[:,None] (B,1)。"""
    B = melds_tile.shape[0]
    kind = melds_kind  # 1碰 2明杠 3暗杠 4补杠
    cnt = jnp.where(kind > 0, jnp.where(kind == 1, 3, 4), 0)  # (B,4)
    return jnp.zeros((B, 28), dtype=jnp.int32).at[
        jnp.arange(B)[:, None], melds_tile.astype(jnp.int32)
    ].add(cnt.astype(jnp.int32))  # kind==0 时 melds_tile=0 会加0, 无碍


def encode_obs(sts: State, seats) -> jax.Array:
    """从每个游戏的 seats[b] 视角编码特征 -> (B, 628) float32。
    sts 是批量状态; seats (B,) int8 每个游戏的"自己"座位。"""
    load_win_tables()
    load_front_table()
    B = sts.hands.shape[0]

    # ---------- v1 基础 (512) ----------
    # 1. 手牌 oh4 (28x4)
    my_hand = jnp.take_along_axis(
        sts.hands, seats[:, None, None].astype(jnp.int32), axis=1)[:, 0]
    f_hand = _oh4(my_hand).reshape(B, 112)

    # 2. 自己副露 oh4 (28x4)
    my_meld_t = jnp.take_along_axis(
        sts.melds_tile, seats[:, None, None].astype(jnp.int32), axis=1)[:, 0]
    my_meld_k = jnp.take_along_axis(
        sts.melds_kind, seats[:, None, None].astype(jnp.int32), axis=1)[:, 0]
    f_self_meld = _oh4(_meld_counts(my_meld_t, my_meld_k)).reshape(B, 112)

    # 3. 对手副露 (3 x 28 x 2)
    f_opp = []
    for rel in range(1, 4):
        opp_seat = (seats + rel) % 4
        ot = jnp.take_along_axis(
            sts.melds_tile, opp_seat[:, None, None].astype(jnp.int32),
            axis=1)[:, 0]                                   # (B,4)
        ok_ = jnp.take_along_axis(
            sts.melds_kind, opp_seat[:, None, None].astype(jnp.int32),
            axis=1)[:, 0]
        valid = ok_ != 0
        peng_v = jnp.where(valid & (ok_ == 1), 1.0, 0.0)   # (B,4) 逐槽
        gang_v = jnp.where(valid & (ok_ >= 2) & (ok_ <= 4), 1.0, 0.0)
        idx = jnp.arange(B)[:, None]
        ot_i = ot.astype(jnp.int32)
        peng_f = jnp.zeros((B, 28)).at[idx, ot_i].add(peng_v)
        gang_f = jnp.zeros((B, 28)).at[idx, ot_i].add(gang_v)
        f_opp.append(peng_f)
        f_opp.append(gang_f)
    f_opp = jnp.concatenate(f_opp, axis=-1)  # (B,168)

    # 4. 各家弃牌计数 (4x28, /4)
    f_disc = []
    for rel in range(4):
        seat = (seats + rel) % 4
        dc = jnp.take_along_axis(
            sts.discards, seat[:, None, None].astype(jnp.int32), axis=1)[:, 0]
        f_disc.append(dc.astype(jnp.float32) / 4.0)
    f_disc = jnp.concatenate(f_disc, axis=-1)  # (B,112)

    # 5. 全局 (8)
    wall_rem = (sts.wall_tail - sts.wall_pos).astype(jnp.float32)
    wall_ratio = wall_rem / 112.0
    is_dealer = (seats == 0).astype(jnp.float32)
    progress = 1.0 - wall_ratio
    # 可见159数
    seen159 = jnp.zeros(B, dtype=jnp.int32)
    for p in range(4):
        dc = sts.discards[:, p]
        seen159 += (dc * IS159.astype(jnp.int32)).sum(-1)
    for p in range(4):
        cnt = _meld_counts(sts.melds_tile[:, p], sts.melds_kind[:, p])
        seen159 += (cnt * IS159.astype(jnp.int32)).sum(-1)
    seen159 += (my_hand.astype(jnp.int32) * IS159.astype(jnp.int32)).sum(-1)
    # 对手副露数 (3个)
    opp_meld_n = []
    for rel in range(1, 4):
        opp_seat = (seats + rel) % 4
        n = jnp.take_along_axis(
            sts.n_melds, opp_seat[:, None].astype(jnp.int32), axis=1)[:, 0]
        opp_meld_n.append(n.astype(jnp.float32) / 4.0)
    f_glob = jnp.stack([
        wall_ratio, is_dealer, progress, seen159.astype(jnp.float32) / 36.0,
        *opp_meld_n, my_hand[:, RED].astype(jnp.float32) / 4.0], axis=-1)

    base = jnp.concatenate([f_hand, f_self_meld, f_opp, f_disc, f_glob],
                           axis=-1)  # (B,512)
    assert base.shape[-1] == 512

    # ---------- v2 派生 (116) ----------
    # 全局可见计数(弃牌+副露+自己手牌)
    visible = jnp.zeros((B, 28), dtype=jnp.int32)
    for p in range(4):
        visible += sts.discards[:, p].astype(jnp.int32)
    for p in range(4):
        visible += _meld_counts(sts.melds_tile[:, p], sts.melds_kind[:, p])
    visible += my_hand.astype(jnp.int32)

    hand14 = my_hand.astype(jnp.int32)          # (B,28) 14张计数
    eye = jnp.eye(28, dtype=jnp.int32)
    # 28 候选一次批量: h13 (B,28,28)
    h13 = hand14[:, None, :] - eye[None, :, :]           # (B,28,28)
    in_hand = hand14 > 0                                  # (B,28)
    sh_all = shanten_batch(h13.reshape(-1, 28)).reshape(B, 28)
    sh = jnp.where(in_hand, sh_all, 99)
    # 784 win 检查一次批量: H (B,28,28,28)
    H = (h13[:, :, None, :] + eye[None, None, :, :]).reshape(-1, 28)
    win_all = _is_win_hand(H.astype(jnp.int8)).reshape(B, 28, 28)
    rem_after = jnp.maximum(4 - h13, 0)                     # (B,28,28) cand×draw
    wc = (win_all * rem_after).sum(-1)                      # (B,28)
    wc = jnp.where(sh == 0, wc, 0)
    tile_shanten = jnp.where(in_hand, sh / 5.0, 1.0)
    tile_waits = wc / 40.0
    # 风险/剩余(28 张批量)
    visible_t = visible                                   # (B,28)
    remain = 4 - visible_t
    penged_tiles = jnp.zeros((B, 28), dtype=jnp.bool_)
    for rel in range(1, 4):
        opp_seat = (seats + rel) % 4
        ot = jnp.take_along_axis(
            sts.melds_tile, opp_seat[:, None, None].astype(jnp.int32),
            axis=1)[:, 0]
        ok_ = jnp.take_along_axis(
            sts.melds_kind, opp_seat[:, None, None].astype(jnp.int32),
            axis=1)[:, 0]
        is_peng = (ok_ == 1).astype(jnp.int32)  # (B,4), 空槽 False
        penged_tiles = penged_tiles.at[
            jnp.arange(B)[:, None], ot.astype(jnp.int32)].add(is_peng)
    risk = jnp.where(jnp.arange(28)[None, :] == RED, 0.0,
                     jnp.where(penged_tiles, 1.0,
                               jnp.where(remain >= 3, 0.4,
                                         jnp.where(remain == 2, 0.2,
                                                   jnp.where(remain == 1,
                                                             0.05, 0.0)))))
    ts = tile_shanten
    tw = tile_waits
    tr = risk
    trm = jnp.maximum(remain, 0) / 4.0
    best_sh = jnp.min(jnp.where(ts < 1.0, ts * 5.0, 99.0), axis=-1) / 5.0
    n_playable = (hand14 > 0).sum(-1).astype(jnp.float32)
    total_useful = jnp.where(ts * 5.0 == best_sh[:, None] * 5.0,
                             tw * 40.0, 0.0).sum(-1)
    cur_sh = shanten_batch(hand14) / 5.0
    extra = jnp.concatenate([
        ts, tw, tr, trm,
        jnp.stack([cur_sh, best_sh, total_useful / 40.0, n_playable / 14.0],
                  axis=-1)], axis=-1)  # (B,116)

    out = jnp.concatenate([base, extra], axis=-1)
    assert out.shape[-1] == FEAT_DIM, out.shape
    return out
