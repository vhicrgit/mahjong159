"""安康159 - 纯 JAX GPU 环境 (MahJax 路线移植)

状态全部为 jnp 数组, reset/step/legal 全部可 jit, vmap 跨局并行。
与 backend/game/engine.py 的 Python 引擎逐状态对拍(backend/jax159/test_parity.py)。

动作编码 (int):
- 0-27: 打出该牌 (phase=discard_wait)
- 28-54: 暗杠 tile=a-28
- 55-81: 补杠 tile=a-55
- react_wait 时: 0=过, 1=碰, 2=明杠

phase: 0=discard_wait, 1=react_wait, 2=game_over
"""

import os
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

RED = 27
N_TILES = 28
WALL_N = 112
MAX_MELDS = 4

# phase
P_DISCARD = 0
P_REACT = 1
P_OVER = 2

# win_kind
W_NONE, W_ZIMO, W_GANG, W_TIANHU = 0, 1, 2, 3

_IS159 = jnp.array([1 if (t < 27 and t % 9 in (0, 4, 8)) else 0
                    for t in range(28)], dtype=jnp.int32)
_POW5 = jnp.array([5 ** i for i in range(9)], dtype=jnp.int32)

_M0 = None
_M1 = None


def load_win_tables(path="models/win_table.npz"):
    global _M0, _M1
    if _M0 is None:
        z = np.load(path)
        _M0 = jnp.asarray(z["M0"])   # (5^9, 5) bool
        _M1 = jnp.asarray(z["M1"])
    return _M0, _M1


class State(NamedTuple):
    wall: jax.Array          # (112,) int8 固定数组; 头=wall_pos, 尾=wall_tail
    wall_pos: jax.Array      # int16 下一个摸牌位置(头)
    wall_tail: jax.Array     # int16 牌堆尾(杠补牌从 wall_tail-1 取)
    hands: jax.Array         # (4,28) int8 计数
    discards: jax.Array      # (4,28) int16 计数(不记顺序)
    melds_tile: jax.Array    # (4,4) int8
    melds_kind: jax.Array    # (4,4) int8  0空 1碰 2明杠 3暗杠 4补杠
    melds_from: jax.Array    # (4,4) int8  明杠的放杠者, 其他 -1
    n_melds: jax.Array       # (4,) int8
    turn: jax.Array          # int8
    phase: jax.Array         # int8
    last_discard: jax.Array  # int8
    last_discarder: jax.Array
    pend_peng: jax.Array     # (4,) bool
    pend_gang: jax.Array     # (4,) bool
    winner: jax.Array        # int8, -1 无
    win_kind: jax.Array      # int8
    n_159: jax.Array         # int8
    scores: jax.Array        # (4,) int16
    draws: jax.Array         # int16 总摸牌数(步数惩罚用)


ALL_SPLITS = [(r0, r1, r2)
              for r0 in range(5) for r1 in range(5) for r2 in range(5) if r0 + r1 + r2 <= 4]


def _is_win_hand(hand: jax.Array) -> jax.Array:
    """hand: (28,) 或 (B,28) int8 计数(3n+2)。查表判胡。"""
    M0, M1 = _M0, _M1
    single = hand.ndim == 1
    if single:
        hand = hand[None, :]
    c = hand.astype(jnp.int32)
    c0 = (c[:, 0:9] * _POW5).sum(1)
    c1 = (c[:, 9:18] * _POW5).sum(1)
    c2 = (c[:, 18:27] * _POW5).sum(1)
    red = c[:, 27]

    win = jnp.zeros((c.shape[0],), dtype=jnp.bool_)
    for r0, r1, r2 in ALL_SPLITS:
        used = r0 + r1 + r2
        ok = used <= red
        left = red - used
        a0, b0 = M0[c0, r0], M1[c0, r0]
        a1, b1 = M0[c1, r1], M1[c1, r1]
        a2, b2 = M0[c2, r2], M1[c2, r2]
        pair_in_suit = (b0 & a1 & a2) | (a0 & b1 & a2) | (a0 & a1 & b2)
        win = win | (ok & (
            ((left % 3 == 0) & pair_in_suit) |
            ((left == 2) & a0 & a1 & a2)))
    return win[0] if single else win


def _settle(state: State, winner, win_kind) -> State:
    """胡牌结算: 杠分 + 159 分"""
    wall_rem = state.wall_tail - state.wall_pos
    # 翻牌: 剩余牌堆头6张; 不足6张按0(与Python一致)
    flip_idx = state.wall_pos + jnp.arange(6)
    flip_tiles = state.wall[jnp.minimum(flip_idx, WALL_N - 1)]
    n = jnp.where(wall_rem >= 6,
                  _IS159[flip_tiles].sum(), jnp.int32(0)).astype(jnp.int16)
    scores = jnp.zeros(4, dtype=jnp.int16)
    # 杠分
    for seat in range(4):
        for mi in range(MAX_MELDS):
            kind = state.melds_kind[seat, mi]
            owner = seat
            frm = state.melds_from[seat, mi]
            # 明杠: 放杠者 -3, 杠者 +3
            scores = scores.at[owner].add(
                jnp.where(kind == 2, 3, 0).astype(jnp.int16))
            scores = scores.at[frm].add(
                jnp.where(kind == 2, -3, 0).astype(jnp.int16))
            # 暗杠/补杠: 其他三家各 -1, 杠者 +3
            other_gang = (kind == 3) | (kind == 4)
            scores = scores.at[owner].add(
                jnp.where(other_gang, 3, 0).astype(jnp.int16))
            for o in range(4):
                scores = scores.at[o].add(
                    jnp.where(other_gang & (o != owner), -1, 0)
                    .astype(jnp.int16))
    # 胡牌分: 输家各赔 n+1
    per = (n + 1).astype(jnp.int16)
    for o in range(4):
        scores = scores.at[o].add(
            jnp.where(o == winner, 3 * per, -per))
    return state._replace(scores=scores, winner=winner.astype(jnp.int8),
                          win_kind=win_kind, n_159=n.astype(jnp.int8),
                          phase=jnp.int8(P_OVER))


def _gang_draw(state: State, seat) -> State:
    """杠后从牌堆尾补牌; 检查杠上花。墙空则黄庄。"""
    wall_len = state.wall_tail - state.wall_pos
    empty = wall_len <= 0
    t = state.wall[state.wall_tail - 1]
    t = jnp.where(empty, 0, t)
    new_pos_tail = state.wall_tail - 1
    hands = state.hands.at[seat, t].add(
        jnp.where(empty, 0, 1).astype(jnp.int8))
    st = state._replace(
        wall_tail=jnp.where(empty, state.wall_tail, new_pos_tail)
        .astype(jnp.int16),
        hands=hands,
        draws=(state.draws + jnp.where(empty, 0, 1)).astype(jnp.int16))
    win = (~empty) & _is_win_hand(st.hands[seat])
    st = jax.lax.cond(
        win, lambda s: _settle(s, seat, jnp.int8(W_GANG)), lambda s: s, st)
    st = jax.lax.cond(
        empty & (st.phase != P_OVER),
        lambda s: s._replace(phase=jnp.int8(P_OVER),
                             winner=jnp.int8(-1)),
        lambda s: s, st)
    return st


def _next_draw(state: State) -> State:
    """下家摸牌; 墙<=6 黄庄(杠分不结, 分数保持0)。"""
    nxt = ((state.last_discarder + 1) % 4).astype(jnp.int8)
    wall_len = state.wall_tail - state.wall_pos
    hz = wall_len <= 6
    t = state.wall[jnp.minimum(state.wall_pos, WALL_N - 1)]
    hands = state.hands.at[nxt, t].add(jnp.where(hz, 0, 1).astype(jnp.int8))
    st = state._replace(
        turn=nxt,
        wall_pos=(state.wall_pos + jnp.where(hz, 0, 1)).astype(jnp.int16),
        hands=hands,
        phase=jnp.where(hz, P_OVER, P_DISCARD).astype(jnp.int8),
        draws=(state.draws + jnp.where(hz, 0, 1)).astype(jnp.int16))
    win = (~hz) & _is_win_hand(st.hands[nxt])
    st = jax.lax.cond(
        win, lambda s: _settle(s, nxt, jnp.int8(W_ZIMO)), lambda s: s, st)
    st = jax.lax.cond(
        hz, lambda s: s._replace(winner=jnp.int8(-1)), lambda s: s, st)
    return st


def _do_discard(state: State, tile) -> State:
    seat = state.turn
    hands = state.hands.at[seat, tile].add(-1)
    discards = state.discards.at[seat, tile].add(1)
    st = state._replace(hands=hands, discards=discards,
                        last_discard=tile.astype(jnp.int8),
                        last_discarder=seat)
    # 其他家碰/杠可能(红中不可被碰杠)
    cnt = st.hands[:, tile]
    is_red = tile == RED
    peng = (cnt >= 2) & (~is_red)
    gang = (cnt >= 3) & (~is_red)
    own = jnp.arange(4) == seat
    pend_peng = peng & (~own)
    pend_gang = gang & (~own)
    any_react = (pend_peng | pend_gang).any()
    st = st._replace(pend_peng=pend_peng, pend_gang=pend_gang)
    st = jax.lax.cond(
        any_react,
        lambda s: s._replace(phase=jnp.int8(P_REACT)),
        _next_draw, st)
    return st


def _add_meld(state: State, seat, tile, kind, frm) -> State:
    i = state.n_melds[seat]
    mt = state.melds_tile.at[seat, i].set(tile.astype(jnp.int8))
    mk = state.melds_kind.at[seat, i].set(jnp.int8(kind))
    mf = state.melds_from.at[seat, i].set(jnp.int8(frm))
    return state._replace(melds_tile=mt, melds_kind=mk, melds_from=mf,
                          n_melds=state.n_melds.at[seat].add(1)
                          .astype(jnp.int8))


def _react_pass(state: State) -> State:
    # 唯一待响应者(数学上至多一家)清掉后进入下家摸牌
    st = state._replace(pend_peng=jnp.zeros(4, bool),
                        pend_gang=jnp.zeros(4, bool))
    return _next_draw(st)


def _react_peng(state: State) -> State:
    seat = jnp.argmax(state.pend_peng.astype(jnp.int8)).astype(jnp.int8)
    t = state.last_discard
    hands = state.hands.at[seat, t].add(-2)
    st = state._replace(hands=hands)
    st = _add_meld(st, seat, t, 1, state.last_discarder)
    # 被碰的牌从弃牌堆回收
    st = st._replace(
        discards=st.discards.at[st.last_discarder, t].add(-1),
        pend_peng=jnp.zeros(4, bool), pend_gang=jnp.zeros(4, bool),
        turn=seat, phase=jnp.int8(P_DISCARD))
    return st


def _react_gang(state: State) -> State:
    seat = jnp.argmax(state.pend_gang.astype(jnp.int8)).astype(jnp.int8)
    t = state.last_discard
    hands = state.hands.at[seat, t].add(-3)
    st = state._replace(hands=hands)
    st = _add_meld(st, seat, t, 2, state.last_discarder)
    st = st._replace(
        discards=st.discards.at[st.last_discarder, t].add(-1),
        pend_peng=jnp.zeros(4, bool), pend_gang=jnp.zeros(4, bool),
        turn=seat, phase=jnp.int8(P_DISCARD))
    return _gang_draw(st, seat)


def _an_gang(state: State, tile) -> State:
    seat = state.turn
    hands = state.hands.at[seat, tile].add(-4)
    st = state._replace(hands=hands)
    st = _add_meld(st, seat, tile, 3, -1)
    return _gang_draw(st, seat)


def _bu_gang(state: State, tile) -> State:
    seat = state.turn
    hands = state.hands.at[seat, tile].add(-1)
    st = state._replace(hands=hands)
    # 已有碰的标记升级为补杠
    match = (st.melds_tile[seat] == tile) & (st.melds_kind[seat] == 1)
    i = jnp.argmax(match.astype(jnp.int8))
    mk = st.melds_kind.at[seat, i].set(jnp.int8(4))
    st = st._replace(melds_kind=mk)
    return _gang_draw(st, seat)


def step(state: State, action) -> State:
    a = action.astype(jnp.int32)
    return jax.lax.switch(
        state.phase.astype(jnp.int32),
        [  # P_DISCARD
            lambda s: jax.lax.switch(
                jnp.where(a < 28, 0, jnp.where(a < 55, 1, 2)),
                [lambda s2: _do_discard(s2, a),
                 lambda s2: _an_gang(s2, a - 28),
                 lambda s2: _bu_gang(s2, a - 55)], s),
            # P_REACT
            lambda s: jax.lax.switch(
                a, [_react_pass, _react_peng, _react_gang], s),
            # P_OVER: 不动
            lambda s: s,
        ], state)


def legal_actions(state: State) -> jax.Array:
    """返回 (82,) bool 合法动作掩码。"""
    m = jnp.zeros(82, dtype=bool)
    is_disc = state.phase == P_DISCARD
    is_react = state.phase == P_REACT
    hand = state.hands[state.turn]
    # 出牌
    m = m.at[:28].set(is_disc & (hand > 0))
    # 暗杠: 手持4张(非红中)
    an = is_disc & (hand == 4)
    m = m.at[28:55].set(an[:27])
    # 补杠: 有碰且手持该牌
    bu_tiles = is_disc & (state.melds_kind[state.turn] == 1) & \
        (hand[state.melds_tile[state.turn]] > 0)
    bu = jnp.zeros(27, bool)
    bu = bu.at[state.melds_tile[state.turn].clip(0, 26)].set(bu_tiles)
    m = m.at[55:82].set(bu)
    # 反应
    any_peng = state.pend_peng.any() & is_react
    any_gang = state.pend_gang.any() & is_react
    m = m.at[0].set(m[0] | is_react)  # 过 = action 0 (与"打1条"共享编码,
    # 由 phase 区分语义)
    m = m.at[1].set(m[1] | (is_react & any_peng))
    m = m.at[2].set(m[2] | (is_react & any_gang))
    return m


def reset_from_wall(wall: jax.Array) -> State:
    """用给定牌墙(112, int8)建初始状态; 含天胡判定。"""
    hands = jnp.zeros((4, 28), dtype=jnp.int8)
    for p in range(4):
        dealt = wall[p * 13:(p + 1) * 13].astype(jnp.int32)
        counts = (dealt[:, None] == jnp.arange(28)).sum(0)
        hands = hands.at[p].set(counts.astype(jnp.int8))
    # 庄家多摸一张
    extra = wall[52]
    hands = hands.at[0, extra].add(1)
    st = State(
        wall=wall, wall_pos=jnp.int16(53), wall_tail=jnp.int16(WALL_N),
        hands=hands,
        discards=jnp.zeros((4, 28), dtype=jnp.int16),
        melds_tile=jnp.zeros((4, MAX_MELDS), dtype=jnp.int8),
        melds_kind=jnp.zeros((4, MAX_MELDS), dtype=jnp.int8),
        melds_from=jnp.full((4, MAX_MELDS), -1, dtype=jnp.int8),
        n_melds=jnp.zeros(4, dtype=jnp.int8),
        turn=jnp.int8(0), phase=jnp.int8(P_DISCARD),
        last_discard=jnp.int8(-1), last_discarder=jnp.int8(0),
        pend_peng=jnp.zeros(4, bool), pend_gang=jnp.zeros(4, bool),
        winner=jnp.int8(-1), win_kind=jnp.int8(W_NONE),
        n_159=jnp.int8(0), scores=jnp.zeros(4, dtype=jnp.int16),
        draws=jnp.int16(0))
    # 天胡: 庄家开局14张即胡
    th = _is_win_hand(st.hands[0])
    st = jax.lax.cond(
        th, lambda s: _settle(s, jnp.int8(0), jnp.int8(W_TIANHU)),
        lambda s: s, st)
    return st


def reset(key) -> State:
    tiles = jnp.tile(jnp.arange(28, dtype=jnp.int8), 4)
    wall = jax.random.permutation(key, tiles)
    return reset_from_wall(wall)

# jit 版本(在单个局面上; vmap 版调用方自行 vmap)
step_jit = jax.jit(step)
legal_jit = jax.jit(legal_actions)
