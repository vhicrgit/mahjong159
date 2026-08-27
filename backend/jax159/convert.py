"""Python Game -> JAX State 转换 (GRPO 世界推演的状态注入)"""

import jax.numpy as jnp

from . import env159

KIND_MAP = {"peng": 1, "ming": 2, "an": 3, "bu": 4}


def state_from_game(g, hero_seat: int = 0) -> env159.State:
    """从任意 Python Game 构造 JAX State。
    牌墙只保留剩余部分(wall_pos=0)。对手暗牌按当前 game 实际手牌注入
    (调用方可在注入前用采样世界覆盖 g.players[i].hand)。"""
    # 明杠的放杠者从 gang_records 恢复
    ming_from = {}
    for rec in g.gang_records:
        if rec["kind"] == "ming":
            ming_from[(rec["seat"], rec["tile"])] = rec["from"]

    wall = jnp.zeros(112, dtype=jnp.int8)
    if g.wall:
        wall = wall.at[:len(g.wall)].set(jnp.array(g.wall, dtype=jnp.int8))
    hands = jnp.zeros((4, 28), dtype=jnp.int8)
    discards = jnp.zeros((4, 28), dtype=jnp.int16)
    melds_tile = jnp.zeros((4, 4), dtype=jnp.int8)
    melds_kind = jnp.zeros((4, 4), dtype=jnp.int8)
    melds_from = jnp.full((4, 4), -1, dtype=jnp.int8)
    n_melds = jnp.zeros(4, dtype=jnp.int8)
    for p in g.players:
        hc = p.hand_counts
        for t in range(28):
            if hc[t]:
                hands = hands.at[p.seat, t].set(hc[t])
        for t in range(28):
            c = p.discards.count(t)
            if c:
                discards = discards.at[p.seat, t].set(c)
        for i, m in enumerate(p.melds):
            kind = KIND_MAP.get(m.get("kind", ""), 1)
            melds_tile = melds_tile.at[p.seat, i].set(m["tile"])
            melds_kind = melds_kind.at[p.seat, i].set(kind)
            if kind == 2:
                frm = ming_from.get((p.seat, m["tile"]), -1)
                melds_from = melds_from.at[p.seat, i].set(frm)
        n_melds = n_melds.at[p.seat].set(len(p.melds))

    phase_map = {"discard_wait": 0, "react_wait": 1, "game_over": 2}
    pend_peng = jnp.zeros(4, dtype=bool)
    pend_gang = jnp.zeros(4, dtype=bool)
    for s, act in g.pending_actions.items():
        pend_peng = pend_peng.at[s].set(act.get("peng", False))
        pend_gang = pend_gang.at[s].set(act.get("gang", False))

    return env159.State(
        wall=wall,
        wall_pos=jnp.int16(0),
        wall_tail=jnp.int16(len(g.wall)),
        hands=hands,
        discards=discards,
        melds_tile=melds_tile,
        melds_kind=melds_kind,
        melds_from=melds_from,
        n_melds=n_melds,
        turn=jnp.int8(g.turn),
        phase=jnp.int8(phase_map[g.phase]),
        last_discard=jnp.int8(g.last_discard if g.last_discard is not None
                              else -1),
        last_discarder=jnp.int8(g.last_discarder
                                if g.last_discarder is not None else 0),
        pend_peng=pend_peng,
        pend_gang=pend_gang,
        winner=jnp.int8(-1 if g.winner is None else g.winner),
        win_kind=jnp.int8({"zimo": 1, "gangshang": 2, "tianhu": 3}
                          .get(g.win_kind or "", 0)),
        n_159=jnp.int8(g.n_159),
        scores=jnp.array([p.score_delta for p in g.players], dtype=jnp.int16),
        draws=jnp.int16(0),
    )


def batch_states(states):
    """把若干单个 State 堆成批量 State (首轴=批量)。"""
    import jax.numpy as jnp
    return env159.State(
        **{f: jnp.stack([getattr(s, f) for s in states], axis=0)
           for f in env159.State._fields})


def slice_state(sts, i, j):
    """批量 State 取 [i:j) 切片(特征分块用)"""
    return env159.State(**{f: getattr(sts, f)[i:j]
                           for f in env159.State._fields})
