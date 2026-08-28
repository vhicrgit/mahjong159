"""批量世界注入: 跳过 Python deepcopy + action_discard, 直接 NumPy 填充 JAX State。"""

import random

import numpy as np

from .env159 import State


def _sample_world(snap, rng, hero):
    from backend.rl.world_grpo import sample_world
    return sample_world(snap, rng, hero_seat=hero)


def build_world_states(snaps, cand_lists, n_worlds, world_seed):
    rng = random.Random(world_seed)

    # 预取基准 & 世界采样
    bases = []
    world_data = []
    for si, (snap, hero) in enumerate(snaps):
        from .convert import state_from_game, batch_states
        bst = batch_states([state_from_game(snap)])
        bases.append(dict(
            hands=np.asarray(bst.hands[0], dtype=np.int8),
            discards=np.asarray(bst.discards[0], dtype=np.int16),
            mt=np.asarray(bst.melds_tile[0]),
            mk=np.asarray(bst.melds_kind[0]),
            mf=np.asarray(bst.melds_from[0]),
            nm=np.asarray(bst.n_melds[0]),
            hero=hero))
        wd = []
        for _ in range(n_worlds):
            hmap, wall = _sample_world(snap, rng, hero)
            opp = np.zeros((4, 28), dtype=np.int8)
            for s, hl in hmap.items():
                for t in hl:
                    opp[s, t] += 1
            w = np.zeros(112, dtype=np.int8)
            w[:len(wall)] = np.array(wall, dtype=np.int8)
            wd.append((opp, w, len(wall)))
        world_data.append(wd)

    B = sum(len(c) * n_worlds for c in cand_lists)
    H = np.zeros((B, 4, 28), dtype=np.int8)
    D = np.zeros((B, 4, 28), dtype=np.int16)
    DW = np.zeros(B, dtype=np.int16)  # draws 初值(react 路径补 1)
    W = np.zeros((B, 112), dtype=np.int8)
    WT = np.zeros(B, dtype=np.int16)
    MT = np.zeros((B, 4, 4), dtype=np.int8)
    MK = np.zeros((B, 4, 4), dtype=np.int8)
    MF = np.zeros((B, 4, 4), dtype=np.int8)
    NM = np.zeros((B, 4), dtype=np.int8)
    TN = np.zeros(B, dtype=np.int8)
    LD = np.full(B, -1, dtype=np.int8)
    LR = np.full(B, -1, dtype=np.int8)
    PH = np.zeros(B, dtype=np.int8)
    PP = np.zeros((B, 4), dtype=bool)
    PG = np.zeros((B, 4), dtype=bool)
    meta = [(0, 0, 0)] * B

    idx = 0
    for si, base in enumerate(bases):
        hero = base["hero"]
        nxt = (hero + 1) % 4
        h0, d0 = base["hands"], base["discards"]
        mt, mk, mf, nm = base["mt"], base["mk"], base["mf"], base["nm"]
        wd = world_data[si]
        for tile in cand_lists[si]:
            for wi in range(n_worlds):
                opp, w, wlen = wd[wi]
                H[idx] = h0
                H[idx, hero, tile] -= 1
                for s in range(4):
                    if s != hero:
                        H[idx, s] = opp[s]
                D[idx] = d0
                D[idx, hero, tile] += 1
                # 候选能否被对手碰/杠: 是则进入 react_wait(与真实引擎一致,
                # 由推演决策碰不碰), 否则直接下家摸牌
                can_react = any(int(opp[s, tile]) >= 2
                                for s in range(4) if s != hero)
                if can_react:
                    PH[idx] = np.int8(1)
                    TN[idx] = np.int8(hero)
                    W[idx] = w
                    WT[idx] = np.int16(wlen)
                    DW[idx] = np.int16(0)  # 与旧版 react_wait 快照一致(draws 由推演计数)
                    for s in range(4):
                        if s != hero:
                            PP[idx, s] = int(opp[s, tile]) >= 2
                            PG[idx, s] = int(opp[s, tile]) >= 3
                else:
                    PH[idx] = np.int8(0)
                    TN[idx] = np.int8(nxt)
                    draw_tile = int(w[0])
                    H[idx, nxt, draw_tile] += 1
                    W[idx] = np.roll(w, -1)
                    W[idx, -1] = 0
                    WT[idx] = np.int16(wlen - 1)
                    DW[idx] = np.int16(0)  # 旧版快照已摸牌但 draws=0
                MT[idx] = mt
                MK[idx] = mk
                MF[idx] = mf
                NM[idx] = nm
                LD[idx] = np.int8(tile)
                LR[idx] = np.int8(hero)
                meta[idx] = (si, tile, hero)
                idx += 1

    return State(
        wall=W, wall_pos=np.zeros(B, dtype=np.int16), wall_tail=WT,
        hands=H, discards=D,
        melds_tile=MT, melds_kind=MK, melds_from=MF, n_melds=NM,
        turn=TN, phase=PH,
        last_discard=LD, last_discarder=LR,
        pend_peng=PP, pend_gang=PG,
        winner=np.full(B, -1, dtype=np.int8),
        win_kind=np.zeros(B, dtype=np.int8), n_159=np.zeros(B, dtype=np.int8),
        scores=np.zeros((B, 4), dtype=np.int16),
        draws=DW,
    ), meta, []