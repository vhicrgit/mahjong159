"""安康159 - 局面特征编码(模仿 Mortal 的特征平面设计)

从某玩家视角把局面编码成定长向量,供神经网络输入。

布局(全部拼接为一维向量):
- 手牌计数:        28 x 4 (one-hot 张数)      = 112
- 自己副露:        28 x 4  (每种牌的副露张数)  = 112
- 其他三家副露:    3 x 28 x 2(碰/杠标记压缩) = 168
- 各家弃牌计数:    4 x 28                     = 112
- 全局标量: 牌堆剩余/112, 是否庄家, 回合进度, 可见159数/36,
           各家副露数 x3, 自己红中数/4         = 8
合计 512 维
"""

import numpy as np

RED = 27
FEAT_DIM = 512


def _oh4(n: int) -> list[float]:
    """张数 0-4 的 one-hot"""
    v = [0.0, 0.0, 0.0, 0.0]
    if 0 <= n <= 3:
        v[n] = 1.0
    else:
        v[3] = 1.0
    return v


def encode_state(game, seat: int) -> np.ndarray:
    """把 game 当前局面从 seat 视角编码为 FEAT_DIM 维向量"""
    feats: list[float] = []
    me = game.players[seat]

    # 手牌计数 one-hot (28 x 4)
    hand_counts = me.hand_counts
    for t in range(28):
        feats.extend(_oh4(hand_counts[t]))

    # 自己副露(28 x 4): 每种牌在副露中的张数
    meld_counts = [0] * 28
    for m in me.melds:
        meld_counts[m["tile"]] += 3 if m["type"] == "peng" else 4
    for t in range(28):
        feats.extend(_oh4(meld_counts[t]))

    # 其他三家副露(每家: 碰28 + 杠28 = 56, 共168)
    for rel in range(1, 4):
        opp = game.players[(seat + rel) % 4]
        peng = [0.0] * 28
        gang = [0.0] * 28
        for m in opp.melds:
            if m["type"] == "peng":
                peng[m["tile"]] = 1.0
            else:
                gang[m["tile"]] = 1.0
        feats.extend(peng)
        feats.extend(gang)

    # 各家弃牌计数(4 x 28, 归一化/4)
    for rel in range(4):
        p = game.players[(seat + rel) % 4]
        dc = [0.0] * 28
        for t in p.discards:
            dc[t] += 1.0 / 4.0
        feats.extend(dc)

    # 全局标量 (8)
    wall_ratio = game.wall_remaining() / 112.0
    is_dealer = 1.0 if game.dealer == seat else 0.0
    progress = 1.0 - wall_ratio
    # 可见159数
    seen_159 = 0
    for p in game.players:
        for t in p.discards:
            if t < 27 and t % 9 in (0, 4, 8):
                seen_159 += 1
        for m in p.melds:
            t = m["tile"]
            if t < 27 and t % 9 in (0, 4, 8):
                seen_159 += 3 if m["type"] == "peng" else 4
    for t in range(27):
        if t % 9 in (0, 4, 8):
            seen_159 += hand_counts[t]
    feats.append(wall_ratio)
    feats.append(is_dealer)
    feats.append(progress)
    feats.append(seen_159 / 36.0)
    for rel in range(1, 4):
        opp = game.players[(seat + rel) % 4]
        feats.append(len(opp.melds) / 4.0)
    feats.append(hand_counts[RED] / 4.0)

    assert len(feats) == FEAT_DIM, f"特征维度错误: {len(feats)} != {FEAT_DIM}"
    return np.asarray(feats, dtype=np.float32)
