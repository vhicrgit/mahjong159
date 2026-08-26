"""安康159 - 增强特征编码 v3

在 v2 (628维) 基础上补齐三类信息缺口:
1. 任意向听的有效进张 (v2 的"打后进张数"仅听牌时非零, 非听牌阶段模型
   看不到牌效进度) —— 每张可打牌: 打后能降向听的进张类型数 + 按未见
   张数加权的进张数。
2. 弃牌时序 (v2 只有计数无顺序) —— 4家(相对座位)×28, 指数衰减 recency。
3. 副露时机 (v2 只有有无) —— 3个对手: 副露数/杠数/首次与最近副露时的
   牌墙比例 (早副露+高牌墙 = 快攻牌型信号)。

布局 (尾部追加, 手牌 one-hot 仍在 [0,112), 不破坏 legal_mask_from_feats):
- v2 特征:                628 维 (不变)
- 每张牌打后有效进张类型数: 28 维 (/28)
- 每张牌打后加权有效进张:   28 维 (/60)
- 弃牌 recency 4×28:      112 维 (0.85^距今步数)
- 副露时机 3对手×4:        12 维
- 全局: 最佳类型数, 最佳加权进张: 2 维
合计: 628 + 182 = 810 维
"""

import numpy as np

from ..rules.ting import useful_draws
from .features_v2 import encode_state as _encode_v2

FEAT_DIM = 810
DECAY = 0.85
WALL0 = 59.0  # 发牌后初始牌墙 112 - 13*4 - 1


def encode_state(game, seat: int) -> np.ndarray:
    """增强版特征编码 v3 (810维)"""
    base = _encode_v2(game, seat)  # 628

    me = game.players[seat]
    hand_counts = me.hand_counts

    visible = [0] * 28
    for p in game.players:
        for t in p.discards:
            visible[t] += 1
        for m in p.melds:
            visible[m["tile"]] += 3 if m["type"] == "peng" else 4
    for t in range(28):
        visible[t] += hand_counts[t]
    unseen = [max(0, 4 - visible[t]) for t in range(28)]

    # 1. 任意向听的有效进张 (仅 14 张手牌时)
    types_arr = [0.0] * 28
    weight_arr = [0.0] * 28
    counts = list(hand_counts)
    if sum(counts) % 3 == 2:
        for t in range(28):
            if counts[t] <= 0:
                continue
            counts[t] -= 1
            ud = useful_draws(counts)
            counts[t] += 1
            n_types = 0
            w = 0
            for d in ud:
                if unseen[d] > 0:
                    n_types += 1
                    w += unseen[d]
            types_arr[t] = n_types / 28.0
            weight_arr[t] = w / 60.0
    best_types = max(types_arr)
    best_weight = max(weight_arr)

    # 2. 弃牌 recency (相对座位: r=0 自己, 1 下家, 2 对家, 3 上家)
    recency = [0.0] * 112
    for r in range(4):
        p = game.players[(seat + r) % 4]
        n = len(p.discards)
        for idx, t in enumerate(p.discards):
            v = DECAY ** (n - 1 - idx)
            off = r * 28 + t
            if v > recency[off]:
                recency[off] = v

    # 3. 副露时机 (3个对手)
    meld_feats = []
    for r in range(1, 4):
        p = game.players[(seat + r) % 4]
        n_melds = len(p.melds)
        n_gang = sum(1 for m in p.melds if m["type"] == "gang")
        wrs = [m["wr"] for m in p.melds if "wr" in m]
        first_frac = wrs[0] / WALL0 if wrs else 0.0
        last_frac = wrs[-1] / WALL0 if wrs else 0.0
        meld_feats += [n_melds / 4.0, n_gang / 4.0, first_frac, last_frac]

    extra = types_arr + weight_arr + recency + meld_feats + [
        best_types, best_weight,
    ]

    result = np.concatenate([base, np.asarray(extra, dtype=np.float32)])
    assert len(result) == FEAT_DIM, f"特征维度: {len(result)} != {FEAT_DIM}"
    return result
