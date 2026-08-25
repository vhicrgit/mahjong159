"""安康159 - 增强特征编码 v2

在 v1 (512维) 基础上加入规则Bot决策所用的派生特征:
- 每张可打牌的: 打后向听数, 打后进张数, 放杠风险
- 全局: 当前向听数, 最佳向听数, 有效进张总数

这让模型不需要从原始 one-hot 自己"算出"向听, 直接看到决策依据。

布局:
- v1 基础特征: 512 维 (不变)
- 每张牌的打后向听数: 28 维 (归一化 /5)
- 每张牌的打后进张数: 28 维 (归一化 /40)
- 每张牌的放杠风险:   28 维 (0~1)
- 每张牌的可见剩余:   28 维 (归一化 /4)
- 全局: 当前向听/5, 最佳向听/5, 有效进张/40, 当前可打牌数/14
合计: 512 + 112 + 4 = 628 维
"""

import numpy as np

from ..rules.win import shanten
from ..rules.ting import discard_options, waiting_tiles, useful_draws
from .features import encode_state as _encode_v1, FEAT_DIM as _V1_DIM

RED = 27
FEAT_DIM = 628


def encode_state(game, seat: int) -> np.ndarray:
    """增强版特征编码 (628维)"""
    base = _encode_v1(game, seat)  # 512

    me = game.players[seat]
    hand_counts = me.hand_counts

    # 可见计数(弃牌+副露+自己手牌)
    visible = [0] * 28
    for p in game.players:
        for t in p.discards:
            visible[t] += 1
        for m in p.melds:
            visible[m["tile"]] += 3 if m["type"] == "peng" else 4
    for t in range(28):
        visible[t] += hand_counts[t]

    # 当前向听 & 有效进张
    cur_shanten = shanten(hand_counts) if sum(hand_counts) % 3 == 1 else shanten(hand_counts)
    # discard_options 已经在算每个 tile 打后的 shanten/waits
    opts = discard_options(hand_counts) if sum(hand_counts) % 3 == 2 else []
    opts_map = {o["tile"]: o for o in opts}

    best_shanten = min((o["shanten"] for o in opts), default=5)
    total_useful = sum(o["wait_count"] for o in opts if o["shanten"] == best_shanten)

    # 每张牌的派生特征
    tile_shanten = []
    tile_waits = []
    tile_risk = []
    tile_remain = []

    for t in range(28):
        o = opts_map.get(t)
        if o is not None:
            tile_shanten.append(o["shanten"] / 5.0)
            tile_waits.append(o["wait_count"] / 40.0)
        else:
            tile_shanten.append(1.0)  # 不可打
            tile_waits.append(0.0)

        # 风险
        remain = max(0, 4 - visible[t])
        if t == RED:
            risk = 0.0
        elif remain >= 3:
            risk = 0.4
        elif remain == 2:
            risk = 0.2
        elif remain == 1:
            risk = 0.05
        else:
            risk = 0.0
        # 对手碰了这张牌则风险极高
        for p in game.players:
            if p.seat != seat and any(
                    m["tile"] == t and m["type"] == "peng" for m in p.melds):
                risk = 1.0
                break
        tile_risk.append(risk)
        tile_remain.append(remain / 4.0)

    n_playable = sum(1 for t in range(28) if hand_counts[t] > 0)

    extra = tile_shanten + tile_waits + tile_risk + tile_remain + [
        cur_shanten / 5.0,
        best_shanten / 5.0,
        total_useful / 40.0,
        n_playable / 14.0,
    ]

    result = np.concatenate([base, np.asarray(extra, dtype=np.float32)])
    assert len(result) == FEAT_DIM, f"特征维度: {len(result)} != {FEAT_DIM}"
    return result
