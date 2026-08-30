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
from ..rules.ting import discard_options
from .features import encode_state as _encode_v1, FEAT_DIM as _V1_DIM

RED = 27
FEAT_DIM = 628


def _derive_py(hand_counts):
    """纯 Python 版的向听派生量。留作 encode_state 的对拍基准。"""
    cur = shanten(hand_counts)
    opts = discard_options(hand_counts) if sum(hand_counts) % 3 == 2 else []
    return cur, {o["tile"]: (o["shanten"], o["wait_count"]) for o in opts}


def _derive_c(hand_counts):
    """同上, 走 C 的 LUT 向听。

    discard_options 对每个候选弃牌重算一遍 Python DFS 向听, profile 显示它占
    自对弈采集的 86%(encode_state cumtime 6.9s / 总 8s)。这里换成一次
    mj_discard_shanten 拿到全部候选, 只对听牌的候选再补一次进张统计 ——
    wait_count 的口径必须与 Python 版逐位一致: unseen = 4 - 自己手上的张数。
    """
    from ..native import native
    cur = native.shanten(list(hand_counts))
    if sum(hand_counts) % 3 != 2:
        return cur, {}
    out = {}
    h = list(hand_counts)
    for tile, s in native.discard_shanten(h):
        wc = 0
        if s == 0:
            h[tile] -= 1
            wc = native.waits_ukeire(h, [4 - n for n in h])
            h[tile] += 1
        out[tile] = (s, wc)
    return cur, out


_DERIVE = None


def _derive(hand_counts):
    """首次调用探测 C 库可用性; 不可用则退回纯 Python(结果相同, 只是慢)。"""
    global _DERIVE
    if _DERIVE is None:
        try:
            _derive_c([1] * 14 + [0] * 14)
            _DERIVE = _derive_c
        except Exception:
            _DERIVE = _derive_py
    return _DERIVE(hand_counts)


def encode_state(game, seat: int, _derive_fn=None) -> np.ndarray:
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

    cur_shanten, opts_map = (_derive_fn or _derive)(hand_counts)

    if opts_map:
        best_shanten = min(s for s, _ in opts_map.values())
        total_useful = sum(w for s, w in opts_map.values()
                           if s == best_shanten)
    else:
        best_shanten, total_useful = 5, 0

    # 每张牌的派生特征
    tile_shanten = []
    tile_waits = []
    tile_risk = []
    tile_remain = []

    for t in range(28):
        o = opts_map.get(t)
        if o is not None:
            tile_shanten.append(o[0] / 5.0)
            tile_waits.append(o[1] / 40.0)
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
