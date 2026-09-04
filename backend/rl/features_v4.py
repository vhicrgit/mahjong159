"""安康159 - 特征编码 v4 = v2(628维) + 对手模型段(90维)

对手段(每个对手 rel=1..3, 座位序 (seat+rel)%4, 与 v1 的对手副露段同序):
  - hold_probs[28]: OppTracker 后验的"该对手手里持有每种牌的概率"(0~1)
  - tenpai_prob:    该对手已听牌的概率
  - E[shanten]/5:   该对手向听数后验均值(归一)

为什么值得加: v2 只有"可见剩余张数"这种一阶信息(tile_remain), 而 tracker 是
用弃牌/碰杠序列做贝叶斯反演的后验 —— 它知道"这家连续打孤张中张, 大概率
已经听牌"这类 v2 表达不了的信息。防守(少放碰/少放杠/不加速对手)是分析器
的盲区: 分析器只看自己的 E, 对 v31n 的实测优势为 0(-0.061±0.078 打平),
要超过它只能靠输入端加信息。

tracker 用 policy=False(结构似然): policy=True 质量更高(听口命中 0.36 vs
0.02)但 3.7s/决策, 无法支撑 6 万状态的采集; no-policy 9.3ms/决策, 向听 MAE
1.12(均匀基线 1.21), hold/tenpai 概率仍有一阶以上的信息。

无 tracker 时(比如旧评估路径)回退: 对手段全 0, 网络自己学会忽略。
"""

import numpy as np

from .features_v2 import encode_state as _encode_v2, FEAT_DIM as _V2_DIM

OPP_DIM = 90                      # 3 对手 × (28 hold + 1 tenpai + 1 shanten)
FEAT_DIM = _V2_DIM + OPP_DIM      # 718


def opp_features(trackers, seat: int) -> np.ndarray:
    """trackers: {(hero_seat, opp_seat): OppTracker} 或 {opp_seat: OppTracker}。
    按 rel=1..3 的顺序输出 90 维; 缺 tracker 的对手段置 0。"""
    out = []
    for rel in (1, 2, 3):
        opp = (seat + rel) % 4
        tr = None
        if isinstance(trackers, dict):
            tr = trackers.get((seat, opp), trackers.get(opp))
        if tr is None:
            out.extend([0.0] * 30)
            continue
        hold = tr.hold_probs()
        out.extend(float(h) for h in hold)          # 28
        out.append(float(tr.tenpai_prob()))         # 1
        sd = tr.shanten_dist()
        tot = sum(sd.values()) or 1.0
        esh = sum(k * v for k, v in sd.items()) / tot
        out.append(float(min(esh, 10.0) / 5.0))     # 1
    return np.asarray(out, dtype=np.float32)


def encode_state(game, seat: int, trackers=None) -> np.ndarray:
    """v4 编码。trackers 缺省时对手段全 0(维度仍为 718, 保证兼容)。"""
    base = _encode_v2(game, seat)
    opp = opp_features(trackers or {}, seat)
    out = np.concatenate([base, opp])
    assert len(out) == FEAT_DIM
    return out
