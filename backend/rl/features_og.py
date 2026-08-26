"""安康159 - Oracle Guiding 特征编码 (盲打 v3 + 完美信息块)

Suphx 式渐进 Oracle Guiding:
- 训练输入 = 盲打特征(810) + oracle 块(336, 尾部追加)
- oracle 块训练中按保留率 γ (1→0) 做 Bernoulli 遮蔽, γ=0 后等价盲打
- 评估/部署时 oracle 块恒为 0 (encode_state_blind)

oracle 块布局 (相对座位):
- 自己未来 8 次摸牌 one-hot 序列: 8×28 = 224 (按无碰杠推算, 同 bot_oracle)
- 三家对手暗手牌计数: 3×28 = 84 (/4)
- 牌墙剩余牌型计数: 28 (/4)
合计: 810 + 336 = 1146 维
"""

import numpy as np

from .features_v3 import encode_state as _encode_v3, FEAT_DIM as BLIND_DIM

N_FUTURE = 8
ORACLE_DIM = N_FUTURE * 28 + 3 * 28 + 28  # 336
FEAT_DIM = BLIND_DIM + ORACLE_DIM         # 1146


def _future_draws(game, seat, max_draws=N_FUTURE):
    """同 bot_oracle._my_future_draws: 我出牌后下一摸牌者摸 wall[0],
    我的下次摸牌在 wall[3], 之后每隔4张; 末尾6张留给翻159。"""
    wall = game.wall
    draws = []
    idx = 3
    while idx < len(wall) - 6 and len(draws) < max_draws:
        draws.append(wall[idx])
        idx += 4
    return draws


def encode_oracle_block(game, seat: int) -> np.ndarray:
    block = np.zeros(ORACLE_DIM, dtype=np.float32)
    for i, t in enumerate(_future_draws(game, seat)):
        block[i * 28 + t] = 1.0
    off = N_FUTURE * 28
    for r in range(1, 4):
        p = game.players[(seat + r) % 4]
        for t in p.hand:
            block[off + (r - 1) * 28 + t] += 0.25
    off += 3 * 28
    for t in game.wall:
        block[off + t] += 0.25
    return block


def encode_state(game, seat: int) -> np.ndarray:
    """训练数据用: 盲打 v3 + oracle 块 (1146维)"""
    return np.concatenate([_encode_v3(game, seat),
                           encode_oracle_block(game, seat)])


def encode_state_blind(game, seat: int) -> np.ndarray:
    """评估/部署用: oracle 块恒为 0"""
    return np.concatenate([_encode_v3(game, seat),
                           np.zeros(ORACLE_DIM, dtype=np.float32)])
