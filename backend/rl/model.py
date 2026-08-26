"""安康159 - 神经网络模型

结构: 输入投影 -> N个残差块 -> Q头(每张牌的Q值, 28维) + value头(期望收益)

Mortal 路线: DQN + CQL, Q值头替代策略头。
- Q头: 输出每个动作的Q值, 推理时取 argmax
- value头: 辅助损失, 估计局面期望收益

模型大小通过 config 调节:
- tiny:  hidden=128, blocks=4   (验证闭环用, CPU/MPS 秒级训练)
- small: hidden=256, blocks=8
- base:  hidden=512, blocks=12  (接近 Mortal 量级, 适合 H20)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .features_v2 import FEAT_DIM

N_ACTIONS = 28  # 打出哪种牌


class ResBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = self.fc2(h)
        return F.relu(x + h)


class MahjongNet(nn.Module):
    def __init__(self, hidden_dim: int = 128, num_blocks: int = 4,
                 feat_dim: int = FEAT_DIM):
        super().__init__()
        self.feat_dim = feat_dim
        self.input_proj = nn.Linear(feat_dim, hidden_dim)
        self.blocks = nn.ModuleList(ResBlock(hidden_dim) for _ in range(num_blocks))
        self.q_head = nn.Linear(hidden_dim, N_ACTIONS)       # Q值头
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x):
        h = F.relu(self.input_proj(x))
        for b in self.blocks:
            h = b(h)
        q_values = self.q_head(h)              # (B, 28) Q值
        value = self.value_head(h).squeeze(-1)  # (B,)
        return q_values, value

    def q(self, x, legal_mask=None):
        """返回合法动作上的 Q 值"""
        q_values, _ = self.forward(x)
        if legal_mask is not None:
            q_values = q_values.masked_fill(~legal_mask, -1e9)
        return q_values

    def policy(self, x, legal_mask=None):
        """从 Q 值导出策略 (softmax over Q)"""
        q_values = self.q(x, legal_mask)
        return F.softmax(q_values, dim=-1)

    def best_action(self, x, legal_mask=None):
        """贪心选择最优动作"""
        q_values = self.q(x, legal_mask)
        return q_values.argmax(dim=-1)


MODEL_CONFIGS = {
    "tiny": dict(hidden_dim=128, num_blocks=4),
    "small": dict(hidden_dim=256, num_blocks=8),
    "base": dict(hidden_dim=512, num_blocks=12),
    "large": dict(hidden_dim=1024, num_blocks=20),
}


def build_model(size: str = "tiny", feat_dim: int = FEAT_DIM) -> MahjongNet:
    return MahjongNet(**MODEL_CONFIGS[size], feat_dim=feat_dim)


def legal_discard_mask(hand_counts: list[int]) -> torch.Tensor:
    """可打出的牌(手里有的)"""
    m = torch.zeros(N_ACTIONS, dtype=torch.bool)
    for t in range(N_ACTIONS):
        if hand_counts[t] > 0:
            m[t] = True
    return m
