"""安康159 - 神经网络模型(模仿 Mortal 的 value/policy 双头结构)

结构: 输入投影 -> N个残差块 -> policy头(打哪张牌, 28类) + value头(期望收益)
Mortal 原版在牌位序列上用 Conv1D, 这里用全连接残差块(小规模验证足够,
接口保持一致, 之后换 Conv1D 只需替换 body)。

模型大小通过 config 调节:
- tiny:  hidden=128, blocks=4   (验证闭环用, CPU/MPS 秒级训练)
- small: hidden=256, blocks=8
- base:  hidden=512, blocks=12  (接近 Mortal 量级, 适合 H20)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .features import FEAT_DIM

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
    def __init__(self, hidden_dim: int = 128, num_blocks: int = 4):
        super().__init__()
        self.input_proj = nn.Linear(FEAT_DIM, hidden_dim)
        self.blocks = nn.ModuleList(ResBlock(hidden_dim) for _ in range(num_blocks))
        self.policy_head = nn.Linear(hidden_dim, N_ACTIONS)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x):
        h = F.relu(self.input_proj(x))
        for b in self.blocks:
            h = b(h)
        logits = self.policy_head(h)          # (B, 28)
        value = self.value_head(h).squeeze(-1)  # (B,)
        return logits, value

    def policy(self, x, legal_mask=None):
        """返回合法动作上的概率分布"""
        logits, _ = self.forward(x)
        if legal_mask is not None:
            logits = logits.masked_fill(~legal_mask, -1e9)
        return F.softmax(logits, dim=-1)


MODEL_CONFIGS = {
    "tiny": dict(hidden_dim=128, num_blocks=4),
    "small": dict(hidden_dim=256, num_blocks=8),
    "base": dict(hidden_dim=512, num_blocks=12),
}


def build_model(size: str = "tiny") -> MahjongNet:
    return MahjongNet(**MODEL_CONFIGS[size])


def legal_discard_mask(hand_counts: list[int]) -> torch.Tensor:
    """可打出的牌(手里有的)"""
    m = torch.zeros(N_ACTIONS, dtype=torch.bool)
    for t in range(N_ACTIONS):
        if hand_counts[t] > 0:
            m[t] = True
    return m
