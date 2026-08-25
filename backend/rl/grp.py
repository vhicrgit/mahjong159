"""安康159 - Global Reward Prediction (GRP)

Suphx/Mortal 的核心组件:
- 训练一个模型, 从中间状态预测最终得分
- 用 GRP 预测的变化作为每步的奖励信号 (替代稀疏的局终分数)
- 大幅降低奖励方差, 让 RL 能学到有效的信用分配

GRP 模型: ResNet 结构 (与策略网络同族), 输入局面特征, 输出4个玩家的预期最终得分
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .features_v2 import FEAT_DIM


class ResBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = self.fc2(h)
        return F.relu(x + h)


class GRPModel(nn.Module):
    """预测每个玩家的最终得分"""

    def __init__(self, hidden_dim: int = 512, num_blocks: int = 6):
        super().__init__()
        self.input_proj = nn.Linear(FEAT_DIM, hidden_dim)
        self.blocks = nn.ModuleList(ResBlock(hidden_dim)
                                    for _ in range(num_blocks))
        self.head = nn.Linear(hidden_dim, 4)  # 4个玩家的预期得分

    def forward(self, x):
        h = F.relu(self.input_proj(x))
        for b in self.blocks:
            h = b(h)
        return self.head(h)  # (B, 4)


def train_grp(model, data, device, epochs=20, batch_size=2048, lr=1e-3):
    """训练 GRP 模型

    data: {"feats": (N, D), "scores": (N, 4)}
      feats: 每个决策点的局面特征
      scores: 该局4个玩家的最终得分
    """
    feats = torch.from_numpy(data["feats"]).to(device)
    scores = torch.from_numpy(data["scores"]).to(device)
    n = len(feats)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        total_loss = 0.0
        nb = 0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            pred = model(feats[idx])
            loss = F.mse_loss(pred, scores[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
            nb += 1
        scheduler.step()
        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"  GRP ep {ep+1}/{epochs}  mse={total_loss/nb:.4f}")
    return model
