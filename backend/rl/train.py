"""安康159 - 训练脚本

阶段1 BC热身: 行为克隆规则Bot的出牌(交叉熵), 让模型先学会牌效
阶段2 AWR强化: 用局终收益做优势加权回归(简单稳定的offline RL),
  收益为正的决策加大权重, 为负的减小权重

用法:
  python -m backend.rl.train --games 300 --size tiny --out model_tiny.pt
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import build_model, N_ACTIONS
from .selfplay import generate_dataset


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train_bc(model, data, device, epochs=10, batch_size=512, lr=1e-3):
    """行为克隆"""
    feats = torch.from_numpy(data["feats"]).to(device)
    acts = torch.from_numpy(data["acts"]).to(device)
    n = len(acts)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for ep in range(epochs):
        perm = torch.randperm(n, device=device)
        total_loss, total_correct = 0.0, 0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            logits, _ = model(feats[idx])
            loss = F.cross_entropy(logits, acts[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(idx)
            total_correct += (logits.argmax(-1) == acts[idx]).sum().item()
        print(f"  BC epoch {ep + 1}/{epochs}  loss={total_loss / n:.4f}  "
              f"acc={total_correct / n:.3f}")


def train_awr(model, data, device, epochs=5, batch_size=512, lr=5e-4,
              beta=1.0, v_epochs=3):
    """AWR: 先训 value head 估计 V(s), 再用 exp(A/beta) 加权 policy 损失"""
    feats = torch.from_numpy(data["feats"]).to(device)
    acts = torch.from_numpy(data["acts"]).to(device)
    rets = torch.from_numpy(data["rets"]).to(device)
    n = len(acts)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    # value 拟合
    for ep in range(v_epochs):
        perm = torch.randperm(n, device=device)
        total_vloss = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            _, v = model(feats[idx])
            vloss = F.mse_loss(v, rets[idx])
            opt.zero_grad()
            vloss.backward()
            opt.step()
            total_vloss += vloss.item() * len(idx)
        print(f"  AWR value epoch {ep + 1}/{v_epochs}  mse={total_vloss / n:.4f}")

    # policy 加权回归
    with torch.no_grad():
        _, v_all = model(feats)
        adv = (rets - v_all)
        weights = torch.exp(adv / beta).clamp(max=20.0)
    for ep in range(epochs):
        perm = torch.randperm(n, device=device)
        total_loss = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            logits, _ = model(feats[idx])
            logp = F.log_softmax(logits, dim=-1)
            act_logp = logp.gather(1, acts[idx].unsqueeze(1)).squeeze(1)
            loss = -(weights[idx] * act_logp).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(idx)
        print(f"  AWR policy epoch {ep + 1}/{epochs}  loss={total_loss / n:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=300)
    ap.add_argument("--size", type=str, default="tiny")
    ap.add_argument("--bc-epochs", type=int, default=10)
    ap.add_argument("--awr-epochs", type=int, default=5)
    ap.add_argument("--out", type=str, default="model_tiny.pt")
    ap.add_argument("--data", type=str, default="")
    args = ap.parse_args()

    device = get_device()
    print(f"设备: {device}, 模型: {args.size}")

    if args.data:
        z = np.load(args.data)
        data = {"feats": z["feats"], "acts": z["acts"], "rets": z["rets"]}
    else:
        print(f"生成 {args.games} 局自对弈数据...")
        data = generate_dataset(args.games)
    print(f"样本数: {len(data['acts'])}")

    model = build_model(args.size).to(device)
    print(f"参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    print("阶段1: BC热身")
    train_bc(model, data, device, epochs=args.bc_epochs)
    print("阶段2: AWR强化")
    train_awr(model, data, device, epochs=args.awr_epochs)

    torch.save({"model": model.state_dict(), "size": args.size}, args.out)
    print(f"已保存: {args.out}")


if __name__ == "__main__":
    main()
