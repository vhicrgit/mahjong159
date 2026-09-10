"""价值网络预训练: 用牌型分析器的 E 值(期望巡数)训练 MahjongNet 的 value 头。

数据来自 tools/rl_value_data.py(v31n 自对弈决策点 + C 引擎 E 标签)。
这一步给躯干(feature 表示)和 value 头一个强初始化, 供后续 actor-critic
RL 热启动; q 头此阶段不训练。

用法: python -m tools.rl_value_pretrain --data models/hv_value_data.npz
"""

import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from backend.rl.model import build_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="models/hv_value_data.npz")
    ap.add_argument("--size", type=str, default="small")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--bc", type=float, default=1.0,
                    help="行为克隆(分析器最优弃牌)损失权重, 0=只学价值")
    ap.add_argument("--out", type=str, default="models/hv_value_pretrained.pt")
    args = ap.parse_args()

    d = np.load(args.data)
    X = torch.from_numpy(d["feats"])          # (N, 628)
    y = torch.from_numpy(d["labels"]) / 20.0  # 巡数归一(~5-25 -> 0.25-1.25)
    B = torch.from_numpy(d["bests"].astype(np.int64))  # 分析器最优弃牌(BC)
    n = len(y)
    idx = torch.randperm(n)
    n_val = max(1000, n // 20)
    vi, ti = idx[:n_val], idx[n_val:]
    Xv, yv, Bv = X[vi], y[vi], B[vi]
    Xt, yt, Bt = X[ti], y[ti], B[ti]
    print(f"训练 {len(ti)} / 验证 {len(vi)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(args.size).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    lossf = nn.SmoothL1Loss(beta=0.1)
    Xv_d, yv_d, Bv_d = Xv.to(device), yv.to(device), Bv.to(device)

    for ep in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(len(yt))
        tot = 0.0
        for i in range(0, len(perm), args.batch):
            b = perm[i: i + args.batch]
            xb, yb, bb = Xt[b].to(device), yt[b].to(device), Bt[b].to(device)
            q, v = model(xb)
            # 价值回归 + 策略行为克隆(q 头对分析器最优弃牌做交叉熵)
            loss = lossf(v, yb) + args.bc * F.cross_entropy(q, bb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(b)
        model.eval()
        with torch.no_grad():
            qv, vv = model(Xv_d)
            mae = (vv - yv_d).abs().mean().item() * 20.0
            acc = (qv.argmax(-1) == Bv_d).float().mean().item()
        print(f"epoch {ep:2d}  train_huber={tot / len(yt):.4f}  "
                  f"val MAE={mae:.3f} 巡  策略命中率={acc:.1%}", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save({"size": args.size, "model": model.state_dict(),
                "val_mae_turns": mae}, args.out)
    print(f"已保存 {args.out}  (val MAE {mae:.3f} 巡)")


if __name__ == "__main__":
    main()
