"""DAgger 重训: 原始(v31n 分布) + DAgger(网络自己分布) 混合的行为克隆。

为什么: 原始 BC 数据的状态由 v31n 自对弈产生, 而网络部署时走自己的分布。
实测这个漂移是灾难性的 ——

  BC top-1 命中  在 v31n 分布上(训练同源)   91.65%
  BC top-1 命中  在网络自己走出来的分布上    52.65%

39 个百分点。对应到棋力: 分析器本体 血战 -0.061 分/局 vs v31n, 而 92% 模仿它
的网络(同样用 E 碰杠)只有 -0.210。DAgger 就是把网络自己分布上的状态补进
训练集, 反复迭代。

按数据集分别报验证命中率, 这样能直接看到漂移有没有被压下去。

用法:
  python -m tools.dagger_train --data models/hv_value_data_all.npz \
      models/dagger_r1.npz --init models/hv_value_pretrained_v2.pt \
      --out models/bc_dagger1.pt
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
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--weights", nargs="*", type=float,
                    help="各数据集的采样权重(默认全 1)")
    ap.add_argument("--size", default="small")
    ap.add_argument("--init", default="", help="热启动检查点")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--vcoef", type=float, default=0.1,
                    help="E 回归损失权重(主目标是 BC; 0 = 只学策略)")
    ap.add_argument("--soft-tau", type=float, default=0.0,
                    help=">0 时用软标签: 目标 ∝ exp(-(E_t-E_min)/tau)。"
                         "只用 argmax 会丢掉候选之间的差距, 而每条标签都很贵")
    ap.add_argument("--out", default="models/bc_dagger.pt")
    args = ap.parse_args()

    Xs, ys, Bs, Ps, tags, tr, va = [], [], [], [], [], [], []
    off = 0
    for k, p in enumerate(args.data):
        d = np.load(p)
        X = torch.from_numpy(d["feats"])
        y = torch.from_numpy(d["labels"]) / 20.0
        B = torch.from_numpy(d["bests"].astype(np.int64))
        n = len(y)
        if "target" in d.files:
            P = torch.from_numpy(d["target"]).float()   # 搜索教师给的分布
            kind = "教师标签"
        elif args.soft_tau > 0 and "evec" in d.files:
            E = torch.from_numpy(d["evec"])           # (n,28), 非法为 nan
            legal = torch.isfinite(E)
            Em = torch.where(legal, E, torch.full_like(E, float("inf")))
            P = torch.softmax(
                torch.where(legal,
                            -(Em - Em.min(1, keepdim=True).values)
                            / args.soft_tau,
                            torch.full_like(E, -1e9)), dim=-1)
            kind = "软标签"
        else:
            P = torch.zeros(n, 28)
            P[torch.arange(n), B] = 1.0               # 退化成硬标签
            kind = "硬标签"
        g = torch.Generator().manual_seed(1234 + k)
        idx = torch.randperm(n, generator=g) + off
        nv = max(500, n // 20)
        va.append((os.path.basename(p), idx[:nv]))
        tr.append(idx[nv:])
        Xs.append(X)
        ys.append(y)
        Bs.append(B)
        Ps.append(P)
        tags.append(os.path.basename(p))
        off += n
        print(f"{p}: {n} 条 (验证 {nv}) {kind}")
    X = torch.cat(Xs)
    y = torch.cat(ys)
    B = torch.cat(Bs)
    P = torch.cat(Ps)

    w = args.weights or [1.0] * len(tr)
    # 按权重重复训练索引 —— DAgger 数据量少但更贴部署分布, 通常要加权
    tr_idx = torch.cat([t.repeat(int(round(wi))) if wi >= 1 else
                        t[:int(len(t) * wi)] for t, wi in zip(tr, w)])
    print(f"训练样本(加权后) {len(tr_idx)}   权重 {w}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    feat_dim = X.shape[1]          # v2=628 / v4=718, 从数据推断
    if feat_dim != 628:
        print(f"特征维度 {feat_dim} (非默认 628)")
    model = build_model(args.size, feat_dim=feat_dim).to(device)
    if args.init and os.path.exists(args.init):
        ck = torch.load(args.init, map_location="cpu", weights_only=True)
        model.load_state_dict(ck["model"], strict=False)
        print(f"热启动 {args.init}")
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    lossf = nn.SmoothL1Loss(beta=0.1)

    best_own = -1.0
    for ep in range(1, args.epochs + 1):
        model.train()
        perm = tr_idx[torch.randperm(len(tr_idx))]
        tot = 0.0
        for i in range(0, len(perm), args.batch):
            b = perm[i:i + args.batch]
            xb, yb = X[b].to(device), y[b].to(device)
            pb = P[b].to(device)
            q, v = model(xb)
            # 目标分布上为 0 的位置就是非法弃牌 —— 掩掉, 别让容量浪费在上面
            q = q.masked_fill(pb <= 0, -1e9)
            logp = F.log_softmax(q, dim=-1)
            loss = -(pb * logp).sum(-1).mean() + args.vcoef * lossf(v, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(b)
        sched.step()
        model.eval()
        line = f"epoch {ep:2d} loss {tot / len(perm):.4f} "
        accs = {}
        with torch.no_grad():
            for tag, vi in va:
                xv, bv, pv = X[vi].to(device), B[vi].to(device), P[vi].to(device)
                qv, _ = model(xv)
                qv = qv.masked_fill(pv <= 0, -1e9)   # 与部署时一致地掩非法
                a = (qv.argmax(-1) == bv).float().mean().item()
                accs[tag] = a
                line += f"| {tag} 命中 {a:.4f} "
        print(line, flush=True)
        # 用最差的那个数据集当选择依据 —— 短板才是约束
        key = min(accs.values())
        if key > best_own:
            best_own = key
            os.makedirs(os.path.dirname(args.out), exist_ok=True)
            torch.save({"size": args.size, "feat_dim": feat_dim,
                        "model":
                        {k: v.cpu() for k, v in model.state_dict().items()},
                        "acc": accs}, args.out)
    print(f"已保存 {args.out}  (自身分布命中 {best_own:.4f})")


if __name__ == "__main__":
    main()
