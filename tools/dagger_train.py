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
import hashlib
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from backend.rl.model import build_model


def legal_from_features(x):
    """v2/v4 prefix: 28 tiles x count one-hot; channel 0 means absent."""
    if x.ndim != 2 or x.shape[1] not in (628, 718):
        raise ValueError("Legacy mask recovery supports only v2/v4 features")
    counts = x[:, :112].reshape(-1, 28, 4)
    if not torch.all((counts == 0) | (counts == 1)) or not torch.all(counts.sum(-1) == 1):
        raise ValueError("Invalid count one-hot in features")
    return counts[:, :, 0] == 0


def policy_loss(q, target, legal):
    """Teacher support and game legality are independent concepts."""
    if not legal.any(-1).all() or (target.masked_select(~legal) > 0).any():
        raise ValueError("Empty legal set or target on an illegal action")
    return -(target * F.log_softmax(q.masked_fill(~legal, -1e9), -1)).sum(-1)


def split_indices(n, seed, groups=None):
    if n < 2:
        raise ValueError("Dataset needs at least two independent units")
    rng = np.random.default_rng(seed)
    if groups is not None:
        unique = np.unique(groups)
        if len(unique) < 2:
            raise ValueError("Need at least two games for a held-out split")
        nv = min(len(unique) - 1, max(1, int(np.ceil(len(unique) * .05))))
        held = rng.permutation(unique)[:nv]
        is_val = np.isin(groups, held)
        return torch.tensor(np.flatnonzero(~is_val)), torch.tensor(np.flatnonzero(is_val))
    order = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    nv = min(n - 1, max(1, min(500, n // 5), n // 20))
    return order[nv:], order[:nv]


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
    ap.add_argument("--seed", type=int, default=20260905)
    ap.add_argument("--split-seed", type=int, default=1234)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--teacher-policy-weight", type=float, default=1.0,
                    help="0 gives the matched anchor-only control; teacher rows stay in batches")
    ap.add_argument("--randomize-teacher", action="store_true",
                    help="Negative control: permute training teacher probabilities within legal actions")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    Xs, ys, Bs, Ps, Ms, Vs, Ts, tr, va = [], [], [], [], [], [], [], [], []
    off = 0
    for k, p in enumerate(args.data):
        d = np.load(p)
        X = torch.from_numpy(d["feats"])
        y = torch.from_numpy(d["labels"]) / 20.0
        B = torch.from_numpy(d["bests"].astype(np.int64))
        n = len(y)
        legal = legal_from_features(X)
        if "legal_mask" in d.files:
            stored = torch.from_numpy(d["legal_mask"].astype(bool))
            if not torch.equal(stored, legal):
                raise ValueError(f"{p}: stored mask differs from hand features")
        teacher = "target" in d.files
        if "target" in d.files:
            P = torch.from_numpy(d["target"]).float()   # 搜索教师给的分布
            kind = "教师标签"
        elif args.soft_tau > 0 and "evec" in d.files:
            E = torch.from_numpy(d["evec"])           # (n,28), 非法为 nan
            if not torch.equal(torch.isfinite(E), legal):
                raise ValueError(f"{p}: E labels do not cover exactly the legal actions")
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
        if not torch.isfinite(P).all() or (P < 0).any() or not torch.allclose(P.sum(1), torch.ones(n)):
            raise ValueError(f"{p}: invalid target distribution")
        if (P.masked_select(~legal) > 0).any() or not legal.gather(1, B[:, None]).all():
            raise ValueError(f"{p}: illegal teacher action")
        # Stable per-file split, independent of dataset ordering or training seed.
        split_seed = (args.split_seed + int(hashlib.sha256(os.path.basename(p).encode()).hexdigest()[:8], 16)) % 2**32
        groups = d["game_id"] if "game_id" in d.files else None
        ti, vi = split_indices(n, split_seed, groups)
        va.append((os.path.basename(p), vi + off))
        tr.append(ti + off)
        Xs.append(X)
        ys.append(y)
        Bs.append(B)
        Ps.append(P)
        Ms.append(legal)
        valid_value = (torch.from_numpy(d["value_valid"].astype(bool)) if "value_valid" in d.files
                       else torch.full((n,), not teacher, dtype=torch.bool))
        if not torch.isfinite(y[valid_value]).all():
            raise ValueError(f"{p}: nonfinite value labels")
        ys[-1] = torch.where(valid_value, y, torch.zeros_like(y))
        Vs.append(valid_value)
        Ts.append(torch.full((n,), teacher, dtype=torch.bool))
        off += n
        print(f"{p}: {n} 条 (验证 {len(vi)}) {kind}; value有效={int(valid_value.sum())}; "
              f"split={'game' if groups is not None else 'ROW-ONLY legacy; strength requires fresh games'}")
    X = torch.cat(Xs)
    y = torch.cat(ys)
    B = torch.cat(Bs)
    P = torch.cat(Ps)
    M, V, T = torch.cat(Ms), torch.cat(Vs), torch.cat(Ts)

    w = args.weights or [1.0] * len(tr)
    if len(w) != len(tr) or any(wi < 0 for wi in w):
        raise ValueError("Need one nonnegative weight per dataset")
    # 按权重重复训练索引 —— DAgger 数据量少但更贴部署分布, 通常要加权
    tr_idx = torch.cat([t.repeat(int(round(wi))) if wi >= 1 else
                        t[:int(len(t) * wi)] for t, wi in zip(tr, w)])
    print(f"训练样本(加权后) {len(tr_idx)}   权重 {w}")
    if len(tr_idx) == 0 or args.epochs < 1 or args.teacher_policy_weight < 0:
        raise ValueError("Empty training set or invalid training parameters")
    train_P = P.clone()
    if args.randomize_teacher:
        rng = torch.Generator().manual_seed(args.seed + 910)
        for i in torch.unique(tr_idx[T[tr_idx]]).tolist():
            legal_idx = torch.where(M[i])[0]
            perm = torch.randperm(len(legal_idx), generator=rng)
            train_P[i, legal_idx] = P[i, legal_idx[perm]]

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    feat_dim = X.shape[1]          # v2=628 / v4=718, 从数据推断
    if feat_dim != 628:
        print(f"特征维度 {feat_dim} (非默认 628)")
    model = build_model(args.size, feat_dim=feat_dim).to(device)
    if args.init:
        if not os.path.exists(args.init):
            raise FileNotFoundError(args.init)
        ck = torch.load(args.init, map_location="cpu", weights_only=True)
        model.load_state_dict(ck["model"], strict=True)
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
            pb, mb, vb = train_P[b].to(device), M[b].to(device), V[b].to(device)
            q, v = model(xb)
            pw = torch.where(T[b], args.teacher_policy_weight, 1.0).to(device)
            loss = (policy_loss(q, pb, mb) * pw).mean()
            if args.vcoef and vb.any():
                loss = loss + args.vcoef * lossf(v[vb], yb[vb])
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
                xv, bv = X[vi].to(device), B[vi].to(device)
                qv, _ = model(xv)
                qv = qv.masked_fill(~M[vi].to(device), -1e9)
                a = (qv.argmax(-1) == bv).float().mean().item()
                accs[tag] = a
                line += f"| {tag} 命中 {a:.4f} "
        print(line, flush=True)
        # 用最差的那个数据集当选择依据 —— 短板才是约束
        key = min(accs.values())
        if key > best_own:
            best_own = key
            os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
            torch.save({"size": args.size, "feat_dim": feat_dim,
                        "model":
                        {k: v.cpu() for k, v in model.state_dict().items()},
                        "acc": accs, "epoch": ep, "train_args": vars(args),
                        "training_protocol": "legal_mask_value_valid_v2"}, args.out)
    # Fixed-budget comparisons use last, avoiding label-accuracy checkpoint selection.
    last = os.path.splitext(args.out)[0] + "_last.pt"
    torch.save({"size": args.size, "feat_dim": feat_dim,
                "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "acc": accs, "epoch": args.epochs, "train_args": vars(args),
                "training_protocol": "legal_mask_value_valid_v2"}, last)
    print(f"固定训练预算终点: {last}")
    print(f"已保存 {args.out}  (自身分布命中 {best_own:.4f})")


if __name__ == "__main__":
    main()
