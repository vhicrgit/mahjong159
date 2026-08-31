"""冠军循环: PPO 出候选, 配对比较当准入闸门。

昨晚测到的核心问题是 PPO 在弱信号上做随机游走 —— 从任何起点都单调退化。
但配对比较能低方差地判定"候选是否强于冠军"(500 seed 下 SE 0.028)。把两者
组合起来: 梯度只负责**提候选**, 是否接受由配对比较说话。退化的更新会被直接
丢弃, 所以这个循环在统计上不可能比起点更差(除了 I 类错误的概率)。

这是把"比分当 loss"落地的非梯度形式 —— 得分不可微, 但可以当选择信号。

每轮:
  1. 从冠军出发跑 M 次配对基线 PPO 得到候选
  2. 候选 vs 冠军做 CRN 配对头对头(轮坐四座位, 对手为 v31n)
  3. t > 阈值 才把冠军换成候选, 否则丢弃并换随机种子重试

用法:
  python -m tools.champion_loop --rounds 12 --inner 12 --seeds 96 \
      --gate-seeds 300 --init models/bc_k0_r2.pt
"""

import argparse
import copy
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from backend.rl import eval_crn
from backend.rl.model import build_model
from tools.rl_bloody_train import NNSeat, cpu_copy
from tools.rl_paired_train import collect


def ppo_steps(learner, frozen, ref, opt, seeds, args):
    """一次配对基线 PPO 更新, 返回 (样本数, adv标准差, 四座位比分差)。"""
    learner.eval()
    data, sum_diff, _ = collect(learner, frozen, seeds, args.temp,
                               args.gang_w, args.frozen_temp)
    if len(data) < 64:
        return 0, 0.0, sum_diff
    X = torch.from_numpy(np.stack([d[0] for d in data]))
    A = torch.tensor([d[1] for d in data], dtype=torch.long)
    LP0 = torch.tensor([d[2] for d in data], dtype=torch.float32)
    M = torch.from_numpy(np.stack([d[3] for d in data]))
    ADV = torch.tensor([d[4] for d in data], dtype=torch.float32)
    adv_std = float(ADV.std())
    ADVn = (ADV - ADV.mean()) / (ADV.std() + 1e-6)
    with torch.no_grad():
        qr, _ = ref(X)
        LPREF = F.log_softmax(qr.masked_fill(~M, -1e9), dim=-1)
    learner.train()
    n = len(data)
    kl_run, nstep = 0.0, 0
    for _ in range(args.epochs):
        perm = torch.randperm(n)
        brk = False
        for k in range(0, n, args.minibatch):
            idx = perm[k:k + args.minibatch]
            q, _ = learner(X[idx])
            lpa = F.log_softmax(q.masked_fill(~M[idx], -1e9), dim=-1)
            probs = lpa.exp()
            lp = lpa.gather(1, A[idx].unsqueeze(1)).squeeze(1)
            ratio = (lp - LP0[idx]).exp()
            a = ADVn[idx]
            loss = -torch.min(
                ratio * a,
                ratio.clamp(1 - args.clip, 1 + args.clip) * a).mean()
            loss = loss + args.kl_ref * (
                probs * (lpa - LPREF[idx])).sum(-1).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(learner.parameters(), 1.0)
            opt.step()
            kl_run += (LP0[idx] - lp).mean().item()
            nstep += 1
            if kl_run / nstep > args.kl_target:
                brk = True
                break
        if brk:
            break
    return n, adv_std, sum_diff


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=12)
    ap.add_argument("--inner", type=int, default=12, help="每轮 PPO 次数")
    ap.add_argument("--seeds", type=int, default=96, help="每次 PPO 的 seed 数")
    ap.add_argument("--gate-seeds", type=int, default=300, help="闸门 seed 数")
    ap.add_argument("--gate-t", type=float, default=1.5, help="准入 t 阈值")
    ap.add_argument("--init", default="models/bc_k0_r2.pt")
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--frozen-temp", type=float, default=1.0)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--minibatch", type=int, default=512)
    ap.add_argument("--kl-target", type=float, default=0.02)
    ap.add_argument("--kl-ref", type=float, default=0.05)
    ap.add_argument("--gang-w", type=float, default=0.25)
    ap.add_argument("--seed0", type=int, default=130000000)
    ap.add_argument("--out", default="models/champ.pt")
    ap.add_argument("--hist", default="logs/champ_hist.json")
    args = ap.parse_args()

    ck = torch.load(args.init, map_location="cpu", weights_only=True)
    size = ck.get("size", "small")
    champ_sd = {k: v.clone() for k, v in ck["model"].items()}
    ref = build_model(size)
    ref.load_state_dict(ck["model"], strict=False)
    ref.eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    print(f"冠军起点 {args.init} (size={size})  闸门 {args.gate_seeds} seed, "
          f"t>{args.gate_t}")

    hist, accepted = [], 0
    for rd in range(1, args.rounds + 1):
        t0 = time.time()
        learner = build_model(size)
        learner.load_state_dict(champ_sd, strict=False)
        frozen = build_model(size)
        frozen.load_state_dict(champ_sd, strict=False)
        frozen.eval()
        for p in frozen.parameters():
            p.requires_grad_(False)
        opt = torch.optim.Adam(learner.parameters(), lr=args.lr)

        ns, advs, sds = 0, [], []
        for j in range(args.inner):
            seeds = list(range(args.seed0 + rd * 100000 + j * args.seeds,
                               args.seed0 + rd * 100000 + (j + 1) * args.seeds))
            n, a, sd = ppo_steps(learner, frozen, ref, opt, seeds, args)
            ns += n
            advs.append(a)
            sds.append(sd)
        t_tr = time.time() - t0

        cand = cpu_copy(learner, size)
        chp = build_model(size)
        chp.load_state_dict(champ_sd, strict=False)
        chp.eval()
        gate_seeds = list(range(args.seed0 + 9000000 + rd * 10000,
                                args.seed0 + 9000000 + rd * 10000
                                + args.gate_seeds))
        t1 = time.time()
        ev = eval_crn.paired_head2head(
            lambda g, s: NNSeat(g, s, cand),
            lambda g, s: NNSeat(g, s, chp), gate_seeds, bloody=True)
        d, se, t = ev["rank"]["mean"], ev["rank"]["se"], ev["rank"]["t"]
        ok = t > args.gate_t
        if ok:
            champ_sd = {k: v.clone() for k, v in cand.state_dict().items()}
            accepted += 1
            os.makedirs(os.path.dirname(args.out), exist_ok=True)
            torch.save({"size": size, "model": champ_sd, "round": rd,
                        "gate": d}, args.out)
        print(f"round {rd:3d} 样本{ns:6d} adv标准差 {np.mean(advs):.3f} "
              f"训练比分差 {np.mean(sds):+.3f} | 闸门 候选-冠军 "
              f"{d:+.4f}±{se:.4f} t={t:+.2f} -> "
              f"{'接受' if ok else '丢弃'}  训练{t_tr:.0f}s 闸门{time.time() - t1:.0f}s",
              flush=True)
        hist.append({"round": rd, "gate_diff": d, "gate_se": se, "t": t,
                     "accepted": bool(ok), "train_diff": float(np.mean(sds))})
        os.makedirs(os.path.dirname(args.hist), exist_ok=True)
        with open(args.hist, "w") as f:
            json.dump(hist, f, ensure_ascii=False, indent=1)

    print(f"完成。{args.rounds} 轮中接受 {accepted} 次")
    if accepted:
        print(f"冠军在 {args.out}")


if __name__ == "__main__":
    main()
