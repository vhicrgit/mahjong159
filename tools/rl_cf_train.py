"""同墙反事实 advantage 的 PPO 训练 —— 不需要 critic。

与 tools/rl_bloody_train.py 的区别只有 advantage 的来源:

  那边: adv = R - v(s)。一个座位一局内 ~10 次决策共享同一个回报, 逐决策
        信噪比 0.063; 而且带一个大的混淆项(corr(弃牌序号,奖励) = -0.31),
        要靠分桶去均值才能压掉。
  这边: adv = R_主 - R_替。在采样到的决策点复制局面, 换一张牌用**同一副
        牌墙**重放到终局。两条支线处于同一个状态、同一巡, 所以既不需要
        critic 当基线, 也天然没有"晚巡=坏"的混淆。

实测(tools/perf/diag_cf_snr.py, 用分析器 ΔE 当共同标尺):
  整局 R-baseline  corr(adv,ΔE) +0.047  corr² 0.0022  每样本 10.8 决策
  同墙反事实       corr(adv,ΔE) +0.130  corr² 0.0169  每样本 45.4 决策
即每样本信号 7.7x, 每单位算力 1.9x。

用法:
  python -m tools.rl_cf_train --iters 60 --games 256 --snap-p 0.10
"""

import argparse
import copy
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from backend.rl import cf_collect, eval_crn
from backend.rl.model import build_model
from tools.rl_bloody_train import cpu_copy, make_nn_factory


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=60)
    ap.add_argument("--games", type=int, default=256)
    ap.add_argument("--snap-p", type=float, default=0.10,
                    help="每个决策点留快照的概率(每个快照一次重放)")
    ap.add_argument("--pretrain", default="models/hv_value_pretrained.pt")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--minibatch", type=int, default=512)
    ap.add_argument("--kl-target", type=float, default=0.02)
    ap.add_argument("--kl-ref", type=float, default=0.02)
    ap.add_argument("--ent", type=float, default=0.0)
    ap.add_argument("--gang-w", type=float, default=0.25)
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--eval-seeds", type=int, default=200)
    ap.add_argument("--seed0", type=int, default=70000000)
    ap.add_argument("--out", default="models/cf_latest.pt")
    ap.add_argument("--hist", default="logs/cf_hist.json")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(args.pretrain, map_location="cpu", weights_only=True)
    size = ck.get("size", "small")
    model = build_model(size).to(device)
    model.load_state_dict(ck["model"], strict=False)
    print(f"热启动 {args.pretrain} (size={size}); 反事实 advantage 不用 critic")

    ref_model = None
    if args.kl_ref > 0:
        ref_model = build_model(size).to(device)
        ref_model.load_state_dict(ck["model"], strict=False)
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad_(False)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    hist, best_key, best = [], -1e9, None
    eval_seeds = list(range(args.seed0 + 900000,
                            args.seed0 + 900000 + args.eval_seeds))

    for it in range(1, args.iters + 1):
        t0 = time.time()
        model.eval()
        data = cf_collect.collect_counterfactual(
            model, device, args.games, args.seed0 + it * args.games,
            temp=args.temp, snap_p=args.snap_p, seed=it,
            gang_w=args.gang_w)
        t_col = time.time() - t0
        if data is None or len(data["adv"]) < 64:
            print(f"iter {it}: 快照太少, 跳过")
            continue

        n = len(data["adv"])
        X = torch.from_numpy(data["feats"]).to(device)
        M = torch.from_numpy(data["masks"]).to(device)
        A = torch.from_numpy(data["acts"]).long().to(device)
        LP0 = torch.from_numpy(data["logps"]).to(device)
        ADV = torch.from_numpy(data["adv"]).to(device)
        adv_raw_std = float(ADV.std())
        ADV = (ADV - ADV.mean()) / (ADV.std() + 1e-6)

        LPREF = None
        if ref_model is not None:
            with torch.no_grad():
                qr, _ = ref_model(X)
                LPREF = F.log_softmax(qr.masked_fill(~M, -1e9), dim=-1)

        model.train()
        st = {"pi": 0.0, "ent": 0.0, "kl": 0.0, "clip": 0.0, "klref": 0.0}
        nstep, n_ep, stop = 0, 0, False
        for _ in range(args.epochs):
            if stop:
                break
            n_ep += 1
            perm = torch.randperm(n, device=device)
            for k in range(0, n, args.minibatch):
                idx = perm[k:k + args.minibatch]
                q, _ = model(X[idx])
                logp_all = F.log_softmax(q.masked_fill(~M[idx], -1e9), dim=-1)
                probs = logp_all.exp()
                ent = -(probs * logp_all).sum(-1).mean()
                lp = logp_all.gather(1, A[idx].unsqueeze(1)).squeeze(1)
                ratio = (lp - LP0[idx]).exp()
                a = ADV[idx]
                loss = -torch.min(
                    ratio * a,
                    ratio.clamp(1 - args.clip, 1 + args.clip) * a).mean()
                loss = loss - args.ent * ent
                klref = torch.zeros((), device=device)
                if LPREF is not None:
                    klref = (probs * (logp_all - LPREF[idx])).sum(-1).mean()
                    loss = loss + args.kl_ref * klref
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                st["pi"] += float(loss)
                st["ent"] += float(ent)
                st["kl"] += float((LP0[idx] - lp).mean())
                st["clip"] += float(((ratio - 1).abs() > args.clip).float()
                                    .mean())
                st["klref"] += float(klref)
                nstep += 1
                if st["kl"] / nstep > args.kl_target:
                    stop = True
                    break
        for k in st:
            st[k] /= max(1, nstep)

        msg = (f"iter {it:4d} 反事实样本{n:6d} 步{nstep:3d}/{n_ep}ep "
               f"pi {st['pi']:+.4f} ent {st['ent']:.2f} kl {st['kl']:+.4f} "
               f"clip {st['clip']:.2f} klref {st['klref']:.3f} "
               f"adv标准差 {adv_raw_std:.3f} 采集{t_col:.1f}s")

        if it % args.eval_every == 0 or it == args.iters:
            m_cpu = cpu_copy(model, size)
            t1 = time.time()
            ev = eval_crn.paired_vs_v31(make_nn_factory(m_cpu), eval_seeds)
            key = ev["rank"]["mean"]
            msg += (f"\n   [评估 {args.eval_seeds}seed×4座位 vs v31n] "
                    f"名次差 {key:+.3f}±{ev['rank']['se']:.3f} "
                    f"(t={ev['rank']['t']:+.1f})  得分差 "
                    f"{ev['score']['mean']:+.3f}±{ev['score']['se']:.3f}"
                    f"  用时{time.time() - t1:.0f}s")
            hist.append({"iter": it, "rank_diff": key,
                         "rank_se": ev["rank"]["se"],
                         "score_diff": ev["score"]["mean"]})
            os.makedirs(os.path.dirname(args.out), exist_ok=True)
            torch.save({"size": size, "model": m_cpu.state_dict(),
                        "iter": it, "rank_diff": key}, args.out)
            if key > best_key:
                best_key, best = key, copy.deepcopy(m_cpu.state_dict())
                torch.save({"size": size, "model": best, "iter": it,
                            "rank_diff": key},
                           args.out.replace(".pt", "_best.pt"))
            os.makedirs(os.path.dirname(args.hist), exist_ok=True)
            with open(args.hist, "w") as f:
                json.dump(hist, f, ensure_ascii=False, indent=1)
        print(msg, flush=True)

    print(f"完成。最佳 名次差 {best_key:+.3f} 分/局 vs v31n")


if __name__ == "__main__":
    main()
