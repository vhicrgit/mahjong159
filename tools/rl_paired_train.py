"""配对基线 PPO —— 用「冻结版在同座位同牌墙下的成绩」当 baseline。

与 tools/rl_bloody_train.py 的唯一区别是 advantage 的来源:

  那边: adv = R - v(s)。critic 只解释 27% 的回报方差, 残差标准差 4.59 分。
  这边: adv = R_训练(seed, seat) - R_冻结(seed, seat)。训练 AI 轮坐四个座位,
        对手固定为 3 个冻结版; 基线是冻结版坐同一个座位、面对同样 3 个冻结版、
        用同一副牌墙打出来的成绩。

数学上干净: R_冻结 只依赖 seed 与冻结策略, 不依赖当前策略的动作, 所以它是
合法 baseline, 策略梯度仍无偏。相比 critic 的好处是配对消掉了共享牌运
(实测降方差 1/(1-ρ), 相邻检查点 ρ≈0.9 -> 约 10x); 坏处是这个 advantage
在一个座位一局内恒定, 逐决策分辨率为零(critic 至少随巡数变化)。两者方向相反,
所以必须实测。

每 seed 要 5 局: 训练 AI 分别坐 4 个座位各一局 + 全冻结一局(一局同时给出
四个座位的基线)。样本只取训练座位的决策。

用法:
  python -m tools.rl_paired_train --iters 100 --seeds 64 \
      --init models/bc_k0_r2.pt
"""

import argparse
import copy
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from backend.game.engine import Game
from backend.rl import cf_collect, eval_crn
from backend.rl.features_v2 import encode_state
from backend.rl.model import build_model, legal_discard_mask
from tools.rl_bloody_train import NNSeat, cpu_copy, make_nn_factory


def _claims(g, s):
    """碰/杠用分析器 E 判据(与部署和评估口径一致)。"""
    from backend.analysis import hv_native
    vis = [0] * 28
    for q in g.players:
        for t in q.discards:
            vis[t] += 1
        for m in q.melds:
            vis[m["tile"]] += 3 if m["type"] == "peng" else 4
    for t, n in enumerate(g.players[s].hand_counts):
        vis[t] += n
    hv_native.set_hand(list(g.players[s].hand_counts), vis, 1.0,
                       False, 2, 0, 6)
    return hv_native


def _react(g):
    while g.phase == "react_wait":
        s = list(g.pending_actions.keys())[0]
        hv = _claims(g, s)
        if g.pending_actions[s].get("gang") and \
                hv.decide_gang(g.last_discard, "ming"):
            g.action_gang(s)
        elif g.pending_actions[s].get("peng") and hv.decide_peng(g.last_discard):
            g.action_peng(s)
        else:
            g.action_pass(s)


def run_batch(games, learner, frozen, learn_seat, temp, record,
              frozen_temp=0.0):
    """games[i] 里 learn_seat[i] 用 learner(采样), 其余用 frozen(贪心)。

    learn_seat[i] = -1 表示全部用 frozen(基线局)。record 为 True 时记录
    learner 的决策。返回 records: list[(gi, seat, feat, tile, logp, mask)]。
    """
    recs = []
    it = 0
    while it < 900:
        it += 1
        for g in games:
            if g.phase == "react_wait":
                _react(g)
        idx = [i for i, g in enumerate(games) if g.phase == "discard_wait"]
        if not idx:
            break
        # 先一次性分区再落子: 落子会改变 phase/turn, 分区判定必须用落子前的状态
        turns = {i: games[i].turn for i in idx}
        group = {"L": [i for i in idx if turns[i] == learn_seat[i]],
                 "F": [i for i in idx if turns[i] != learn_seat[i]]}
        for who in ("L", "F"):
            sel = group[who]
            if not sel:
                continue
            m = learner if who == "L" else frozen
            gs = [games[i] for i in sel]
            ss = [turns[i] for i in sel]
            feats = np.stack([encode_state(g, s) for g, s in zip(gs, ss)])
            masks = torch.stack([legal_discard_mask(g.players[s].hand_counts)
                                 for g, s in zip(gs, ss)])
            with torch.no_grad():
                q, _ = m(torch.from_numpy(feats))
                logits = q.masked_fill(~masks, -1e9)
                if who == "L":
                    lp = F.log_softmax(logits / max(temp, 1e-6), dim=-1)
                    a = torch.multinomial(lp.exp(), 1).squeeze(-1)
                    lpa = lp.gather(1, a.unsqueeze(1)).squeeze(1)
                elif frozen_temp > 0:
                    # 冻结版同温采样: 两条臂都随机, 配对更紧(否则比较被
                    # "学习者要探索"这一项主导)
                    p = F.softmax(logits / frozen_temp, dim=-1)
                    a = torch.multinomial(p, 1).squeeze(-1)
                    lpa = torch.zeros(len(sel))
                else:
                    a = logits.argmax(-1)          # 冻结版贪心
                    lpa = torch.zeros(len(sel))
            for k, i in enumerate(sel):
                if who == "L" and record:
                    recs.append((i, ss[k], feats[k], int(a[k]),
                                 float(lpa[k]), masks[k].numpy()))
                games[i].action_discard(ss[k], int(a[k]))
    for g in games:
        if g.phase == "react_wait":
            _react(g)
    return recs


def collect(learner, frozen, seeds, temp, gang_w, frozen_temp=0.0):
    """每个 seed: 4 局(训练 AI 各坐一个座位) + 1 局全冻结。"""
    games, lseat, tag = [], [], []
    for si, sd in enumerate(seeds):
        for s in range(4):
            games.append(Game(seed=sd, human_seat=-1, bloody=True))
            lseat.append(s)
            tag.append((si, s))
        games.append(Game(seed=sd, human_seat=-1, bloody=True))
        lseat.append(-1)
        tag.append((si, -1))
    recs = run_batch(games, learner, frozen, lseat, temp, record=True,
                     frozen_temp=frozen_temp)
    R = {}
    for gi, g in enumerate(games):
        si, s = tag[gi]
        if s == -1:
            for t in range(4):
                R[("F", si, t)] = cf_collect.default_reward(g, t, gang_w)
        else:
            R[("L", si, s)] = cf_collect.default_reward(g, s, gang_w)
    out = []
    for (gi, seat, feat, tile, logp, mask) in recs:
        si, s = tag[gi]
        out.append((feat, tile, logp, mask,
                    R[("L", si, s)] - R[("F", si, s)]))
    # 汇总: 四座位求和的比分差(用户想看的那个量)
    tot = [sum(R[("L", si, s)] - R[("F", si, s)] for s in range(4))
           for si in range(len(seeds))]
    return out, float(np.mean(tot)), float(np.std(tot) / np.sqrt(len(tot)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--seeds", type=int, default=64, help="每轮 seed 数(×5 局)")
    ap.add_argument("--init", default="models/bc_k0_r2.pt")
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--frozen-temp", type=float, default=0.0,
                    help=">0 时冻结版也按该温度采样, 配对更紧")
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--minibatch", type=int, default=512)
    ap.add_argument("--kl-target", type=float, default=0.02)
    ap.add_argument("--kl-ref", type=float, default=0.05)
    ap.add_argument("--gang-w", type=float, default=0.25)
    ap.add_argument("--eval-every", type=int, default=15)
    ap.add_argument("--eval-seeds", type=int, default=250)
    ap.add_argument("--seed0", type=int, default=110000000)
    ap.add_argument("--out", default="models/pair_latest.pt")
    ap.add_argument("--hist", default="logs/pair_hist.json")
    args = ap.parse_args()

    ck = torch.load(args.init, map_location="cpu", weights_only=True)
    size = ck.get("size", "small")
    learner = build_model(size)
    learner.load_state_dict(ck["model"], strict=False)
    frozen = build_model(size)
    frozen.load_state_dict(ck["model"], strict=False)
    frozen.eval()
    for p in frozen.parameters():
        p.requires_grad_(False)
    ref = build_model(size)
    ref.load_state_dict(ck["model"], strict=False)
    ref.eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    print(f"起点/冻结版 = {args.init} (size={size})")

    opt = torch.optim.Adam(learner.parameters(), lr=args.lr)
    hist, best_key, best = [], -1e9, None
    eval_seeds = list(range(args.seed0 + 900000,
                            args.seed0 + 900000 + args.eval_seeds))

    for it in range(1, args.iters + 1):
        t0 = time.time()
        seeds = list(range(args.seed0 + it * args.seeds,
                           args.seed0 + (it + 1) * args.seeds))
        learner.eval()
        data, sum_diff, sum_se = collect(learner, frozen, seeds, args.temp,
                                         args.gang_w, args.frozen_temp)
        t_col = time.time() - t0
        if len(data) < 64:
            print(f"iter {it}: 样本太少, 跳过")
            continue

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
        st = {"pi": 0.0, "kl": 0.0, "clip": 0.0, "klref": 0.0, "ent": 0.0}
        n = len(data)
        nstep, n_ep, stop = 0, 0, False
        for _ in range(args.epochs):
            if stop:
                break
            n_ep += 1
            perm = torch.randperm(n)
            for k in range(0, n, args.minibatch):
                idx = perm[k:k + args.minibatch]
                q, _ = learner(X[idx])
                lpa = F.log_softmax(q.masked_fill(~M[idx], -1e9), dim=-1)
                probs = lpa.exp()
                ent = -(probs * lpa).sum(-1).mean()
                lp = lpa.gather(1, A[idx].unsqueeze(1)).squeeze(1)
                ratio = (lp - LP0[idx]).exp()
                a = ADVn[idx]
                loss = -torch.min(
                    ratio * a,
                    ratio.clamp(1 - args.clip, 1 + args.clip) * a).mean()
                klref = (probs * (lpa - LPREF[idx])).sum(-1).mean()
                loss = loss + args.kl_ref * klref
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(learner.parameters(), 1.0)
                opt.step()
                st["pi"] += loss.item()
                st["kl"] += (LP0[idx] - lp).mean().item()
                st["clip"] += ((ratio - 1).abs() > args.clip).float() \
                    .mean().item()
                st["klref"] += klref.item()
                st["ent"] += ent.item()
                nstep += 1
                if st["kl"] / nstep > args.kl_target:
                    stop = True
                    break
        for k in st:
            st[k] /= max(1, nstep)

        msg = (f"iter {it:4d} 样本{n:5d} 步{nstep:3d}/{n_ep}ep "
               f"pi {st['pi']:+.4f} ent {st['ent']:.2f} kl {st['kl']:+.4f} "
               f"clip {st['clip']:.2f} klref {st['klref']:.3f} "
               f"adv标准差 {adv_std:.3f} 四座位比分差 {sum_diff:+.3f}"
               f"±{sum_se:.3f} 采集{t_col:.1f}s")

        if it % args.eval_every == 0 or it == args.iters:
            m_cpu = cpu_copy(learner, size)
            ev = eval_crn.paired_vs_v31(make_nn_factory(m_cpu), eval_seeds)
            key = ev["rank"]["mean"]
            msg += (f"\n   [评估 vs v31n] 名次差 {key:+.3f}±"
                    f"{ev['rank']['se']:.3f} (t={ev['rank']['t']:+.1f})")
            hist.append({"iter": it, "rank_diff": key,
                         "sum_diff": sum_diff, "adv_std": adv_std})
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
