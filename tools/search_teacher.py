"""搜索教师(expert iteration): 用同墙反事实 rollout 给出比分析器更强的标签。

为什么不是 RL: 同一份反事实信号, 当策略梯度用时每样本 corr²(adv,ΔE) 只有
0.017, 实跑必然退化(从任何起点都掉 0.2~0.3 分/局, 已试过整局/反事实/温和
步长/温度匹配四种配置)。但同一份信号**当监督标签**用时, 可以对同一个
(状态, 动作) 重复 N 次 rollout 取平均, 噪声按 1/√N 收敛, 再喂给交叉熵 ——
监督梯度的方差与标签噪声不是一回事。

与分析器教师的区别: 分析器最小化期望巡数 E, 看不见防守和对手; rollout 直接
用终局名次奖励打分, 天然包含"这张牌会不会喂给别人"。所以它的上限高于分析器,
这是唯一一条能真正超过 -0.06 分/局 那条线的路。

做法:
  对采样到的状态, 取策略前 K 个候选(全枚举太贵), 每个候选用**同一副牌墙**
  重放 N 次(四家都用当前策略, temp>0 制造多样性), 取平均名次奖励;
  标签 = softmax(score/τ) 只在这 K 个候选上。

用法:
  python -m tools.search_teacher --model models/bc_k0_r3.pt --states 4000 \
      --topk 3 --rolls 8 --out models/teach_r1.npz
"""

import argparse
import copy
import time

import numpy as np
import torch
import torch.nn.functional as F

from backend.game.engine import Game
from backend.rl import cf_collect
from backend.rl.features_v2 import encode_state
from backend.rl.model import build_model, legal_discard_mask


def collect_states(model, n_games, seed0, snap_p, seed, bloody=True, temp=1.0):
    rng = np.random.default_rng(seed)
    games = [Game(seed=seed0 + i, human_seat=-1, bloody=bloody)
             for i in range(n_games)]
    _, snaps = cf_collect.run_games(model, "cpu", games, temp, snap_p, rng)
    return snaps


def topk_actions(model, snaps, k):
    """每个状态取策略概率最高的 k 个合法弃牌。"""
    feats = np.stack([sn["feat"] for sn in snaps])
    masks = torch.from_numpy(np.stack([sn["mask"] for sn in snaps]))
    with torch.no_grad():
        q, _ = model(torch.from_numpy(feats))
        q = q.masked_fill(~masks, -1e9)
    nleg = masks.sum(1)
    order = q.argsort(dim=-1, descending=True)
    out = []
    for i in range(len(snaps)):
        kk = int(min(k, nleg[i].item()))
        out.append([int(t) for t in order[i, :kk]])
    return out, feats, masks.numpy()


def score_candidates(model, snaps, cands, rolls, temp, chunk_states=150):
    """同墙重放: 返回 (n_states, max_k) 的平均奖励矩阵(无效位 nan)。"""
    maxk = max(len(c) for c in cands)
    S = np.full((len(snaps), maxk), np.nan, dtype=np.float32)
    for lo in range(0, len(snaps), chunk_states):
        hi = min(lo + chunk_states, len(snaps))
        jobs, keys = [], []
        for i in range(lo, hi):
            for ci, a in enumerate(cands[i]):
                for _ in range(rolls):
                    g = copy.deepcopy(snaps[i]["game"])
                    g.action_discard(snaps[i]["seat"], a)
                    jobs.append(g)
                    keys.append((i, ci))
        if not jobs:
            continue
        jobs, _ = cf_collect.run_games(model, "cpu", jobs, temp)
        acc = {}
        for g, (i, ci) in zip(jobs, keys):
            r = cf_collect.default_reward(g, snaps[i]["seat"])
            acc.setdefault((i, ci), []).append(r)
        for (i, ci), vs in acc.items():
            S[i, ci] = float(np.mean(vs))
        print(f"  ...状态 {hi}/{len(snaps)}  重放 {len(jobs)} 局", flush=True)
    return S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/bc_k0_r3.pt")
    ap.add_argument("--states", type=int, default=4000)
    ap.add_argument("--games", type=int, default=0,
                    help="0 = 按 states 自动推算局数")
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--rolls", type=int, default=8)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--tau", type=float, default=0.5, help="标签 softmax 温度")
    ap.add_argument("--seed0", type=int, default=90000000)
    ap.add_argument("--out", default="models/teach_r1.npz")
    args = ap.parse_args()

    ck = torch.load(args.model, map_location="cpu", weights_only=True)
    model = build_model(ck["size"])
    model.load_state_dict(ck["model"], strict=False)
    model.eval()

    # 血战一局约 45 次弃牌 × snap_p 0.06 ≈ 2.7 个状态/局
    n_games = args.games or max(50, int(args.states / 2.7))
    t0 = time.time()
    snaps = collect_states(model, n_games, args.seed0, 0.06, 7)
    snaps = snaps[:args.states]
    print(f"采到 {len(snaps)} 个状态 ({n_games} 局, {time.time() - t0:.0f}s)",
          flush=True)

    cands, feats, masks = topk_actions(model, snaps, args.topk)
    t0 = time.time()
    S = score_candidates(model, snaps, cands, args.rolls, args.temp)
    print(f"打分完成 {time.time() - t0:.0f}s", flush=True)

    # 标签: 只在 topk 候选上做 softmax(score/tau), 其余为 0
    P = np.zeros((len(snaps), 28), dtype=np.float32)
    best = np.zeros(len(snaps), dtype=np.int8)
    for i, cs in enumerate(cands):
        sc = S[i, :len(cs)]
        ok = np.isfinite(sc)
        if ok.sum() == 0:
            best[i] = cs[0]
            P[i, cs[0]] = 1.0
            continue
        z = np.where(ok, sc, -1e9)
        z = (z - z[ok].max()) / args.tau
        w = np.exp(np.where(ok, z, -np.inf))
        w = w / w.sum()
        for ci, a in enumerate(cs):
            P[i, a] = w[ci]
        best[i] = cs[int(np.nanargmax(sc))]

    np.savez_compressed(args.out, feats=feats, target=P, bests=best,
                        scores=S, labels=np.zeros(len(snaps), np.float32))
    # 教师与当前策略的分歧率 = 这批标签能带来多少改动
    with torch.no_grad():
        q, _ = model(torch.from_numpy(feats))
        q = q.masked_fill(~torch.from_numpy(masks), -1e9)
        cur = q.argmax(-1).numpy()
    print(f"已存 {args.out}  {len(snaps)} 条; 教师与当前策略分歧 "
          f"{(cur != best).mean():.1%}")
    gap = np.nanmax(S, axis=1) - np.array(
        [S[i, cands[i].index(int(cur[i]))] if int(cur[i]) in cands[i]
         else np.nan for i in range(len(snaps))])
    print(f"教师最优 vs 当前选择的奖励差: 均 {np.nanmean(gap):+.4f} "
          f"(>0 表示教师找到更好的)")


if __name__ == "__main__":
    main()
