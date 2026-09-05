"""Search teacher with explicit hidden-world and reward semantics.

The CLI defaults to first-win, uniform count-consistent hidden-world sampling,
raw terminal score, and deployed HV claim rules. Candidates share the same set
of worlds. Continuation bots observe only their own encoded features. This is
an approximate world model, not a calibrated posterior from opponent history.

The low-level score_candidates function retains explicit legacy fixed-world
support for older callers. Repeating a fixed hidden world only reduces policy
sampling noise; it does not integrate hidden-state uncertainty.

Use check_world_teacher to select and evaluate actions on disjoint worlds before
claiming teacher improvement. In-sample max score is selection-biased.
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
from backend.rl.hidden_worlds import sample_world


def claim_factory(model, claim):
    if claim == "hv":
        from tools.dagger_collect import _Seat
        return lambda g, s: _Seat(g, s, model)
    from backend.ai.bot_native import NativeV31
    return NativeV31


def collect_states(model, n_games, seed0, snap_p, seed, bloody=True, temp=1.0,
                   claim="v31"):
    rng = np.random.default_rng(seed)
    games = [Game(seed=seed0 + i, human_seat=-1, bloody=bloody)
             for i in range(n_games)]
    _, snaps = cf_collect.run_games(model, "cpu", games, temp, snap_p, rng,
                                    make_claim_bot=claim_factory(model, claim))
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


def score_candidates(model, snaps, cands, rolls, temp, chunk_states=150,
                     world_mode="fixed", seed=0, return_rollouts=False,
                     reward="rank", claim="v31"):
    """同墙重放: 返回 (n_states, max_k) 的平均奖励矩阵(无效位 nan)。"""
    maxk = max(len(c) for c in cands)
    S = np.full((len(snaps), maxk), np.nan, dtype=np.float32)
    returns = np.full((len(snaps), maxk, rolls), np.nan, dtype=np.float32)
    rng = np.random.default_rng(seed)
    if world_mode not in ("fixed", "resample") or rolls < 1 or reward not in ("rank", "score"):
        raise ValueError("Invalid world mode, reward, or rollout count")
    if world_mode == "fixed":
        print("WARNING: legacy fixed-world rollouts do not average hidden-world uncertainty", flush=True)
    for lo in range(0, len(snaps), chunk_states):
        hi = min(lo + chunk_states, len(snaps))
        jobs, keys = [], []
        for i in range(lo, hi):
            worlds = [sample_world(snaps[i]["game"], snaps[i]["seat"], rng)
                      if world_mode == "resample" else snaps[i]["game"]
                      for _ in range(rolls)]
            for ci, a in enumerate(cands[i]):
                for wi, world in enumerate(worlds):
                    g = copy.deepcopy(world)
                    g.action_discard(snaps[i]["seat"], a)
                    jobs.append(g)
                    keys.append((i, ci, wi))
        if not jobs:
            continue
        jobs, _ = cf_collect.run_games(model, "cpu", jobs, temp,
                                       make_claim_bot=claim_factory(model, claim))
        acc = {}
        for g, (i, ci, wi) in zip(jobs, keys):
            seat = snaps[i]["seat"]
            r = g.players[seat].score_delta if reward == "score" else cf_collect.default_reward(g, seat)
            returns[i, ci, wi] = r
            acc.setdefault((i, ci), []).append(r)
        for (i, ci), vs in acc.items():
            S[i, ci] = float(np.mean(vs))
        print(f"  ...状态 {hi}/{len(snaps)}  重放 {len(jobs)} 局", flush=True)
    return (S, returns) if return_rollouts else S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/bc_r2_s3.pt")
    ap.add_argument("--states", type=int, default=4000)
    ap.add_argument("--games", type=int, default=0,
                    help="0 = 按 states 自动推算局数")
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--rolls", type=int, default=8)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--tau", type=float, default=0.5, help="标签 softmax 温度")
    ap.add_argument("--seed0", type=int, default=90000000)
    ap.add_argument("--out", default="models/teach_r1.npz")
    ap.add_argument("--world-mode", choices=["fixed", "resample"], default="resample")
    ap.add_argument("--bloody", action="store_true", help="Legacy blood-battle requires --world-mode fixed")
    ap.add_argument("--seed", type=int, default=20260905)
    ap.add_argument("--reward", choices=["rank", "score"], default="score")
    ap.add_argument("--claim", choices=["v31", "hv"], default="hv")
    args = ap.parse_args()
    if args.bloody and args.world_mode != "fixed":
        ap.error("Blood-battle needs a history-conditioned sampler; use first-win or explicit legacy fixed")
    if args.states < 1 or args.rolls < 1 or args.topk < 1 or args.tau <= 0:
        ap.error("states, rolls, topk and tau must be positive")
    torch.manual_seed(args.seed)

    ck = torch.load(args.model, map_location="cpu", weights_only=True)
    model = build_model(ck["size"])
    model.load_state_dict(ck["model"], strict=False)
    model.eval()

    # 血战一局约 45 次弃牌 × snap_p 0.06 ≈ 2.7 个状态/局
    n_games = args.games or max(50, int(args.states / 2.7))
    t0 = time.time()
    snaps = collect_states(model, n_games, args.seed0, 0.06, args.seed,
                           bloody=args.bloody, temp=args.temp, claim=args.claim)
    snaps = snaps[:args.states]
    print(f"采到 {len(snaps)} 个状态 ({n_games} 局, {time.time() - t0:.0f}s)",
          flush=True)

    cands, feats, masks = topk_actions(model, snaps, args.topk)
    t0 = time.time()
    S, returns = score_candidates(model, snaps, cands, args.rolls, args.temp,
                                  world_mode=args.world_mode, seed=args.seed,
                                  return_rollouts=True, reward=args.reward, claim=args.claim)
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

    candidates = np.full(S.shape, -1, dtype=np.int8)
    for i, cs in enumerate(cands):
        candidates[i, :len(cs)] = cs
    np.savez_compressed(args.out, feats=feats, target=P, bests=best,
                        scores=S, rollout_returns=returns, candidates=candidates,
                        legal_mask=masks, value_valid=np.zeros(len(snaps), bool),
                        game_id=np.array([args.seed0 + sn['gi'] for sn in snaps]),
                        world_mode=args.world_mode, reward=args.reward,
                        bloody=args.bloody, policy_model=args.model, claim=args.claim,
                        labels=np.zeros(len(snaps), np.float32))
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
