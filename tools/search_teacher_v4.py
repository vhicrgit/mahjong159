"""search_teacher 的 v4 版: 快照存 v4 特征(带对手模型段), 学生是 v4 网络。

设计(成本约束下的折中):
- 主线 rollout 用 v2 策略(bc_r2_s3, 便宜、无需 tracker 参与决策), 但**全程
  维护快照座位的 3 个 OppTracker**, 快照特征 = v2 628 维 + tracker 90 维。
- 候选 topk 与重放世界模型都用 v2 策略 —— 教师世界模型与学生(v4)有错位,
  但 expert iteration 本来就在策略错位下迭代; 重放若也带 tracker, 成本
  ×4 不可行。
- 标签 = softmax(同墙重放平均分/τ), 只覆盖 v2 策略的 top-3 候选。
  v4 学生从中学"在同样的候选里, 名次收益偏好哪张" —— 这个偏好含防守,
  而 v4 特征恰好能表达"对手可能持有/听什么", 两者是互补的:
  E 标签不含名次信息(v4 学不到动机), v2 特征不含对手信息(教师偏好
  无法被条件化表达)。这是 1+2 单独都失败后剩下的唯一组合假设。

用法:
  python -m tools.search_teacher_v4 --rollout-model models/bc_r2_s3.pt \
      --states 3000 --topk 3 --rolls 64 --out models/teachv4_0.npz
"""

import argparse
import copy
import time

import numpy as np
import torch
import torch.nn.functional as F

from backend.analysis.opp_model import OppTracker
from backend.game.engine import Game
from backend.game.bot_driver import finish_self_gangs
from backend.rl import cf_collect
from backend.rl.features_v2 import encode_state as encode_v2
from backend.rl.features_v4 import opp_features
from backend.rl.model import build_model, legal_discard_mask
from tools.search_teacher import score_candidates


def topk_v2(model, snaps, k):
    """候选用 v2 特征(快照 feat 的前 628 维)从世界模型取 —— v2 网络吃不下
    718 维, 且世界模型本来就是 v2 的。"""
    feats = np.stack([sn["feat"][:628] for sn in snaps])
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
    return out


def make_trackers(g, hero, n_init, beam, seed):
    trs = {}
    for rel in (1, 2, 3):
        opp = (hero + rel) % 4
        trs[opp] = OppTracker(opp, list(g.players[hero].hand_counts),
                              n_init=n_init, beam=beam, policy=False,
                              seed=seed * 10 + opp, hero_seat=hero)
    for tr in trs.values():
        tr.notify_deal(g.dealer)
    return trs


def collect_v4(model, n_games, seed0, snap_p, seed, n_init, beam):
    """v2 策略走局 + 每局为每个座位维护 tracker(快照时按座位取)。"""
    rng = np.random.default_rng(seed)
    games = [Game(seed=seed0 + i, human_seat=-1, bloody=True)
             for i in range(n_games)]
    # trs_all[i][hero] = {opp: tracker}
    trs_all = [{h: make_trackers(g, h, n_init, beam, seed0 + i)
                for h in range(4)} for i, g in enumerate(games)]
    snaps = []
    it = 0
    while it < 900:
        it += 1
        for i, g in enumerate(games):
            while g.phase == "react_wait":
                s = list(g.pending_actions.keys())[0]
                tile, disc = g.last_discard, g.last_discarder
                from backend.ai.bot_native import NativeV31
                b = NativeV31(g, s)
                if g.pending_actions[s].get("gang") and \
                        b.decide_gang(tile, "ming"):
                    act = "gang"
                    ev = g.action_gang(s)
                elif g.pending_actions[s].get("peng") and b.decide_peng(tile):
                    act = "peng"
                    ev = g.action_peng(s)
                else:
                    act = None
                    ev = g.action_pass(s)
                for hero in range(4):
                    for tr in trs_all[i][hero].values():
                        tr.notify_claim(s, act, tile, disc)
                if ev.get("event") in ("draw", "gang_draw"):
                    s2, t2 = ev["seat"], ev.get("tile")
                    for hero in range(4):
                        for tr in trs_all[i][hero].values():
                            tr.notify_draw(s2, t2 if hero == s2 else None,
                                           g.wall_remaining())
        from backend.ai.bot_native import NativeV31
        for i, g in enumerate(games):
            finish_self_gangs(g, NativeV31,
                             [tr for per_hero in trs_all[i].values() for tr in per_hero.values()])
        idx = [i for i, g in enumerate(games) if g.phase == "discard_wait"]
        if not idx:
            break
        gs = [games[i] for i in idx]
        ss = [games[i].turn for i in idx]
        feats = np.stack([encode_v2(g, s) for g, s in zip(gs, ss)])
        masks = torch.stack([legal_discard_mask(g.players[s].hand_counts)
                             for g, s in zip(gs, ss)])
        with torch.no_grad():
            q, _ = model(torch.from_numpy(feats))
            lp = F.log_softmax(q.masked_fill(~masks, -1e9), dim=-1)
            a = torch.multinomial(lp.exp(), 1).squeeze(-1)
        for k, i in enumerate(idx):
            g, s = games[i], ss[k]
            t = int(a[k])
            if rng.random() < snap_p:
                p = lp[k].exp().numpy().copy()
                p[t] = 0.0
                if p.sum() > 1e-9:
                    alt = int(rng.choice(len(p), p=p / p.sum()))
                    v4feat = np.concatenate(
                        [feats[k], opp_features(trs_all[i][s], s)])
                    snaps.append({"gi": i, "game": copy.deepcopy(g),
                                  "seat": s, "tile": t, "alt": alt,
                                  "feat": v4feat,
                                  "mask": masks[k].numpy()})
            for hero in range(4):
                for tr in trs_all[i][hero].values():
                    tr.notify_discard(s, t, g.wall_remaining())
            ev = g.action_discard(s, t)
            if ev.get("event") == "draw":
                s2, t2 = ev["seat"], ev["tile"]
                for hero in range(4):
                    for tr in trs_all[i][hero].values():
                        tr.notify_draw(s2, t2 if hero == s2 else None,
                                       g.wall_remaining())
    return snaps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollout-model", default="models/bc_r2_s3.pt")
    ap.add_argument("--states", type=int, default=3000)
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--rolls", type=int, default=64)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--tau", type=float, default=0.45)
    ap.add_argument("--n-init", type=int, default=1500)
    ap.add_argument("--beam", type=int, default=300)
    ap.add_argument("--seed0", type=int, default=220000000)
    ap.add_argument("--out", default="models/teachv4_0.npz")
    args = ap.parse_args()

    ck = torch.load(args.rollout_model, map_location="cpu", weights_only=True)
    model = build_model(ck["size"], feat_dim=ck.get("feat_dim", 628))
    model.load_state_dict(ck["model"], strict=False)
    model.eval()

    n_games = max(50, int(args.states / 2.7))
    t0 = time.time()
    snaps = collect_v4(model, n_games, args.seed0, 0.06, 7,
                       args.n_init, args.beam)
    snaps = snaps[:args.states]
    print(f"采到 {len(snaps)} 个 v4 状态 ({n_games} 局, "
          f"{time.time() - t0:.0f}s)", flush=True)

    # 候选与打分都走 v2 世界模型(便宜); 特征已是 v4
    cands = topk_v2(model, snaps, args.topk)
    t0 = time.time()
    S = score_candidates(model, snaps, cands, args.rolls, args.temp)
    print(f"打分完成 {time.time() - t0:.0f}s", flush=True)

    feats = np.stack([sn["feat"] for sn in snaps])
    masks = np.stack([sn["mask"] for sn in snaps])
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
        w /= w.sum()
        for ci, a_ in enumerate(cs):
            P[i, a_] = w[ci]
        best[i] = cs[int(np.nanargmax(sc))]
    np.savez_compressed(args.out, feats=feats, masks=masks, target=P,
                        bests=best, scores=S,
                        labels=np.zeros(len(snaps), np.float32))
    print(f"已存 {args.out}  {len(snaps)} 条 (v4 718维)")


if __name__ == "__main__":
    main()
