"""v4(带对手模型特征)的 DAgger 数据生成。

与 tools/dagger_collect.py 的区别: 每局为四个座位各维护 3 个 OppTracker
(policy=False, 结构似然, ~9ms/决策/局), 采状态时存 718 维 v4 特征。
走局面的策略仍是 r2 模型(628维 v2 特征), 标签仍由 C 引擎的分析器 E 打
(kai=0), 与 dagger_collect 完全同管线 —— 唯一变化是特征多了对手段。

用法:
  python -m tools.dagger_collect_v4 --model models/bc_k0_r2.pt --games 2500 \
      --keep 0.5 --procs 9 --out models/k0_nn_v4.npz
"""

import argparse
import multiprocessing as mp
import time

import numpy as np
import torch

from backend.ai.bot_native import NativeV31
from backend.analysis.opp_model import OppTracker
from backend.game.engine import Game
from backend.rl.features_v2 import encode_state as encode_v2
from backend.rl.features_v4 import opp_features
from backend.rl.model import build_model, legal_discard_mask
from tools.dagger_collect import _label_chunk, visible_of


def make_trackers(g, n_init, beam, seed):
    """每局 4 座位 × 3 对手 = 12 个 tracker, key=(hero, opp)。"""
    trs = {}
    for hero in range(4):
        for rel in (1, 2, 3):
            opp = (hero + rel) % 4
            trs[(hero, opp)] = OppTracker(
                opp, list(g.players[hero].hand_counts), n_init=n_init,
                beam=beam, policy=False, seed=seed * 100 + hero * 10 + opp,
                hero_seat=hero)
    for tr in trs.values():
        tr.notify_deal(g.dealer)
    return trs


def feed(trs, kind, **kw):
    for tr in trs.values():
        if kind == "draw":
            tr.notify_draw(kw["seat"], kw["tile"], kw["wr"])
        elif kind == "discard":
            tr.notify_discard(kw["seat"], kw["tile"], kw["wr"])
        elif kind == "claim":
            tr.notify_claim(kw["seat"], kw["action"], kw["tile"],
                            kw["discarder"])


def rollout(model, n_games, seed0, keep, bloody, rng, n_init, beam,
            feat_dim=628):
    games = [Game(seed=seed0 + i, human_seat=-1, bloody=bloody)
             for i in range(n_games)]
    trs = [make_trackers(g, n_init, beam, seed0 + i)
           for i, g in enumerate(games)]
    out = []
    it = 0
    while it < 900:
        it += 1
        for i, g in enumerate(games):
            while g.phase == "react_wait":
                s = list(g.pending_actions.keys())[0]
                tile, disc = g.last_discard, g.last_discarder
                # 碰/杠决策与采集管线一致: v31 规则(评估口径)
                b = NativeV31(g, s)
                if g.pending_actions[s].get("gang") and \
                        b.decide_gang(tile, "ming"):
                    feed(trs[i], "claim", seat=s, action="gang", tile=tile,
                         discarder=disc)
                    ev = g.action_gang(s)
                elif g.pending_actions[s].get("peng") and \
                        b.decide_peng(tile):
                    feed(trs[i], "claim", seat=s, action="peng", tile=tile,
                         discarder=disc)
                    ev = g.action_peng(s)
                else:
                    feed(trs[i], "claim", seat=s, action=None, tile=tile,
                         discarder=disc)
                    ev = g.action_pass(s)
                if ev.get("event") in ("draw", "gang_draw"):
                    seat2, t2 = ev["seat"], ev.get("tile")
                    # hero 摸牌喂真值(它自己看得见), 其余 tracker 喂 None
                    for (hero, opp), tr in trs[i].items():
                        tr.notify_draw(seat2,
                                       t2 if hero == seat2 else None,
                                       g.wall_remaining())
        idx = [i for i, g in enumerate(games) if g.phase == "discard_wait"]
        if not idx:
            break
        gs = [games[i] for i in idx]
        ss = [games[i].turn for i in idx]
        if feat_dim == 718:
            # v4 模型: 决策特征带 tracker 段(与存储特征同口径)
            feats = np.stack([
                np.concatenate([encode_v2(g, s), opp_features(trs[i], s)])
                for i, g, s in zip(idx, gs, ss)])
        else:
            feats = np.stack([encode_v2(g, s) for g, s in zip(gs, ss)])
        masks = torch.stack([legal_discard_mask(g.players[s].hand_counts)
                             for g, s in zip(gs, ss)])
        with torch.no_grad():
            q, _ = model(torch.from_numpy(feats))
            p = torch.softmax(q.masked_fill(~masks, -1e9), dim=-1)
            acts = torch.multinomial(p, 1).squeeze(-1).numpy()
        for k, i in enumerate(idx):
            g, s = games[i], ss[k]
            if rng.random() < keep:
                # feat_dim=718 时 feats[k] 已是 v4(决策与存储同口径), 别重拼
                v4 = feats[k] if feats.shape[1] == 718 else \
                    np.concatenate([feats[k], opp_features(trs[i], s)])
                out.append((v4, list(g.players[s].hand_counts),
                            visible_of(g, s)))
            t = int(acts[k])
            feed(trs[i], "discard", seat=s, tile=t, wr=g.wall_remaining())
            ev = g.action_discard(s, t)
            if ev.get("event") == "draw":
                seat2, t2 = ev["seat"], ev["tile"]
                # hero 摸牌给真值(该 hero 的 3 个 tracker), 别人摸牌给 None
                for (hero, opp), tr in trs[i].items():
                    tr.notify_draw(seat2, t2 if hero == seat2 else None,
                                   g.wall_remaining())
            elif ev.get("event") == "react":
                pass    # react 在下一轮循环开头处理
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/bc_k0_r2.pt")
    ap.add_argument("--games", type=int, default=2500)
    ap.add_argument("--keep", type=float, default=0.5)
    ap.add_argument("--kai", type=int, default=0)
    ap.add_argument("--procs", type=int, default=9)
    ap.add_argument("--n-init", type=int, default=1500)
    ap.add_argument("--beam", type=int, default=300)
    ap.add_argument("--bloody", action="store_true", default=True)
    ap.add_argument("--first-win", dest="bloody", action="store_false")
    ap.add_argument("--seed0", type=int, default=140000000)
    ap.add_argument("--out", default="models/k0_nn_v4.npz")
    args = ap.parse_args()

    ck = torch.load(args.model, map_location="cpu", weights_only=True)
    model = build_model(ck["size"], feat_dim=ck.get("feat_dim", 628))
    model.load_state_dict(ck["model"], strict=False)
    model.eval()

    rng = np.random.default_rng(3)
    t0 = time.time()
    st = rollout(model, args.games, args.seed0, args.keep, args.bloody,
                 rng, args.n_init, args.beam,
                 feat_dim=ck.get("feat_dim", 628))
    print(f"自对弈 {args.games} 局 -> {len(st)} 个 v4 状态  "
          f"{time.time() - t0:.0f}s", flush=True)

    feats = np.stack([s[0] for s in st])
    hands = np.array([s[1] for s in st], dtype=np.int8)
    vises = np.array([s[2] for s in st], dtype=np.int8)
    # 立即落盘中间产物: 上一轮 6 个进程在标签阶段静默消失, 22 分钟 rollout
    # 全部丢失。raw 先写盘, 标签崩了也能续。
    raw = args.out.replace(".npz", "_raw.npz")
    np.savez_compressed(raw, feats=feats, hands=hands, vises=vises)
    print(f"raw 已落盘 {raw}", flush=True)

    per = 2000
    tasks = [(hands[i:i + per].tolist(), vises[i:i + per].tolist(), args.kai)
             for i in range(0, len(st), per)]
    t0 = time.time()
    if args.procs <= 1:
        rets = [_label_chunk(t) for t in tasks]      # 同进程, 不 fork
    else:
        with mp.get_context("fork").Pool(args.procs) as pool:
            rets = pool.map(_label_chunk, tasks, chunksize=1)
    bests = np.concatenate([r[0] for r in rets])
    labels = np.concatenate([r[1] for r in rets])
    evec = np.concatenate([r[2] for r in rets])
    print(f"标签 kai={args.kai} × {args.procs} 进程  {time.time() - t0:.0f}s")
    np.savez_compressed(args.out, feats=feats, labels=labels, bests=bests,
                        evec=evec)
    print(f"已存 {args.out}  {len(bests)} 条  feats {feats.shape[1]} 维")


if __name__ == "__main__":
    main()
