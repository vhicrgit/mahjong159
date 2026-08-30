"""DAgger 数据生成 —— 在网络自己走出来的状态上向分析器要标签。

为什么需要: 原始 BC 数据(tools/rl_value_data.py)的状态是 **v31n 自对弈** 走出来
的, 而网络部署时走的是自己的分布。实测这个漂移值 0.15 分/局:

  分析器弃牌 + E 碰杠            血战 -0.061 分/局 vs v31n
  92%模仿它的网络 + E 碰杠       血战 -0.210
  (同样换成 v31 碰杠: -0.104 / -0.324, 所以碰杠另占 ~0.11)

DAgger 是这个漂移的标准解法, 而且是纯监督问题 —— 没有 RL 那些信噪比麻烦。

分两段跑: 网络自对弈采状态(单进程, 批量前向, 快), 标签用 C 引擎多进程打
(纯 C 无 torch, fork 安全; torch 在 fork 的 worker 里做前向会死锁, 实测过)。

用法:
  python -m tools.dagger_collect --model models/hv_value_pretrained_v2.pt \
      --games 3000 --keep 0.35 --kai 1 --procs 8 --out models/dagger_r1.npz
"""

import argparse
import multiprocessing as mp
import time

import numpy as np
import torch

from backend.analysis import hv_native
from backend.game.engine import Game
from backend.rl import cf_collect
from backend.rl.features_v2 import encode_state
from backend.rl.model import build_model, legal_discard_mask
from backend.rules.tiles import TILE_COUNT


def visible_of(g, seat):
    v = [0] * TILE_COUNT
    for q in g.players:
        for t in q.discards:
            v[t] += 1
        for m in q.melds:
            v[m["tile"]] += 3 if m["type"] == "peng" else 4
    for t, n in enumerate(g.players[seat].hand_counts):
        v[t] += n
    return v


class _Seat:
    """NN 弃牌 + 分析器 E 碰杠(实测比 v31 规则碰杠好 0.11 分/局)。"""

    def __init__(self, game, seat, model, kai_claim=0):
        self.game, self.seat, self.model, self.kc = game, seat, model, kai_claim

    def _hv(self):
        hv_native.set_hand(list(self.game.players[self.seat].hand_counts),
                           visible_of(self.game, self.seat), 1.0,
                           self.kc > 0, 2, self.kc, 6)
        return hv_native

    def decide_peng(self, tile):
        return self._hv().decide_peng(tile)

    def decide_gang(self, tile, kind):
        return self._hv().decide_gang(tile, kind)


def rollout_states(model, n_games, seed0, keep, bloody, temp, seed):
    """网络自对弈, 按 keep 概率留下决策点的 (feat, hand, visible)。"""
    rng = np.random.default_rng(seed)
    games = [Game(seed=seed0 + i, human_seat=-1, bloody=bloody)
             for i in range(n_games)]
    out = []
    it = 0
    while it < 900:
        it += 1
        for g in games:
            if g.phase == "react_wait":
                _react_hv(g, model)
        idx = [i for i, g in enumerate(games) if g.phase == "discard_wait"]
        if not idx:
            break
        gs = [games[i] for i in idx]
        ss = [games[i].turn for i in idx]
        feats = np.stack([encode_state(g, s) for g, s in zip(gs, ss)])
        masks = torch.stack([legal_discard_mask(g.players[s].hand_counts)
                             for g, s in zip(gs, ss)])
        x = torch.from_numpy(feats)
        with torch.no_grad():
            q, _ = model(x)
            p = torch.softmax(q.masked_fill(~masks, -1e9) / max(temp, 1e-6),
                              dim=-1)
            a = torch.multinomial(p, 1).squeeze(-1).numpy()
        for k, i in enumerate(idx):
            g, s = games[i], ss[k]
            if rng.random() < keep:
                out.append((feats[k], list(g.players[s].hand_counts),
                            visible_of(g, s)))
            g.action_discard(s, int(a[k]))
    return out


def _react_hv(g, model):
    while g.phase == "react_wait":
        s = list(g.pending_actions.keys())[0]
        b = _Seat(g, s, model)
        if g.pending_actions[s].get("gang") and \
                b.decide_gang(g.last_discard, "ming"):
            g.action_gang(s)
        elif g.pending_actions[s].get("peng") and b.decide_peng(g.last_discard):
            g.action_peng(s)
        else:
            g.action_pass(s)


def _label_chunk(payload):
    """纯 C 打标签: 返回 (best_tile, best_E)。fork 安全(不碰 torch)。"""
    hands, vises, kai = payload
    bt, be = [], []
    for hand, vis in zip(hands, vises):
        hv_native.set_hand(hand, vis, 1.0, kai > 0, 2, kai, 6)
        best_e, best_t = 1e18, -1
        for t in range(TILE_COUNT):
            if hand[t] <= 0:
                continue
            e = hv_native.e_after_discard(t)
            if e < best_e:
                best_e, best_t = e, t
        bt.append(best_t)
        be.append(best_e)
    return np.array(bt, dtype=np.int8), np.array(be, dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/hv_value_pretrained_v2.pt")
    ap.add_argument("--games", type=int, default=3000)
    ap.add_argument("--keep", type=float, default=0.35)
    ap.add_argument("--kai", type=int, default=1, help="标签的换型档位")
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--bloody", action="store_true", default=True)
    ap.add_argument("--first-win", dest="bloody", action="store_false")
    ap.add_argument("--seed0", type=int, default=80000000)
    ap.add_argument("--out", default="models/dagger_r1.npz")
    args = ap.parse_args()

    ck = torch.load(args.model, map_location="cpu", weights_only=True)
    model = build_model(ck["size"])
    model.load_state_dict(ck["model"], strict=False)
    model.eval()

    t0 = time.time()
    st = rollout_states(model, args.games, args.seed0, args.keep,
                        args.bloody, args.temp, seed=1)
    print(f"网络自对弈 {args.games} 局 -> {len(st)} 个状态  "
          f"{time.time() - t0:.0f}s", flush=True)

    feats = np.stack([s[0] for s in st])
    hands = [s[1] for s in st]
    vises = [s[2] for s in st]
    per = (len(st) + args.procs - 1) // args.procs
    tasks = [(hands[i:i + per], vises[i:i + per], args.kai)
             for i in range(0, len(st), per)]
    t0 = time.time()
    with mp.get_context("fork").Pool(args.procs) as pool:
        rets = pool.map(_label_chunk, tasks)
    bests = np.concatenate([r[0] for r in rets])
    labels = np.concatenate([r[1] for r in rets])
    print(f"标签 kai={args.kai} × {args.procs} 进程  {time.time() - t0:.0f}s")

    np.savez_compressed(args.out, feats=feats, labels=labels, bests=bests)
    print(f"已存 {args.out}  {len(bests)} 条  E: 均 {labels.mean():.2f} "
          f"标准差 {labels.std():.2f}")


if __name__ == "__main__":
    main()
