"""生成价值网络预训练数据: 局面特征 -> 牌型分析器 E 值(期望巡数)。

从 v31n 自对弈中采样所有座位的出牌决策点(14 张状态), 标签 =
C 引擎(mj159 LUT)对每个候选弃牌 E 的最小值(kai_max=1, ρ=1 口径),
同时记录最优弃牌(可供行为克隆用)。

并行: 进程池, 每 worker 跑一批种子。

用法: python -m tools.rl_value_data --games 2000 --procs 12 --out models/hv_value_data.npz
"""

import argparse
import multiprocessing as mp
import os
import time

import numpy as np

from backend.ai.bot_native import NativeV31
from backend.analysis import hv_native
from backend.game.engine import Game
from backend.rl.features_v2 import encode_state


def _visible_for(game, seat):
    visible = [0] * 28
    for q in game.players:
        for t in q.discards:
            visible[t] += 1
        for m in q.melds:
            visible[m["tile"]] += 3 if m["type"] == "peng" else 4
    for t, n in enumerate(game.players[seat].hand_counts):
        visible[t] += n
    return visible


def work(payload):
    """打一批局, 返回 (feats, labels, best_tiles)。"""
    seed0, n_games, kai_max = payload
    feats, labels, bests = [], [], []
    for gi in range(n_games):
        g = Game(seed=seed0 + gi, human_seat=-1)
        bots = {i: NativeV31(g, i) for i in range(4)}
        guard = 0
        while g.phase != "game_over" and guard < 500:
            guard += 1
            if g.phase == "discard_wait":
                s = g.turn
                hand = g.players[s].hand_counts
                if sum(hand) % 3 == 2:
                    vis = _visible_for(g, s)
                    if hv_native.set_hand(hand, vis, rho=1.0, kaizen=True,
                                          kai_max=kai_max, kai_topk=6):
                        best_e, best_t = 1e18, -1
                        for t in range(28):
                            if hand[t] <= 0:
                                continue
                            e = hv_native.e_after_discard(t)
                            if e < best_e:
                                best_e, best_t = e, t
                        feats.append(encode_state(g, s))
                        labels.append(best_e)
                        bests.append(best_t)
                g.action_discard(s, bots[s].choose_discard())
            else:
                ss = list(g.pending_actions.keys())[0]
                pend = g.pending_actions[ss]
                b = bots[ss]
                if pend.get("gang") and b.decide_gang(g.last_discard, "ming"):
                    g.action_gang(ss)
                elif pend.get("peng") and b.decide_peng(g.last_discard):
                    g.action_peng(ss)
                else:
                    g.action_pass(ss)
    return (np.asarray(feats, dtype=np.float32),
            np.asarray(labels, dtype=np.float32),
            np.asarray(bests, dtype=np.int8))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=2000)
    ap.add_argument("--seed0", type=int, default=1000000)
    ap.add_argument("--procs", type=int, default=12)
    ap.add_argument("--kai-max", type=int, default=1)
    ap.add_argument("--out", type=str, default="models/hv_value_data.npz")
    args = ap.parse_args()

    per = max(1, args.games // args.procs)
    tasks = [(args.seed0 + i * per, per, args.kai_max)
             for i in range(args.procs)]
    t0 = time.time()
    with mp.get_context("fork").Pool(args.procs) as pool:
        rets = pool.map(work, tasks)
    feats = np.concatenate([r[0] for r in rets])
    labels = np.concatenate([r[1] for r in rets])
    bests = np.concatenate([r[2] for r in rets])
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez_compressed(args.out, feats=feats, labels=labels, bests=bests)
    dt = time.time() - t0
    print(f"状态数={len(labels)}  E标签: mean={labels.mean():.2f} "
          f"std={labels.std():.2f} min={labels.min():.2f} max={labels.max():.2f}")
    print(f"耗时 {dt:.0f}s -> {args.out}")


if __name__ == "__main__":
    main()
