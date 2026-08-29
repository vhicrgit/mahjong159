"""实证对拍 961000 第4巡: 打5饼(v31 的选择) vs 打7条(用户的主张)。

两部分:
1. 列出两个打法的有效牌明细(每种有效牌 + 剩余张数), 核对"445饼 的 3/4/6饼"
   和 "7条 只能当将" 这两个结构性论断
2. 复盘到该状态, 用 N 个共享世界把两个候选推演到底(v31n), 比期望塑形得分

用法: python -m tools.perf.arbitrate_961000 --worlds 2048
"""

import argparse
import copy
import multiprocessing as mp
import random

import numpy as np

from backend.ai.bot_native import NativeV31
from backend.game.engine import Game
from backend.native import native
from backend.rl.gen_offline import _shaped_scores
from backend.rl.world_grpo import sample_world
from backend.rules.tiles import tile_name


def replay_to(seed, hero, target_turn):
    """用 v31n 复盘到座 hero 的第 target_turn 次出牌决策前。"""
    g = Game(seed=seed, human_seat=-1)
    bots = {i: NativeV31(g, i) for i in range(4)}
    tc, guard = 0, 0
    while g.phase != "game_over" and guard < 500:
        guard += 1
        if g.phase == "discard_wait":
            if g.turn == hero:
                tc += 1
                if tc == target_turn:
                    g.log = []
                    return g
            g.action_discard(g.turn, bots[g.turn].choose_discard())
        else:
            s = list(g.pending_actions.keys())[0]
            b = bots[s]
            if g.pending_actions[s].get("gang") and \
                    b.decide_gang(g.last_discard, "ming"):
                g.action_gang(s)
            elif g.pending_actions[s].get("peng") and \
                    b.decide_peng(g.last_discard):
                g.action_peng(s)
            else:
                g.action_pass(s)
    return g


def useful_list(hand13, unseen):
    """有效牌明细: {tile: 剩余张数}。听牌时=能胡的, 否则=能降向听的。"""
    s = native.shanten(hand13)
    out = {}
    for t in range(28):
        if hand13[t] >= 4:
            continue
        h = list(hand13)
        h[t] += 1
        if (s == 0 and native.is_win(h)) or (s > 0 and native.shanten(h) < s):
            out[t] = unseen[t]
    return s, out


def rollout_world(args):
    snap, seat, tile, wseed = args
    hands, wall = sample_world(snap, random.Random(wseed), hero_seat=seat)
    g = copy.deepcopy(snap)
    for s2, h in hands.items():
        g.players[s2].hand = list(h)
    g.wall = list(wall)
    bots = {i: NativeV31(g, i) for i in range(4)}
    g.action_discard(seat, tile)
    guard = 0
    while g.phase != "game_over" and guard < 500:
        guard += 1
        if g.phase == "discard_wait":
            g.action_discard(g.turn, bots[g.turn].choose_discard())
        else:
            s = list(g.pending_actions.keys())[0]
            b = bots[s]
            if g.pending_actions[s].get("gang") and \
                    b.decide_gang(g.last_discard, "ming"):
                g.action_gang(s)
            elif g.pending_actions[s].get("peng") and \
                    b.decide_peng(g.last_discard):
                g.action_peng(s)
            else:
                g.action_pass(s)
    return float(_shaped_scores(g, 0.02)[seat]), 1 if g.winner == seat else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=961000)
    ap.add_argument("--turn", type=int, default=4)
    ap.add_argument("--worlds", type=int, default=2048)
    ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--cands", default="12,6")   # 5饼=12, 7条=6
    args = ap.parse_args()

    g = replay_to(args.seed, 0, args.turn)
    seat = 0
    p = g.players[seat]
    hand = list(p.hand_counts)
    print("复盘到 seed=%d 座%d 第%d巡: 手牌 %s  墙剩 %d"
          % (args.seed, seat, args.turn,
             " ".join(tile_name(t) for t in p.hand), g.wall_remaining()))

    visible = [0] * 28
    for q in g.players:
        for t in q.discards:
            visible[t] += 1
        for m in q.melds:
            visible[m["tile"]] += 3 if m["type"] == "peng" else 4
    for t, n in enumerate(hand):
        visible[t] += n
    unseen = [max(0, 4 - v) for v in visible]

    cands = [int(x) for x in args.cands.split(",")]
    print("\n=== 两个打法的有效牌明细 ===")
    for t in cands:
        h13 = list(hand)
        h13[t] -= 1
        s, ul = useful_list(h13, unseen)
        tot = sum(ul.values())
        detail = " ".join(f"{tile_name(k)}×{v}" for k, v in sorted(ul.items()))
        print(f"打{tile_name(t)}: 向听{s} 进张合计 {tot} 张")
        print(f"    {detail}")

    print(f"\n=== 共享世界推演 ({args.worlds} 世界/候选, v31n 四家) ===")
    tasks = []
    for t in cands:
        for w in range(args.worlds):
            tasks.append((copy.deepcopy(g), seat, t, 310000 + w))
    with mp.Pool(args.procs) as pool:
        rets = pool.map(rollout_world, tasks, chunksize=64)
    res = {t: [] for t in cands}
    win = {t: [] for t in cands}
    for (snap_, seat_, t, wseed), (r, w) in zip(tasks, rets):
        res[t].append(r)
        win[t].append(w)
    print(f"{'候选':>6s} {'期望塑形分':>10s} {'胜率':>8s} {'标准误':>8s}")
    for t in cands:
        r = np.array(res[t])
        w = np.array(win[t])
        print(f"打{tile_name(t):4s} {r.mean():10.3f} {w.mean():8.1%} "
              f"{r.std(ddof=1)/np.sqrt(len(r)):8.3f}")
    a, b = cands[0], cands[1]
    diff = np.array(res[a]) - np.array(res[b])   # 共享世界 → 配对差
    se = diff.std(ddof=1) / np.sqrt(len(diff))
    print(f"\n配对差(打{tile_name(a)} − 打{tile_name(b)}): "
          f"{diff.mean():+.3f} ± {1.96*se:.3f} (95% CI)")


if __name__ == "__main__":
    main()
