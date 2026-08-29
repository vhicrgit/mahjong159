"""961000 第4巡的完整仲裁, 回答两个问题:

1. 打5饼(v31) / 打7条(五搭子拆孤张) / 打8饼或9饼(如果 89饼 是死搭就拆掉它)
   在普通共享世界里谁最优
2. 把"座3 持有 7饼7饼7饼(暗刻)"这一真实信息钉进世界采样后, 结论怎么变 —
   即"对手手牌知识"在这一手值多少分

pin 机制: 从可采样池里先移除被钉的牌发给指定座位, 其余照旧随机。
"""

import argparse
import copy
import multiprocessing as mp
import random

import numpy as np

from backend.ai.bot_native import NativeV31
from backend.native import native
from backend.rl.gen_offline import _shaped_scores
from backend.rl.world_grpo import sample_world
from backend.rules.tiles import tile_name

from tools.perf.arbitrate_961000 import replay_to


def sample_world_pinned(snap, rng, hero_seat, pin):
    """同 sample_world, 但 pin={seat: [tile,...]} 的牌先发给它(从池里移除)。"""
    unseen = []
    visible = [0] * 28
    for q in snap.players:
        for t in q.discards:
            visible[t] += 1
        for m in q.melds:
            visible[m["tile"]] += 3 if m["type"] == "peng" else 4
    me = snap.players[hero_seat]
    for t, c in enumerate(me.hand_counts):
        visible[t] += c
    for t in range(28):
        unseen += [t] * (4 - visible[t])
    for seat, tiles in pin.items():
        for t in tiles:
            unseen.remove(t)          # 若池里没有会直接 ValueError
    rng.shuffle(unseen)
    pos = 0
    hands = {}
    for q in snap.players:
        if q.seat == hero_seat:
            continue
        need = len(q.hand)
        pinned = list(pin.get(q.seat, []))
        hands[q.seat] = sorted(pinned + unseen[pos:pos + need - len(pinned)])
        pos += need - len(pinned)
    wall = unseen[pos:]
    return hands, wall


def rollout_world(args):
    snap, seat, tile, wseed, pin = args
    hands, wall = sample_world_pinned(snap, random.Random(wseed), seat, pin)
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


def run_block(g, seat, cands, worlds, procs, pin, seed_base):
    tasks = [(copy.deepcopy(g), seat, t, seed_base + w, pin)
             for t in cands for w in range(worlds)]
    with mp.Pool(procs) as pool:
        rets = pool.map(rollout_world, tasks, chunksize=64)
    res = {t: [] for t in cands}
    win = {t: [] for t in cands}
    for (_, _, t, _, _), (r, w) in zip(tasks, rets):
        res[t].append(r)
        win[t].append(w)
    return res, win


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=961000)
    ap.add_argument("--turn", type=int, default=4)
    ap.add_argument("--worlds", type=int, default=2048)
    ap.add_argument("--procs", type=int, default=6)
    args = ap.parse_args()

    g = replay_to(args.seed, 0, args.turn)
    seat = 0
    # 候选: 5饼(13), 7条(6), 8饼(16), 9饼(17), 6万(23)
    cands = [13, 6, 16, 17, 23]
    names = {13: "打5饼(v31)", 6: "打7条(你)", 16: "打8饼(拆89边)",
             17: "打9饼(拆89边)", 23: "打6万"}

    for label, pin in (("普通世界(不知对手手牌)", {}),
                       ("钉住座3持 7饼×3(真实信息)", {3: [15, 15, 15]})):
        res, win = run_block(g, seat, cands, args.worlds, args.procs, pin,
                             310000)
        print(f"\n=== {label}, {args.worlds} 世界/候选 ===")
        ref = np.array(res[13])
        print(f"{'候选':>18s} {'期望分':>8s} {'胜率':>7s} "
              f"{'与打5饼的配对差':>18s}")
        for t in cands:
            r = np.array(res[t])
            d = r - ref
            se = d.std(ddof=1) / np.sqrt(len(d))
            print(f"{names[t]:>18s} {r.mean():8.3f} "
                  f"{np.mean(win[t]):7.1%}   {d.mean():+.3f} ± "
                  f"{1.96*se:.3f}")


if __name__ == "__main__":
    main()
