"""检验"推演 bot 自己破坏策略意图"的问题。

961000 第4巡: A=打5饼(v31 本意是弃7条留445饼? 不, A 是打5饼留7条),
B=打7条(用户本意: 保住 445饼 复合形)。

问题: 打完候选牌后, 由 v31n 继续打。如果 B 线里 v31 下一巡把 5饼/4饼 打掉,
那"保 445饼"的意图就没被执行, rollout 对比的就是别的东西。

方法: 每个共享世界里, 走完候选弃牌 -> 其他三家反应 -> 正常推进到 hero 的
下一次出牌决策, 记录 v31n 实际打了什么。统计两条线的下一巡弃牌分布。
"""

import argparse
import copy
import multiprocessing as mp
import random
from collections import Counter

from backend.ai.bot_native import NativeV31
from backend.rl.world_grpo import sample_world
from backend.rules.tiles import tile_name

from tools.perf.arbitrate_961000 import replay_to


def next_hero_discard(args):
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
            if g.turn == seat:
                # hero 的第二次决策: 记录它打了什么
                drawn = g.last_drawn["tile"] if g.last_drawn else None
                return tile_name(drawn) if drawn is not None else "?", \
                    tile_name(bots[seat].choose_discard())
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
    return None, None     # 没轮到 hero 再出牌就结束了


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=961000)
    ap.add_argument("--turn", type=int, default=4)
    ap.add_argument("--worlds", type=int, default=512)
    ap.add_argument("--procs", type=int, default=6)
    args = ap.parse_args()

    g = replay_to(args.seed, 0, args.turn)
    seat = 0
    for cand, label, watch in ((13, "A: 打5饼(留7条)", {"7条"}),
                               (6, "打7条(保445饼)", {"4饼", "5饼"})):
        tasks = [(copy.deepcopy(g), seat, cand, 310000 + w)
                 for w in range(args.worlds)]
        with mp.Pool(args.procs) as pool:
            rets = pool.map(next_hero_discard, tasks, chunksize=32)
        draws = Counter(d for d, _ in rets if d)
        discs = Counter(x for _, x in rets if x)
        n = sum(discs.values())
        broke = sum(discs[t] for t in watch)
        print(f"\n=== {label} —— {n}/{args.worlds} 个世界里轮到了下一次出牌 ===")
        print(f"  摸到的牌: {dict(draws.most_common(8))}")
        print(f"  v31 下一次实际打出: {dict(discs.most_common(10))}")
        print(f"  其中打掉 {sorted(watch)}(破坏你关注的结构) 的比例: "
              f"{broke}/{n} = {broke/max(n,1):.1%}")


if __name__ == "__main__":
    main()
