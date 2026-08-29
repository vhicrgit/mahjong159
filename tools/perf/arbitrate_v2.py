"""仲裁器 v2: 接续策略可换。

--hero-bot v31n        原口径: 候选之后由 v31 继续(受 bot 能力偏置)
--hero-bot cheat_fulln  全信息接续: 后续每一步由知道牌墙+对手手牌的 cheat bot
                        打出(近最优), 候选价值不再被 v31 的弱点污染

用法:
  python -m tools.perf.arbitrate_v2 --hero-bot cheat_fulln --worlds 512
"""

import argparse
import copy
import multiprocessing as mp
import random

import numpy as np

from backend.rl.gen_offline import _shaped_scores
from backend.rl.grpo_train import _BOT_REGISTRY
from backend.rules.tiles import tile_name
from tools.perf.arbitrate_961000 import replay_to
from tools.perf.arbitrate_961000_full import sample_world_pinned


def rollout_world(args):
    snap, seat, tile, wseed, pin, hero_bot, opp_bot = args
    hands, wall = sample_world_pinned(snap, random.Random(wseed), seat, pin)
    g = copy.deepcopy(snap)
    for s2, h in hands.items():
        g.players[s2].hand = list(h)
    g.wall = list(wall)
    hcfg = _BOT_REGISTRY[hero_bot]
    ocfg = _BOT_REGISTRY[opp_bot]
    hero = hcfg["cls"](g, seat, **hcfg["kwargs"])
    opps = {i: ocfg["cls"](g, i, **ocfg["kwargs"]) for i in range(4)
            if i != seat}
    g.action_discard(seat, tile)
    guard = 0
    while g.phase != "game_over" and guard < 500:
        guard += 1
        if g.phase == "discard_wait":
            b = hero if g.turn == seat else opps[g.turn]
            g.action_discard(g.turn, b.choose_discard())
        else:
            s = list(g.pending_actions.keys())[0]
            b = hero if s == seat else opps[s]
            if g.pending_actions[s].get("gang") and \
                    b.decide_gang(g.last_discard, "ming"):
                g.action_gang(s)
            elif g.pending_actions[s].get("peng") and \
                    b.decide_peng(g.last_discard):
                g.action_peng(s)
            else:
                g.action_pass(s)
    return float(_shaped_scores(g, 0.02)[seat]), 1 if g.winner == seat else 0


def run_block(g, seat, cands, worlds, procs, pin, hero_bot, opp_bot,
              seed_base):
    tasks = [(copy.deepcopy(g), seat, t, seed_base + w, pin, hero_bot,
              opp_bot) for t in cands for w in range(worlds)]
    with mp.Pool(procs) as pool:
        rets = pool.map(rollout_world, tasks, chunksize=8)
    res = {t: [] for t in cands}
    win = {t: [] for t in cands}
    for (_, _, t, _, _, _, _), (r, w) in zip(tasks, rets):
        res[t].append(r)
        win[t].append(w)
    return res, win


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=961000)
    ap.add_argument("--turn", type=int, default=4)
    ap.add_argument("--worlds", type=int, default=512)
    ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--hero-bot", default="cheat_fulln")
    ap.add_argument("--opp-bot", default="v31n")
    ap.add_argument("--cands", default="13,6,16,17")
    args = ap.parse_args()

    g = replay_to(args.seed, 0, args.turn)
    seat = 0
    cands = [int(x) for x in args.cands.split(",")]
    names = {13: "打5饼(v31原选)", 6: "打7条(你的主张)",
             16: "打8饼(拆89边)", 17: "打9饼(拆89边)", 23: "打6万"}
    print(f"接续: hero={args.hero_bot} 对手={args.opp_bot}")

    for label, pin in (("普通世界", {}),
                       ("钉住座3持7饼×3", {3: [15, 15, 15]})):
        res, win = run_block(g, seat, cands, args.worlds, args.procs, pin,
                             args.hero_bot, args.opp_bot, 310000)
        print(f"\n=== {label}, {args.worlds} 世界/候选 ===")
        ref = np.array(res[cands[0]])
        print(f"{'候选':>18s} {'期望分':>8s} {'胜率':>7s}  {'与第1候选的配对差':>18s}")
        for t in cands:
            r = np.array(res[t])
            d = r - ref
            se = d.std(ddof=1) / np.sqrt(len(d))
            print(f"{names.get(t, tile_name(t)):>18s} {r.mean():8.3f} "
                  f"{np.mean(win[t]):7.1%}   {d.mean():+.3f} ± "
                  f"{1.96*se:.3f}", flush=True)


if __name__ == "__main__":
    main()
