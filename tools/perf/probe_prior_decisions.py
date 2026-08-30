"""诊断: 对手先验在多大概率上改变学者的出牌决策?

先验臂打法(eval_hv_prior 同款), 但每个座0 决策点同时算:
  - 均匀假设下的选择(学者原版)
  - 对手先验下的选择
报告一致率, 以及不一致时的 E 差。用来解释胜率 A/B 的零结果:
若决策几乎不变, 则零效果是结构性的(E 排序对墙组成扰动鲁棒)。

用法: python -m tools.perf.probe_prior_decisions --games 8 --procs 8
"""

import argparse
import multiprocessing as mp

import numpy as np

from backend.ai import bot_hv
from backend.ai.bot_hv import Bot as HVBot
from backend.analysis.opp_model import OppTracker
from backend.game.engine import Game

HERO = 0


def play(seed, beam, n_init):
    g = Game(seed=seed, human_seat=-1)
    trackers = {s: OppTracker(s, g.players[HERO].hand_counts,
                              n_init=n_init, beam=beam, policy=True,
                              seed=seed * 10 + s)
                for s in (1, 2, 3)}
    for tr in trackers.values():
        tr.notify_deal(g.dealer)
    memo = {}
    bots = {HERO: HVBot(g, HERO, memo=memo)}
    for i in range(1, 4):
        bots[i] = HVBot(g, i, memo={})

    def visible_hero():
        visible = [0] * 28
        for q in g.players:
            for t in q.discards:
                visible[t] += 1
            for m in q.melds:
                visible[m["tile"]] += 3 if m["type"] == "peng" else 4
        for t, n in enumerate(g.players[HERO].hand_counts):
            visible[t] += n
        return visible

    recs = []
    guard = 0
    while g.phase != "game_over" and guard < 500:
        guard += 1
        if g.phase == "discard_wait":
            s = g.turn
            if s == HERO:
                hand = g.players[HERO].hand_counts
                vis = visible_hero()
                held = [0.0] * 28
                for tr in trackers.values():
                    ec = tr.expected_counts()
                    for t in range(28):
                        held[t] += ec[t]
                u_eff = [max(0.0, 4 - vis[t] - held[t]) for t in range(28)]
                d_uni = bot_hv.choose_discard(hand, vis, 1.0, memo)
                d_pri = bot_hv.choose_discard(hand, vis, 1.0, memo,
                                              u_eff=u_eff, held_exp=held)
                recs.append((d_uni == d_pri))
                d = d_pri
            else:
                d = bots[s].choose_discard()
            for tr in trackers.values():
                tr.notify_discard(s, d, g.wall_remaining())
            ev = g.action_discard(s, d)
        else:
            s = list(g.pending_actions.keys())[0]
            pend = g.pending_actions[s]
            tile, discarder = g.last_discard, g.last_discarder
            b = bots[s]
            if pend.get("gang") and b.decide_gang(tile, "ming"):
                act = "gang"
            elif pend.get("peng") and b.decide_peng(tile):
                act = "peng"
            else:
                act = None
            for tr in trackers.values():
                tr.notify_claim(s, act, tile, discarder)
            if act == "gang":
                ev = g.action_gang(s)
            elif act == "peng":
                ev = g.action_peng(s)
            else:
                ev = g.action_pass(s)
        if ev.get("event") in ("draw", "gang_draw"):
            for tr in trackers.values():
                tr.notify_draw(ev["seat"],
                               ev["tile"] if ev["seat"] == HERO else None,
                               g.wall_remaining())
    return recs


def work(payload):
    seed, beam, n_init = payload
    return play(seed, beam, n_init)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=8)
    ap.add_argument("--seed0", type=int, default=997000)
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--beam", type=int, default=600)
    ap.add_argument("--n-init", type=int, default=3000)
    args = ap.parse_args()

    seeds = [(args.seed0 + i, args.beam, args.n_init)
             for i in range(args.games)]
    if args.procs > 1:
        with mp.Pool(args.procs) as pool:
            rets = pool.map(work, seeds, chunksize=1)
    else:
        rets = [work(s) for s in seeds]
    recs = [r for one in rets for r in one]
    agree = np.mean(recs)
    print(f"座0 出牌决策 n={len(recs)}: 先验与均匀一致率 {agree:.1%}, "
          f"改变率 {1 - agree:.1%}")


if __name__ == "__main__":
    main()
