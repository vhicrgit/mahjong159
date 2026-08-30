"""标定探针: 学者的真实决策 vs v10 打分替代模型的一致率。

收集(全部从学者本人视角计算, 复刻其对局中的信息集):
  1) 出牌: 学者实际打的牌在 v10 score_discards_v10 降序中的名次、与榜首的分差
     -> 拟合软似然 P(学者打d | v10名次/分差)
  2) 碰: 学者 decide_peng(E口径) vs 代理规则"碰后最优向听严格下降" 的混淆矩阵
  3) 杠(明杠): 学者 decide_gang 固定口径(不破坏听口), 天然精确, 只统计触发率

用法: python -m tools.perf.probe_scholar_policy --games 40
"""

import argparse
from collections import Counter

import numpy as np

from backend.ai.bot_hv import Bot as HVBot
from backend.ai.bot_native import NativeV31
from backend.game.engine import Game
from backend.native import native

RED = 27


def scholar_view(g, seat):
    """学者 seat 的 unseen(=4-可见) / penged(别人碰过的) / eg。"""
    visible = [0] * 28
    for q in g.players:
        for t in q.discards:
            visible[t] += 1
        for m in q.melds:
            visible[m["tile"]] += 3 if m["type"] == "peng" else 4
    for t, n in enumerate(g.players[seat].hand_counts):
        visible[t] += n
    unseen = [max(0, 4 - v) for v in visible]
    penged = [0] * 28
    for q in g.players:
        if q.seat == seat:
            continue
        for m in q.melds:
            if m["type"] == "peng":
                penged[m["tile"]] = 1
    eg = max(0.0, min(1.0, (60 - g.wall_remaining()) / 60.0))
    return unseen, penged, eg


def peng_proxy(counts, tile):
    """代理: 碰后(打一张最优)向听严格下降。"""
    before = native.shanten(counts)
    h2 = list(counts)
    h2[tile] -= 2
    best = 99
    for d in range(28):
        if h2[d] <= 0:
            continue
        h2[d] -= 1
        best = min(best, native.shanten(h2))
        h2[d] += 1
    return best < before


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=40)
    ap.add_argument("--seed0", type=int, default=985000)
    args = ap.parse_args()

    rank_cnt = Counter()          # 学者实际选择的 v10 名次(0=榜首)
    gap_bins = Counter()          # 分差桶
    n_dec = 0
    peng_cm = Counter()           # (proxy, actual)
    gang_cnt = Counter()

    for gi in range(args.games):
        g = Game(seed=args.seed0 + gi, human_seat=-1)
        bots = {0: NativeV31(g, 0)}
        for i in range(1, 4):
            bots[i] = HVBot(g, i, memo={})
        guard = 0
        while g.phase != "game_over" and guard < 500:
            guard += 1
            if g.phase == "discard_wait":
                s = g.turn
                if s != 0:
                    hand = g.players[s].hand_counts
                    unseen, penged, eg = scholar_view(g, s)
                    rows = native.score_discards_v10(hand, unseen, penged, eg)
                    actual = bots[s].choose_discard()
                    order = sorted(rows, key=lambda r: -r["score"])
                    rank = next(i for i, r in enumerate(order)
                                if r["tile"] == actual)
                    gap = order[0]["score"] - next(
                        r["score"] for r in rows if r["tile"] == actual)
                    rank_cnt[min(rank, 8)] += 1
                    gap_bins[min(int(gap // 2) * 2, 20)] += 1
                    n_dec += 1
                    d = actual
                else:
                    d = bots[s].choose_discard()
                g.action_discard(s, d)
            else:
                s = list(g.pending_actions.keys())[0]
                pend = g.pending_actions[s]
                tile = g.last_discard
                b = bots[s]
                if s != 0 and pend.get("peng"):
                    actual = b.decide_peng(tile)
                    proxy = peng_proxy(g.players[s].hand_counts, tile)
                    peng_cm[(proxy, actual)] += 1
                    if pend.get("gang"):
                        gang_cnt[b.decide_gang(tile, "ming")] += 1
                    if pend.get("gang") and b.decide_gang(tile, "ming"):
                        g.action_gang(s)
                    elif actual:
                        g.action_peng(s)
                    else:
                        g.action_pass(s)
                else:
                    if pend.get("gang") and b.decide_gang(tile, "ming"):
                        g.action_gang(s)
                    elif pend.get("peng") and b.decide_peng(tile):
                        g.action_peng(s)
                    else:
                        g.action_pass(s)

    tot = sum(rank_cnt.values())
    print(f"出牌决策 n={tot}")
    acc = 0
    for r in sorted(rank_cnt):
        acc += rank_cnt[r]
        lab = f"top{r + 1}" if r < 8 else "top9+"
        print(f"  v10名次 {r}: {rank_cnt[r]:5d} ({rank_cnt[r] / tot:6.2%})"
              f"  累计 {acc / tot:6.2%}")
    print("分差桶(榜首-实际): ",
          {k: round(v / tot, 4) for k, v in sorted(gap_bins.items())})
    tp = peng_cm[(True, True)]
    fp = peng_cm[(True, False)]
    fn = peng_cm[(False, True)]
    tn = peng_cm[(False, False)]
    n_p = sum(peng_cm.values())
    print(f"\n碰决策 n={n_p}: 代理&实际一致率 "
          f"{(tp + tn) / max(1, n_p):.2%}  (碰且代理碰 {tp}, "
          f"碰但代理不碰 {fn}, 不碰但代理碰 {fp}, 双否 {tn})")
    print(f"明杠判定: {dict(gang_cnt)}")


if __name__ == "__main__":
    main()
