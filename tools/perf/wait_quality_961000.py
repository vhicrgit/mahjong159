"""验证"红中绑定"论点: 961000 第4巡, 打5饼 vs 打7条。

对每种打法的每张有效牌 x:
  摸进 x -> 按 v31 口径选最优弃牌(听牌时取听口最宽的) ->
  记录: 到达的听口宽度(加权重) + 红中是否被烧掉(从手牌消失)

如果论点成立: 打5饼 的条子进张(5-9条)烧掉红中、听口很窄;
而 打7条 的饼子进张(3-7饼)不烧红中、听口宽。
"""

import argparse

from backend.game.engine import Game
from backend.ai.bot_native import NativeV31
from backend.native import native
from backend.rules.tiles import tile_name

from tools.perf.arbitrate_961000 import replay_to


def best_tenpai_after_draw(hand13, draw, unseen):
    """摸进 draw 后的最优处理: 枚举打出每张, 在向听最小的结果里取听口最宽。
    返回 (打出的牌, 打后向听, 听口宽度, 红中是否还在手里)。"""
    h14 = list(hand13)
    h14[draw] += 1
    best = None
    for d in range(28):
        if h14[d] <= 0:
            continue
        h = list(h14)
        h[d] -= 1
        s = native.shanten(h)
        if s != 0:
            # 非听牌: 用进张数衡量(广义)
            u = 0
            for t in range(28):
                if h[t] >= 4:
                    continue
                h2 = list(h)
                h2[t] += 1
                if native.shanten(h2) < s:
                    u += unseen[t]
            key = (s, -u)
        else:
            w = 0
            for t in range(28):
                if h[t] >= 4:
                    continue
                h2 = list(h)
                h2[t] += 1
                if native.is_win(h2):
                    w += unseen[t]
            key = (s, -w)
        if best is None or key < best[0]:
            best = (key, d, s, (w if s == 0 else u), h)
    _, d, s, width, h = best
    return d, s, width, h[27] > 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=961000)
    ap.add_argument("--turn", type=int, default=4)
    args = ap.parse_args()

    g = replay_to(args.seed, 0, args.turn)
    p = g.players[0]
    hand = list(p.hand_counts)
    visible = [0] * 28
    for q in g.players:
        for t in q.discards:
            visible[t] += 1
        for m in q.melds:
            visible[m["tile"]] += 3 if m["type"] == "peng" else 4
    for t, n in enumerate(hand):
        visible[t] += n
    unseen = [max(0, 4 - v) for v in visible]
    total_unseen = sum(unseen)

    for cand, label in ((13, "打5饼(v31)"), (6, "打7条(你)")):
        h13 = list(hand)
        h13[cand] -= 1
        s = native.shanten(h13)
        print(f"\n=== {label}: 打后向听 {s} ===")
        print(f"{'摸':>5s} {'剩余':>4s} {'打出':>5s} {'后向听':>5s} "
              f"{'听口/进张宽':>8s} {'红中':>4s}")
        rows = []
        for x in range(28):
            if h13[x] >= 4 or unseen[x] <= 0:
                continue
            h = list(h13)
            h[x] += 1
            improves = (native.is_win(h) if s == 0
                        else native.shanten(h) < s)
            if not improves:
                continue
            d, s2, width, red_alive = best_tenpai_after_draw(h13, x, unseen)
            rows.append((x, unseen[x], d, s2, width, red_alive))
            print(f"{tile_name(x):>5s} {unseen[x]:4d} {tile_name(d):>5s} "
                  f"{s2:5d} {width:8d} {'在' if red_alive else '烧掉':>4s}")
        # 加权的期望听口宽度(到听牌/改善后的形态质量)
        num = sum(unseen[x] * w for x, un, d, s2, w, r in rows)
        den = sum(unseen[x] for x, *_ in rows)
        burn = sum(unseen[x] for x, un, d, s2, w, r in rows if not r)
        print(f"加权平均宽度 {num/den:.1f} 张 | 烧掉红中的进张占 "
              f"{burn}/{den} = {burn/den:.0%}")


if __name__ == "__main__":
    main()
