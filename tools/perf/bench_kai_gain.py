"""换型深度/宽度的收益标定: 两组 (kai_max, kai_topk) 的推荐弃牌是否改变、E 差多少。

规格写作 "kai:topk", 例如:
  python -m tools.perf.bench_kai_gain --pair 2:6,3:6     # 加一层深度值不值
  python -m tools.perf.bench_kai_gain --pair 3:6,3:3     # 深度衰减剪枝的失真
"""

import argparse
import random
import time

from backend.analysis import hv_native
from backend.rules.tiles import TILE_COUNT, build_wall, tile_name


def rand_hand(rng, ntile):
    wall = build_wall()
    rng.shuffle(wall)
    hand = [0] * TILE_COUNT
    for t in wall[:ntile]:
        hand[t] += 1
    return hand, list(hand)


def table(hand, vis, kai, topk):
    es = {}
    for t in range(TILE_COUNT):
        if hand[t]:
            hv_native.set_hand(hand, vis, 1.0, kai > 0, 2, kai, topk)
            es[t] = hv_native.e_after_discard(t)
    return es, min(es, key=lambda t: (es[t], t))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hands", type=int, default=10)
    ap.add_argument("--ntile", type=int, default=11)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--pair", default="2:6,3:6")
    args = ap.parse_args()
    a, b = [tuple(int(v) for v in s.split(":"))
            for s in args.pair.split(",")]

    flips = 0
    de_sum = de_max = tie_sum = 0.0
    ta = tb = 0.0
    for i in range(args.hands):
        hand, vis = rand_hand(random.Random(args.seed * 1000 + i), args.ntile)
        t0 = time.perf_counter()
        ea, ba = table(hand, vis, *a)
        t1 = time.perf_counter()
        eb, bb = table(hand, vis, *b)
        t2 = time.perf_counter()
        ta += t1 - t0
        tb += t2 - t1
        d = abs(eb[bb] - ea[ba])
        de_sum += d
        de_max = max(de_max, d)
        if ba != bb:
            flips += 1
            # 用更"深"的一侧(b)当裁判, 看被 a 选中的那张差多少
            tie_sum += eb[ba] - eb[bb]
            print(f"  #{i} 翻转: {a}->{tile_name(ba)}  {b}->{tile_name(bb)}"
                  f"  |  在 {b} 口径下 {eb[ba]:.3f} vs {eb[bb]:.3f}"
                  f" (亏 {eb[ba] - eb[bb]:.3f} 巡)")
    n = args.hands
    print(f"n={n} ntile={args.ntile}  翻转 {flips}/{n} ({flips / n:.1%})  "
          f"翻转局平均亏 {tie_sum / flips if flips else 0:.3f} 巡  "
          f"|ΔE|均 {de_sum / n:.4f} 最大 {de_max:.4f}")
    print(f"耗时 {a} {ta / n:.3f}s/手   {b} {tb / n:.3f}s/手")


if __name__ == "__main__":
    main()
