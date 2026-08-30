"""换型档位 kai_max 的代价标定: 整表耗时 / DP 状态数 / 缓存命中。

用法:
  python -m tools.perf.bench_kai_depth [--hands N] [--kai 0,1,2,3] [--topk 6]
"""

import argparse
import ctypes
import random
import time

from backend.analysis import hv_native
from backend.rules.tiles import TILE_COUNT, build_wall


def rand_hand(rng, ntile=11):
    """从洗好的牌墙里发前 ntile 张, 返回 (hand28, visible28)。"""
    wall = build_wall()
    rng.shuffle(wall)
    hand = [0] * TILE_COUNT
    for t in wall[:ntile]:
        hand[t] += 1
    vis = list(hand)
    return hand, vis


def stats(L, kind):
    out = (ctypes.c_uint64 * 5)()
    fn = L.mj_hv_stats if kind == "mj159" else L.mjc_hv_stats
    fn.argtypes = [ctypes.POINTER(ctypes.c_uint64)]
    fn(out)
    return list(out)


def stats_reset(L, kind):
    (L.mj_hv_stats_reset if kind == "mj159" else L.mjc_hv_stats_reset)()


def load_alt(path):
    """把另一份 libmj159.so(比如 -DHV_E_BITS=24 编的)塞进 hv_native 的后端槽。"""
    import os

    from backend.native import native
    L = ctypes.CDLL(path)
    L.mj_init.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
    L.mj_init.restype = ctypes.c_int
    root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(native.__file__))))
    rc = L.mj_init(os.path.join(root, "models", "mj_front.bin").encode(),
                   os.path.join(root, "models", "mj_win.bin").encode())
    if rc != 0:
        raise RuntimeError(f"mj_init rc={rc}")
    i8p = ctypes.POINTER(ctypes.c_int8)
    L.mj_hv_set2.argtypes = [i8p, i8p, ctypes.c_double, ctypes.c_int,
                             ctypes.c_int, ctypes.c_int, ctypes.c_int]
    L.mj_hv_set2.restype = ctypes.c_int
    L.mj_hv_e_after_discard.argtypes = [ctypes.c_int]
    L.mj_hv_e_after_discard.restype = ctypes.c_double
    hv_native._STATE.update(lib=L, kind="mj159", tried=True)
    return L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hands", type=int, default=20)
    ap.add_argument("--kai", default="0,1,2,3")
    ap.add_argument("--topk", type=int, default=6)
    ap.add_argument("--ntile", type=int, default=11)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--so", help="替换后端 .so 路径(扫 E memo 容量用)")
    args = ap.parse_args()

    L = load_alt(args.so) if args.so else hv_native.lib()
    kind = hv_native.backend_kind()
    print(f"backend={kind}  topk={args.topk}  ntile={args.ntile}")
    hands = [rand_hand(random.Random(args.seed * 1000 + i), args.ntile)
             for i in range(args.hands)]

    for kai in [int(x) for x in args.kai.split(",")]:
        tot, worst, worst_i = 0.0, 0.0, -1
        miss = hit = 0
        for i, (hand, vis) in enumerate(hands):
            stats_reset(L, kind)
            t0 = time.perf_counter()
            for t in range(TILE_COUNT):
                if hand[t]:
                    hv_native.set_hand(hand, vis, 1.0, kai > 0, 2,
                                       kai, args.topk)
                    hv_native.e_after_discard(t)
            dt = time.perf_counter() - t0
            s = stats(L, kind)
            miss += s[0]
            hit += s[1]
            tot += dt
            if dt > worst:
                worst, worst_i = dt, i
        n = len(hands)
        print(f"kai_max={kai}: 均 {tot / n:7.3f}s  最坏 {worst:7.3f}s"
              f"(#{worst_i})  DP状态/手 {miss // n:>9}  命中率 "
              f"{hit / max(1, hit + miss):.3f}")


if __name__ == "__main__":
    main()
