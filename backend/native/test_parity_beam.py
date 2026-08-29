"""对拍: 原生 beam search vs bot_oracle.search_first_discard_detail。

detail 字典必须完全相同 —— 它决定 cheat_full/oracle 的出牌排序,
差一个 win_depth 就换一张牌。
"""

import argparse
import random
import time

from backend.ai.bot_oracle import search_first_discard_detail as py_beam
from backend.native import native


def rand_case(rng):
    """随机 14 张手牌 + 随机 future_draws(模拟牌墙里自己的摸牌序列)。"""
    pool = [t for t in range(27) for _ in range(4)] + [27] * 4
    rng.shuffle(pool)
    nred = rng.choice([0, 0, 0, 1, 1, 2])
    c = [0] * 28
    used = []
    for t in pool:
        if len(used) >= 14 - nred:
            break
        if t == 27:
            continue
        if c[t] < 4:
            c[t] += 1
            used.append(t)
    c[27] = nred
    horizon = rng.choice([0, 1, 3, 6, 9, 12, 14, 18])
    future = [rng.randrange(28) for _ in range(horizon)]
    return c, future


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--beam", type=int, default=18)
    ap.add_argument("--seed", type=int, default=4242)
    args = ap.parse_args()

    native.lib()
    rng = random.Random(args.seed)
    cases = [rand_case(rng) for _ in range(args.n)]

    t0 = time.process_time()
    nat = [native.beam_detail(c, f, args.beam) for c, f in cases]
    t_nat = time.process_time() - t0

    t0 = time.process_time()
    pyr = [py_beam(list(c), list(f), args.beam) for c, f in cases]
    t_py = time.process_time() - t0

    bad = 0
    for i, (a, b) in enumerate(zip(nat, pyr)):
        if a != b:
            bad += 1
            if bad <= 3:
                only = {k: (a.get(k), b.get(k)) for k in set(a) | set(b)
                        if a.get(k) != b.get(k)}
                print(f"  MISMATCH case {i} horizon={len(cases[i][1])}: {only}")
    print(f"样本 {args.n} 个局面 (beam={args.beam}): detail 不一致 {bad}")
    print(f"  cpu: native {t_nat:.3f}s vs py {t_py:.2f}s "
          f"-> {t_py/max(t_nat,1e-9):.0f}x "
          f"({t_nat/args.n*1000:.2f} vs {t_py/args.n*1000:.1f} ms/次)")
    print("PARITY", "OK" if bad == 0 else "FAIL")
    raise SystemExit(0 if bad == 0 else 1)


if __name__ == "__main__":
    main()
