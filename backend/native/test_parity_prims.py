"""对拍: 原生 shanten/is_win vs backend.rules.win 的 DFS 实现。

覆盖: 手牌张数 1..14(含副露缩短的 10/11 张)、红中 0..4、以及 4 张同牌的极端型。
任何不一致都直接打印手牌, 一致数为 0 才允许启用原生 Bot。
"""

import argparse
import random
import time

import numpy as np

from backend.native import native
from backend.rules.win import is_win as py_is_win, shanten as py_shanten


def rand_hand(rng, size, nred):
    """随机 size 张手牌(含 nred 张红中), 每种普通牌 <=4 张。"""
    nred = min(nred, size)
    pool = []
    for t in range(27):
        pool += [t] * 4
    rng.shuffle(pool)
    c = [0] * 28
    for t in pool[:size - nred]:
        c[t] += 1
    c[27] = nred
    return c


def _minus(hand, t):
    h = list(hand)
    h[t] -= 1
    return h


def check_ukeire_chain(rng, n):
    """进张链对拍: native.score_discards_v10 vs bot_v31 的副露感知参考。

    只比对外层 mj_shanten/mj_is_win 不够: tiles_info→ukeire 这条内部链曾漏掉
    need==0(四副露只差将)特判, 单钓被当成两面/嵌张, 进张集偏大,
    NativeV10/NativeV31 在这些局面选错牌。所以这里必须覆盖短手牌决策态。
    """
    from backend.ai.bot_v31 import _second_step_m, _sh, _ukeire_m

    bad = bad_cont = 0
    worst_cont = 0.0
    for i in range(n):
        total = rng.choice([2, 2, 5, 8, 11, 14])
        n_melds = (14 - total) // 3
        hand = rand_hand(rng, total, rng.choice([0, 0, 1, 2]))
        unseen = [min(rng.randrange(5), 4 - hand[t]) for t in range(28)]
        c = {d["tile"]: d for d in native.score_discards_v10(
            hand, unseen, [0] * 28, 0.0, 100.0, 1.0, 0.5, 0.0, 2)}
        min_sh = min(_sh(tuple(_minus(hand, t)), n_melds)
                     for t in range(28) if hand[t] > 0)
        for t in sorted(c):
            h = tuple(_minus(hand, t))
            s = _sh(h, n_melds)
            u = 0 if s > min_sh else _ukeire_m(h, n_melds, tuple(unseen))
            if c[t]["shanten"] != s or c[t]["ukeire"] != u:
                bad += 1
                if bad <= 5:
                    print("   UKEIRE MISMATCH total", total, "hand",
                          [(k, v) for k, v in enumerate(hand) if v],
                          "打出", t, "native", c[t]["shanten"], c[t]["ukeire"],
                          "py", s, u)
            elif s <= min(min_sh, 2) and i % 20 == 0:
                py_cont = _second_step_m(h, n_melds, tuple(unseen))
                diff = abs(py_cont - c[t]["cont"])
                worst_cont = max(worst_cont, diff)
                if diff > 1e-6 * max(1.0, abs(py_cont)):
                    bad_cont += 1
                    if bad_cont <= 3:
                        print("   CONT MISMATCH total", total, "打出", t,
                              "native", c[t]["cont"], "py", py_cont)
    print(f"进张链对拍 {n} 手: shanten/ukeire 不一致 {bad}, cont 不一致 {bad_cont} "
          f"(cont 最大绝对差 {worst_cont:.2e})")
    return bad == 0 and bad_cont == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60000)
    ap.add_argument("--seed", type=int, default=20260829)
    args = ap.parse_args()

    native.lib()
    rng = random.Random(args.seed)
    hands = []
    # 向听: 任意张数; 判胡: 只对 3n+2 张有意义(其它张数两边都应返回 False)
    for _ in range(args.n):
        size = rng.choice([13, 13, 13, 14, 14, 11, 10, 8, 7, 5, 4, 2, 1])
        nred = rng.choice([0, 0, 0, 1, 1, 2, 3, 4])
        hands.append(rand_hand(rng, size, nred))
    # 补充极端型: 同牌 4 张 + 红中, 但总张数必须 <=14(合法手牌上限, 见 mj159.c 注释)
    for _ in range(args.n // 20):
        c = [0] * 28
        k = rng.randint(1, 3)
        picked = rng.sample(range(27), k)
        for t in picked:
            c[t] = 4
        c[27] = min(rng.randrange(5), 14 - 4 * k)
        hands.append(c)
    hands.append([0] * 28)
    hands.append([0] * 27 + [4])

    H = np.array(hands, dtype=np.int8)
    t0 = time.process_time()
    nsh = native.shanten_batch(H)
    nwin = native.is_win_batch(H)
    t_nat = time.process_time() - t0

    t0 = time.process_time()
    psh = np.array([py_shanten(list(map(int, h))) for h in hands],
                   dtype=np.int32)
    t_py_sh = time.process_time() - t0
    t0 = time.process_time()
    pwin = np.array([1 if py_is_win(list(map(int, h))) else 0 for h in hands],
                    dtype=np.int32)
    t_py_win = time.process_time() - t0

    bad_sh = np.nonzero(nsh != psh)[0]
    bad_win = np.nonzero(nwin != pwin)[0]
    n = len(hands)
    print(f"样本 {n} 手")
    print(f"  shanten 不一致 {len(bad_sh)}")
    print(f"  is_win  不一致 {len(bad_win)}")
    for i in bad_sh[:5]:
        print("   SH MISMATCH native", int(nsh[i]), "py", int(psh[i]),
              [(t, int(v)) for t, v in enumerate(hands[i]) if v])
    for i in bad_win[:5]:
        print("   WIN MISMATCH native", int(nwin[i]), "py", int(pwin[i]),
              [(t, int(v)) for t, v in enumerate(hands[i]) if v])
    print(f"  cpu: native(sh+win) {t_nat:.2f}s | py shanten {t_py_sh:.2f}s | "
          f"py is_win {t_py_win:.2f}s")
    print(f"  加速比: shanten ~{t_py_sh/max(t_nat,1e-9):.0f}x (原生含判胡, 偏保守)")
    ok_chain = check_ukeire_chain(rng, max(500, args.n // 30))
    ok = len(bad_sh) == 0 and len(bad_win) == 0 and ok_chain
    print("PARITY", "OK" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
