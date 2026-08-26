"""红中向听的分花色查表法验证 (MahJax 移植的核心算法 PoC)

思路: 手牌按花色拆成 3 个 9 位 base-5 编码 + 红中数。
每花色独立计算 (m面子, t搭子, p将, 用红数) 的 Pareto 前沿,
再枚举红中在三个花色间的分配并合并前沿, 剩余红中按 divmod 归并。
若与 win.py 的整体 DFS 全等 → 查表法可移植 JAX (MahJax 同款路径)。
"""

import random
import sys
from functools import lru_cache

sys.path.insert(0, ".")

from backend.rules.win import shanten


@lru_cache(maxsize=None)
def suit_front(code: int, red_avail: int) -> tuple:
    """单花色(9种牌)在给定可用红中数下的 Pareto 前沿。
    返回 ((m, t, p, r_used), ...)。code: base-5 计数编码。"""
    c = []
    x = code
    for _ in range(9):
        c.append(x % 5)
        x //= 5
    counts = tuple(c)
    return _suit_dfs(counts, red_avail)


@lru_cache(maxsize=None)
def _suit_dfs(counts: tuple, red: int) -> tuple:
    # 找最小非零下标
    t = -1
    for i in range(9):
        if counts[i] > 0:
            t = i
            break
    if t == -1:
        # 本花色无牌: 不用红
        return ((0, 0, 0, 0),)

    cands = set()

    def add(sub, dm, dt, dp, dr):
        for m, tt, p, r in sub:
            cands.add((m + dm, tt + dt, p + dp, r + dr))

    def mod(i, d):
        l = list(counts)
        l[i] += d
        return tuple(l)

    # 孤张跳过
    add(_suit_dfs(mod(t, -1), red), 0, 0, 0, 0)
    # 对子: 将 / 搭子
    if counts[t] >= 2:
        sub = _suit_dfs(mod(t, -2), red)
        add(sub, 0, 0, 1, 0)
        add(sub, 0, 1, 0, 0)
    # 对子将: 1真+1红
    if counts[t] >= 1 and red >= 1:
        add(_suit_dfs(mod(t, -1), red - 1), 0, 0, 1, 1)
    # 刻子
    if counts[t] >= 3:
        add(_suit_dfs(mod(t, -3), red), 1, 0, 0, 0)
    if counts[t] >= 2 and red >= 1:
        add(_suit_dfs(mod(t, -2), red - 1), 1, 0, 0, 1)
    if counts[t] >= 1 and red >= 2:
        add(_suit_dfs(mod(t, -1), red - 2), 1, 0, 0, 2)
    # 顺子: t 为头/中/尾
    for start in (t - 2, t - 1, t):
        if start < 0 or start + 2 > 8 or tile_rank9(start) > 7:
            continue
        trio = (start, start + 1, start + 2)
        use = [1 if counts[x] >= 1 else 0 for x in trio]
        need = 3 - sum(use)
        if need > red:
            continue
        c2 = list(counts)
        for x, u in zip(trio, use):
            c2[x] -= u
        add(_suit_dfs(tuple(c2), red - need), 1, 0, 0, need)
    # 搭子: 两面/嵌张/红中补
    if t + 1 <= 8 and counts[t + 1] >= 1:
        c2 = list(counts); c2[t] -= 1; c2[t + 1] -= 1
        add(_suit_dfs(tuple(c2), red), 0, 1, 0, 0)
    if t + 2 <= 8 and counts[t + 2] >= 1:
        c2 = list(counts); c2[t] -= 1; c2[t + 2] -= 1
        add(_suit_dfs(tuple(c2), red), 0, 1, 0, 0)
    if red >= 1:
        add(_suit_dfs(mod(t, -1), red - 1), 0, 1, 0, 1)
    return _prune4(cands)


def tile_rank9(i):
    return i % 9 + 1


def _prune4(cands: set) -> tuple:
    """4维 Pareto: (m,t,p 越大越好, r_used 越小越好)"""
    out = []
    for x in cands:
        dominated = False
        for y in cands:
            if y is x:
                continue
            if (y[0] >= x[0] and y[1] >= x[1] and y[2] >= x[2]
                    and y[3] <= x[3]
                    and (y[0], y[1], y[2], -y[3]) != (x[0], x[1], x[2], -x[3])):
                dominated = True
                break
        if not dominated:
            out.append(x)
    return tuple(out)


def shanten_via_suits(counts28) -> int:
    """分花色合并的向听数: 与 win.py:shanten 全等为验证目标。"""
    suits = []
    red_total = counts28[27]
    for s in range(3):
        c = counts28[s * 9:(s + 1) * 9]
        code = 0
        for i in range(8, -1, -1):
            code = code * 5 + c[i]
        suits.append(code)
    need = max(1, (sum(counts28) - 1) // 3)

    # 枚举红中分配 r1+r2+r3 <= red_total, 合并三花色前沿
    best = 99
    fronts0 = suit_front(suits[0], min(red_total, 4))
    for f0 in fronts0:
        r0 = f0[3]
        for r1a in range(red_total - r0 + 1):
            for f1 in suit_front(suits[1], r1a):
                r1 = f1[3]
                if r1 > r1a:
                    continue
                for r2a in range(red_total - r0 - r1 + 1):
                    for f2 in suit_front(suits[2], r2a):
                        if f2[3] > r2a:
                            continue
                        m = f0[0] + f1[0] + f2[0]
                        t = f0[1] + f1[1] + f2[1]
                        p = f0[2] + f1[2] + f2[2]
                        r_left = red_total - r0 - r1 - f2[3]
                        q, rem = divmod(r_left, 3)
                        m += q
                        if rem == 2:
                            variants = ((m, t, p + 1), (m, t + 1, p))
                        else:
                            variants = ((m, t, p),)
                        for mm, tt, pp in variants:
                            mm2 = min(mm, need)
                            tt2 = min(tt, need - mm2)
                            pp2 = min(pp, 1)
                            s = 2 * need - 2 * mm2 - tt2 - pp2
                            best = min(best, s)
    return best


if __name__ == "__main__":
    rng = random.Random(5)
    bad = 0
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
    import time
    t0 = time.time()
    for _ in range(N):
        nr = rng.choice([0, 0, 0, 1, 1, 2, 3, 4])
        pool = []
        for t in range(27):
            pool += [t] * 4
        rng.shuffle(pool)
        c = [0] * 28
        for t in pool[:13 - nr]:
            c[t] += 1
        c[27] = nr
        a = shanten(c)
        b = shanten_via_suits(c)
        if a != b:
            bad += 1
            if bad <= 5:
                print("MISMATCH", a, b,
                      [(t, n) for t, n in enumerate(c) if n])
    print(f"{N} 手对比, 不一致 {bad}, 用时 {time.time()-t0:.1f}s")
