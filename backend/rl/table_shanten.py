"""查表版向听计算: 读 build_suit_table 生成的前沿表, 合并三花色。

接口与 rules.win.shanten 完全一致, 供规则Bot/评估提速。
正确性由 table_vs_dfs 校验(全等才允许启用)。
"""

import os
from functools import lru_cache

import numpy as np

_TABLE = None
PAD = 255
K_MAX = 24


def _load():
    global _TABLE
    if _TABLE is None:
        path = os.environ.get("SUIT_TABLE",
                              "models/suit_front_table.npz")
        z = np.load(path)
        _TABLE = z["table"]
    return _TABLE


def _suit_code(counts, s):
    code = 0
    for i in range(s * 9 + 8, s * 9 - 1, -1):
        code = code * 5 + counts[i]
    return code


def _front(code, r):
    row = _load()[code, r]  # (K,4) uint8
    return row[row[:, 0] != PAD]


@lru_cache(maxsize=200000)
def table_shanten(counts28: tuple) -> int:
    codes = [_suit_code(counts28, s) for s in range(3)]
    red = counts28[27]
    total = sum(counts28)
    need = max(1, (total - 1) // 3)
    f0 = _front(codes[0], red)
    f1 = _front(codes[1], red)
    f2 = _front(codes[2], red)
    best = 99
    for m0, t0, p0, r0 in f0.tolist():
        rem1 = red - int(r0)
        if rem1 < 0:
            continue
        for m1, t1, p1, r1 in f1:
            if r1 > rem1:
                continue
            m01, t01, p01 = m0 + m1, t0 + t1, p0 + p1
            rem2 = rem1 - int(r1)
            for m2, t2, p2, r2 in f2:
                if r2 > rem2:
                    continue
                m = int(m01 + m2)
                t = int(t01 + t2)
                p = int(p01 + p2)
                left = rem2 - int(r2)
                q, rem = divmod(left, 3)
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
                    if s < best:
                        best = s
    return best


def table_vs_dfs(n: int = 20000, seed: int = 17) -> int:
    """对拍 rules.win.shanten, 返回不一致数。"""
    import random
    from ..rules.win import shanten
    rng = random.Random(seed)
    bad = 0
    for _ in range(n):
        nr = rng.choice([0, 0, 0, 1, 1, 2, 3, 4])
        pool = []
        for t in range(27):
            pool += [t] * 4
        rng.shuffle(pool)
        c = [0] * 28
        for t in pool[:13 - nr]:
            c[t] += 1
        c[27] = nr
        if shanten(c) != table_shanten(tuple(c)):
            bad += 1
    return bad


if __name__ == "__main__":
    import time
    t0 = time.time()
    bad = table_vs_dfs(int(os.environ.get("N", "20000")))
    dt = time.time() - t0
    print(f"对拍 {os.environ.get('N', '20000')} 手, 不一致 {bad}, "
          f"用时 {dt:.1f}s ({dt/max(1,int(os.environ.get('N','20000')))*1000:.2f}ms/手)")
