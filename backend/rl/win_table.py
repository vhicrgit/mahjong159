"""159 判胡查表法 (MahJax 移植核心)

从前沿表(suit_front_table.npz)导出每张花色的两个布尔函数:
- M0(code, r): 该花色牌 + 至多 r 张红中, 能否**全部**组成面子
- M1(code, r): 该花色牌 + 至多 r 张红中, 能否组成 **1 将 + 全面子**
判胡 = 枚举将所在花色(或双红中为将) × 枚举红中分配, 全部用表查询。
纯 numpy 参考实现; JAX 版在 jax159 中做同样的 gather。
"""

import os
import sys

import numpy as np

sys.path.insert(0, ".")

N_CODE = 5 ** 9
PAD = 255


def build_win_tables(front_table: np.ndarray):
    """front_table: (N_CODE, 5, K, 4) uint8 -> M0, M1: (N_CODE, 5) bool"""
    m = front_table[..., 0].astype(np.int32)   # (C,5,K)
    t = front_table[..., 1]
    p = front_table[..., 2]
    r = front_table[..., 3].astype(np.int32)
    valid = front_table[..., 0] != PAD

    # 每种 code 的真实牌数: base-5 解码各位求和
    codes = np.arange(N_CODE, dtype=np.int64)
    pow5 = (5 ** np.arange(9)).astype(np.int64)
    digits = (codes[:, None] // pow5[None, :]) % 5          # (C,9)
    tile_cnt = digits.sum(axis=1).astype(np.int32)          # (C,)

    # M0: t==0,p==0 且 3m == 该花色真实牌数 + r_used (红中是额外补进来的牌)
    full0 = valid & (t == 0) & (p == 0) & \
        (3 * m == tile_cnt[:, None, None] + r)
    # M1: t==0,p==1 且 3m + 2 == 真实牌数 + r_used
    full1 = valid & (t == 0) & (p == 1) & \
        (3 * m + 2 == tile_cnt[:, None, None] + r)

    # 按 (code, r_avail) 归约: 表第2轴即 r_avail, 其前沿已满足 r_used<=r_avail
    M0 = full0.any(axis=-1)
    M1 = full1.any(axis=-1)
    return M0, M1


def _suit_code(counts, s):
    code = 0
    for i in range(s * 9 + 8, s * 9 - 1, -1):
        code = code * 5 + counts[i]
    return code


def win_from_table(counts28, M0, M1) -> bool:
    """counts28: 长度28计数(张数须 3n+2)。与 rules.win.is_win 全等为目标。"""
    red = counts28[27]
    cs = [_suit_code(counts28, s) for s in range(3)]
    # 枚举三花色红中用量分配
    for r0 in range(red + 1):
        for r1 in range(red - r0 + 1):
            for r2 in range(red - r0 - r1 + 1):
                left = red - r0 - r1 - r2
                # 将在花色 s: 剩红必须 0 或成刻(3)
                if left % 3 == 0:
                    if M1[cs[0], r0] and M0[cs[1], r1] and M0[cs[2], r2]:
                        return True
                    if M0[cs[0], r0] and M1[cs[1], r1] and M0[cs[2], r2]:
                        return True
                    if M0[cs[0], r0] and M0[cs[1], r1] and M1[cs[2], r2]:
                        return True
                # 将 = 两张红中: left 必须恰好 == 2
                if left == 2 and M0[cs[0], r0] and M0[cs[1], r1] \
                        and M0[cs[2], r2]:
                    return True
    return False


def main():
    import random
    from backend.rules.win import is_win
    z = np.load("models/suit_front_table.npz")
    M0, M1 = build_win_tables(z["table"])
    np.savez_compressed("models/win_table.npz", M0=M0, M1=M1)
    print("win 表已导出: models/win_table.npz")

    rng = random.Random(23)
    bad = 0
    N = int(os.environ.get("N", "50000"))
    for _ in range(N):
        nr = rng.choice([0, 0, 0, 1, 1, 2, 3, 4])
        pool = []
        for t in range(27):
            pool += [t] * 4
        rng.shuffle(pool)
        c = [0] * 28
        for t in pool[:14 - nr]:
            c[t] += 1
        c[27] = nr
        if is_win(c) != win_from_table(c, M0, M1):
            bad += 1
            if bad <= 3:
                print("MISMATCH", is_win(c), [(t, n) for t, n in enumerate(c) if n])
    print(f"判胡对拍 {N} 手, 不一致 {bad}")


if __name__ == "__main__":
    main()
