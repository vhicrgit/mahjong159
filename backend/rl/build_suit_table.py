"""生成 159 红中向听的单花色 Pareto 前沿表 (JAX 移植的核心资产)。

表结构: front_table[code, r] = 该花色计数向量(base-5 编码) 在 r 张可用红中下的
Pareto 前沿, 编码为 K 个 (m, t, p, r_used) uint8 四元组, 不足补 (255,255,255,255)。

用法: python -m backend.rl.build_suit_table --out models/suit_front_table.npz
校验: 生成后用 suit_table_poc.shanten_via_suits 对拍(其 suit_front 改为读表)。
"""

import argparse
import sys
import time

import numpy as np

sys.path.insert(0, ".")

from .suit_table_poc import suit_front  # noqa: 复用已验证的 DFS

K_MAX = 26          # 前沿上限(实测 <<16)
N_CODE = 5 ** 9     # 1953125
PAD = 255


def encode_front(front):
    # 公式上限 need<=4 → 分量截断 (m<=4, t<=4, p<=1) 后再支配剪枝,
    # 不丢全局信息(全局公式同样截断, 单调安全)
    capped = {(min(m, 4), min(t, 4), min(p, 1), r) for m, t, p, r in front}
    pruned = [x for x in capped if not any(
        y != x and y[0] >= x[0] and y[1] >= x[1] and y[2] >= x[2]
        and y[3] <= x[3] for y in capped)]
    assert len(pruned) <= K_MAX, f"前沿 {len(pruned)} 超 K_MAX={K_MAX}"
    out = np.full((K_MAX, 4), PAD, dtype=np.uint8)
    for i, (m, t, p, r) in enumerate(pruned):
        out[i] = (m, t, p, r)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="models/suit_front_table.npz")
    args = ap.parse_args()

    # 按总牌数分层枚举所有 code (总数<=14 才可达)
    levels: list[list[int]] = [[] for _ in range(15)]

    def gen(pos, remaining, code, k):
        if pos == 9:
            levels[k].append(code)
            return
        for v in range(0, min(4, remaining) + 1):
            gen(pos + 1, remaining - v, code * 5 + v, k + v)

    gen(0, 14, 0, 0)
    total = sum(len(l) for l in levels)
    print(f"可达 code 数: {total} (全空间 {N_CODE})")

    table = np.full((N_CODE, 5, K_MAX, 4), PAD, dtype=np.uint8)
    t0 = time.time()
    done = 0
    for k, codes in enumerate(levels):
        for code in codes:
            for r in range(5):
                table[code, r] = encode_front(suit_front(code, r))
        done += len(codes)
        print(f"level {k}: {len(codes)} codes, 累计 {done}/{total}, "
              f"{time.time()-t0:.0f}s", flush=True)
    np.savez_compressed(args.out, table=table, k_max=K_MAX)
    print(f"已保存 {args.out}, "
          f"内存占用 {table.nbytes/1e6:.0f}MB, 用时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
