"""把已验证的单花色前沿表压成 C 端可直接 mmap 的紧凑表。

产物(models/):
  mj_front.bin : (5^9, 5, 10) uint8 —— A[code][r_used][m*2+p] = 该组合下可达的最大搭子数 t,
                 255 = 不可达。(m<=4, p<=1, t<=4 均已按公式上限截断, 单调安全)
  mj_win.bin   : (5^9, 5, 2) uint8 —— W[code][r_used][0]=能否全部成面子,
                 [1]=能否 1将+全面子 (均为"恰好用 r_used 张红中")

为什么可以只保留 max_t: 最终公式
  shanten = 2*need - 2*min(m,need) - min(t, need-min(m,need)) - min(p,1)
对固定 (m,p) 关于 t 单调不减, 所以每个 (m,p) 只需最大 t。
合并代价从原表的 35×26^3 降到 <=15×100×2。
"""

import argparse
import time

import numpy as np

N_CODE = 5 ** 9
PAD = 255


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="models/suit_front_table.npz")
    ap.add_argument("--out-front", default="models/mj_front.bin")
    ap.add_argument("--out-win", default="models/mj_win.bin")
    args = ap.parse_args()

    t0 = time.time()
    z = np.load(args.src)
    T = z["table"]                      # (N,5,K,4) uint8: (m,t,p,r_used)
    K = T.shape[2]
    F = T[:, 4]                         # r_avail=4 的前沿含所有 r_used<=4 的非支配项
    print(f"载入 {args.src} K={K} {time.time()-t0:.0f}s")

    m = F[..., 0].astype(np.int64)
    tt = F[..., 1].astype(np.int64)
    p = F[..., 2].astype(np.int64)
    r = F[..., 3].astype(np.int64)
    valid = F[..., 0] != PAD

    # ---- 紧凑前沿表 A[code][r_used][m*2+p] = max t ----
    A = np.zeros(N_CODE * 5 * 10, dtype=np.uint8)
    reach = np.zeros(N_CODE * 5 * 10, dtype=bool)
    codes = np.arange(N_CODE, dtype=np.int64)
    for k in range(K):
        vk = valid[:, k]
        if not vk.any():
            continue
        idx = codes[vk]
        mk = np.minimum(m[vk, k], 4)
        pk = np.minimum(p[vk, k], 1)
        tk = np.minimum(tt[vk, k], 4).astype(np.uint8)
        rk = r[vk, k]
        flat = (idx * 5 + rk) * 10 + (mk * 2 + pk)
        # 同一 k 内 flat 互不相同(每个 code 只有一项), 可直接花式索引取 max
        A[flat] = np.maximum(A[flat], tk)
        reach[flat] = True
    A = np.where(reach, A, PAD).astype(np.uint8)
    A.tofile(args.out_front)
    print(f"{args.out_front}: {A.nbytes/1e6:.1f}MB  可达槽 {reach.sum()} "
          f"{time.time()-t0:.0f}s")

    # ---- 判胡表 W[code][r_used][0/1] ----
    pow5 = (5 ** np.arange(9)).astype(np.int64)
    digits = (codes[:, None] // pow5[None, :]) % 5
    tile_cnt = digits.sum(axis=1)                       # (N,)
    W = np.zeros((N_CODE, 5, 2), dtype=np.uint8)
    cnt_b = tile_cnt[:, None]
    full0 = valid & (tt == 0) & (p == 0) & (3 * m == cnt_b + r)
    full1 = valid & (tt == 0) & (p == 1) & (3 * m + 2 == cnt_b + r)
    for k in range(K):
        for ru in range(5):
            sel0 = full0[:, k] & (r[:, k] == ru)
            if sel0.any():
                W[sel0, ru, 0] = 1
            sel1 = full1[:, k] & (r[:, k] == ru)
            if sel1.any():
                W[sel1, ru, 1] = 1
    W.tofile(args.out_win)
    print(f"{args.out_win}: {W.nbytes/1e6:.1f}MB  "
          f"W0={int(W[...,0].sum())} W1={int(W[...,1].sum())} "
          f"{time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
