"""横向对比各后端的"批量向听"吞吐, 回答: 为什么 v31 加速要选 C 而不是 numpy/torch/jax。

四个后端算同一件事 —— 给 N 手牌算向听, 且都与 rules.win.shanten 同口径:
  native : backend/native/mj159.c   紧凑表 A[code][r][m*2+p] + 哈希记忆化
  numpy  : 同一张紧凑表, 纯 numpy 的 (max,+) 卷积
  torch  : 同一张紧凑表, 同样的卷积(本机 torch 是 CPU build)
  jax    : 既有 backend/jax159/shanten.py, 用原始前沿表做 35×K^3 合并

结论的两个支点(脚本会分别给出数字):
  1. 单手成本: 紧凑表把三花色合并从 35×26^3 降到 <=15×100×2, 所以 native/numpy
     远快于既有 jax 实现; 而 native 又比 numpy 快一个量级(小张量上 numpy 全是
     per-op 开销)。
  2. 调用次数: 一次 v10/v31 决策在有记忆化时只算 ~2.4 万次向听, 关掉记忆化要
     ~24 万次。向量化后端没法记忆化, 必须把分支算满, 白多干 10 倍。
"""

import argparse
import time

import numpy as np

NCODE = 5 ** 9
PAD = 255
POW5 = (5 ** np.arange(9)).astype(np.int64)

# 每决策的向听调用数: 有 lru_cache 实测值 / 纯枚举口径(~10候选×28摸×28打×30)
SH_MEMO = 23890
SH_NOMEMO = 235200
DEC_PER_GAME = 29


def load_compact(path="models/mj_front.bin"):
    return np.fromfile(path, dtype=np.uint8).reshape(NCODE, 5, 10)


class NP:
    """numpy 适配层。"""
    name = "numpy"

    @staticmethod
    def table(A):
        return A

    @staticmethod
    def hands(h):
        return h

    @staticmethod
    def to_np(x):
        return np.asarray(x)

    minimum = staticmethod(np.minimum)
    maximum = staticmethod(np.maximum)
    where = staticmethod(np.where)
    clip = staticmethod(np.clip)

    @staticmethod
    def i64(x):
        return x.astype(np.int64)

    @staticmethod
    def i16(x):
        return x.astype(np.int16)

    @staticmethod
    def i32(x):
        return x.astype(np.int32)

    @staticmethod
    def full(shape, v, dt):
        return np.full(shape, v, dtype={"i16": np.int16, "i32": np.int32}[dt])

    @staticmethod
    def pow5():
        return POW5

    @staticmethod
    def sync():
        pass


class TORCH:
    name = "torch"

    def __init__(self):
        import torch
        self.torch = torch
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.name = f"torch({self.dev})"

    def table(self, A):
        return self.torch.from_numpy(A).to(self.dev)

    def hands(self, h):
        return self.torch.from_numpy(h).to(self.dev)

    def to_np(self, x):
        return x.cpu().numpy()

    def minimum(self, a, b):
        t = self.torch
        if not t.is_tensor(b):
            b = t.tensor(b, dtype=a.dtype, device=a.device)
        return t.minimum(a, b.to(a.dtype))

    def maximum(self, a, b):
        t = self.torch
        if not t.is_tensor(b):
            b = t.tensor(b, dtype=a.dtype, device=a.device)
        return t.maximum(a, b.to(a.dtype))

    def where(self, c, a, b):
        t = self.torch
        if not t.is_tensor(a):
            a = t.tensor(a, dtype=b.dtype, device=b.device)
        if not t.is_tensor(b):
            b = t.tensor(b, dtype=a.dtype, device=a.device)
        return t.where(c, a, b)

    def i64(self, x):
        return x.to(self.torch.int64)

    def i16(self, x):
        return x.to(self.torch.int16)

    def i32(self, x):
        return x.to(self.torch.int32)

    def full(self, shape, v, dt):
        t = self.torch
        return t.full(shape, v,
                      dtype={"i16": t.int16, "i32": t.int32}[dt],
                      device=self.dev)

    def pow5(self):
        return self.torch.from_numpy(POW5).to(self.dev)

    def sync(self):
        if self.dev == "cuda":
            self.torch.cuda.synchronize()


def shanten_batch_xp(hands, A, xp):
    """后端无关的紧凑表向听。hands (B,28) -> (B,) 向听数。"""
    c = xp.i64(hands)
    p5 = xp.pow5()
    c0 = (c[:, 0:9] * p5).sum(1)
    c1 = (c[:, 9:18] * p5).sum(1)
    c2 = (c[:, 18:27] * p5).sum(1)
    red = c[:, 27]
    total = c.sum(1)
    need = xp.minimum(xp.maximum((total - 1) // 3, 1), 4)
    B = hands.shape[0]

    def first(code):
        G = xp.full((B, 5, 10), -1, "i16")
        tbl = A[code]
        for r in range(5):
            a = xp.i16(tbl[:, r, :])
            ok = (a != PAD) & (r <= red)[:, None]
            G[:, r, :] = xp.where(ok, a, G[:, r, :])
        return G

    def merge(G, code):
        out = xp.full((B, 5, 10), -1, "i16")
        tbl = A[code]
        for r_add in range(5):
            a = xp.i16(tbl[:, r_add, :])
            av = a != PAD
            for R in range(5 - r_add):
                g = G[:, R, :]
                Rn = R + r_add
                budget = Rn <= red
                for i in range(10):
                    gi = g[:, i]
                    okg = (gi >= 0) & budget
                    m0, p0 = i >> 1, i & 1
                    for j in range(10):
                        idx = min(m0 + (j >> 1), 4) * 2 + min(p0 + (j & 1), 1)
                        t = xp.minimum(gi + a[:, j], 4)
                        ok = okg & av[:, j]
                        out[:, Rn, idx] = xp.where(
                            ok, xp.maximum(out[:, Rn, idx], t),
                            out[:, Rn, idx])
        return out

    G = merge(merge(first(c0), c1), c2)

    best = xp.full((B,), 99, "i32")
    for R in range(5):
        left = red - R
        okR = left >= 0
        q = xp.where(okR, left // 3, 0)
        rem = xp.where(okR, left % 3, 0)
        is2 = rem == 2
        for i in range(10):
            t = xp.i32(G[:, R, i])
            mm = xp.minimum((i >> 1) + q, need)
            room = need - mm
            p = i & 1
            base = 2 * need - 2 * mm
            sA = base - xp.minimum(t + 1, room) - p
            sB = base - xp.minimum(t, room) - min(p + 1, 1)
            sN = base - xp.minimum(t, room) - p
            s = xp.i32(xp.where(is2, xp.minimum(sA, sB), sN))
            good = (t >= 0) & okR
            best = xp.where(good, xp.minimum(best, s), best)
    return best


def timed(fn, n, warm=1):
    for _ in range(warm):
        fn()
    t0 = time.perf_counter()
    fn()
    return n / (time.perf_counter() - t0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8192)
    ap.add_argument("--backends", default="native,numpy,torch,jax")
    ap.add_argument("--check", type=int, default=2000)
    args = ap.parse_args()

    rng = np.random.default_rng(7)
    hands = np.zeros((args.n, 28), dtype=np.int8)
    for i in range(args.n):
        pool = np.repeat(np.arange(27), 4)
        rng.shuffle(pool)
        nred = int(rng.choice([0, 0, 0, 1, 1, 2]))
        for t in pool[:13 - nred]:
            hands[i, t] += 1
        hands[i, 27] = nred

    want = args.backends.split(",")
    res = {}
    ref = None

    from backend.native import native
    native.lib()
    ref = native.shanten_batch(hands[:args.check])
    if "native" in want:
        res["native (C, 紧凑表+记忆化)"] = timed(
            lambda: native.shanten_batch(hands), args.n)

    A = load_compact()
    for tag in ("numpy", "torch"):
        if tag not in want:
            continue
        xp = NP() if tag == "numpy" else TORCH()
        At = xp.table(A)
        Ht = xp.hands(hands)
        got = xp.to_np(shanten_batch_xp(xp.hands(hands[:args.check]), At, xp))
        bad = int((got != ref).sum())
        print(f"{xp.name} 与 native 对拍 {args.check} 手: 不一致 {bad}")

        def run():
            shanten_batch_xp(Ht, At, xp)
            xp.sync()
        res[f"{xp.name} (紧凑表, 无记忆化)"] = timed(run, args.n)

    if "jax" in want:
        try:
            import backend.jax159  # noqa: 触发 nvjitlink 预载, 否则 jax 静默用 CPU
            import jax
            import jax.numpy as jnp
            from backend.jax159.shanten import shanten_batch
            dev = jax.devices()[0]
            h = jnp.asarray(hands)
            t0 = time.perf_counter()
            out = shanten_batch(h).block_until_ready()
            tc = time.perf_counter() - t0
            res[f"jax({dev.platform}, 既有前沿表 35xK^3)"] = timed(
                lambda: shanten_batch(h).block_until_ready(), args.n, warm=0)
            bad = int((np.asarray(out[:args.check]) != ref).sum())
            print(f"jax({dev.platform}) 与 native 对拍 {args.check} 手: "
                  f"不一致 {bad}; 首次(含编译) {tc:.1f}s")
        except Exception as e:
            print("jax 后端跳过:", type(e).__name__, e)

    print(f"\n单手向听吞吐 (N={args.n}, 随机手牌=记忆化命中率最差的情况):")
    for k, v in sorted(res.items(), key=lambda kv: -kv[1]):
        print(f"  {k:38s} {v:12,.0f} 手/s")

    print("\n为什么向量化赢不了 —— 一次迭代(32快照×4候选×128世界=16384局)需要的向听次数:")
    print(f"  有记忆化(C 走这条): 16384×{DEC_PER_GAME}×{SH_MEMO:,} = "
          f"{16384*DEC_PER_GAME*SH_MEMO:,}")
    print(f"  无记忆化(向量化必走): 16384×{DEC_PER_GAME}×{SH_NOMEMO:,} = "
          f"{16384*DEC_PER_GAME*SH_NOMEMO:,}  (多 "
          f"{SH_NOMEMO/SH_MEMO:.0f}x)")
    print("  注: C 的实测有效速率远高于上表(上表是冷表随机手牌), 因为记忆化命中;")
    print("      端到端实测见 logs/v12_smoke_v31n_w128.log —— 128 世界 ~100s/迭代。")


if __name__ == "__main__":
    main()
