"""react 决策的 JAX 版: 与 backend.rules.win 的 v1 语义对拍。

v1 语义(_peng_gang_ok): 碰/杠后最优弃牌向听 < 当前向听。
JAX 版用 shanten_batch 查表(need 自动由张数推导), 批量处理。
"""
import jax
import jax.numpy as jnp
import numpy as np

from backend.jax159.shanten import shanten_batch, load_front_table
from backend.jax159.rollout import _peng_gang_ok


@jax.jit
def peng_gang_ok_batch(hands, tiles, n: int):
    """hands: (R, 28) int 计数(13张); tiles: (R,); n: 2(碰)/3(杠)
    返回 (R,) bool。语义与 _peng_gang_ok 一致。
    注意 jit: R 变化会重编译, 调用方应 pad 到固定批量。"""
    R = hands.shape[0]
    idx = jnp.arange(R)
    c11 = hands.at[idx, tiles].add(-n).astype(jnp.int32)      # (R,28) 碰后
    eye = jnp.eye(28, dtype=jnp.int32)
    # 枚举弃牌 d; 负计数 clip 到 0(valid mask 会丢弃这些行)
    c10 = jnp.clip(c11[:, None, :] - eye[None, :, :], 0, 4)   # (R,28,28)
    valid = c11 > 0
    s = shanten_batch(c10.reshape(-1, 28)).reshape(R, 28)
    after = jnp.where(valid, s, 99).min(axis=1)
    before = shanten_batch(hands.astype(jnp.int32))
    return after < before


def main():
    load_front_table()
    rng = np.random.RandomState(42)
    # 批量生成随机合法手牌(13张)
    B = 512
    counts_all = np.zeros((B, 28), dtype=np.int32)
    for i in range(B):
        remain = 13
        while remain > 0:
            t = rng.randint(0, 28)
            if counts_all[i, t] < 4:
                counts_all[i, t] += 1
                remain -= 1
    mismatch = 0
    total = 0
    for n in (2, 3):
        # 每手随机选一张 >=n 的牌作为目标
        tiles = np.zeros(B, dtype=np.int32)
        ok = np.zeros(B, dtype=bool)
        for i in range(B):
            cands = [t for t in range(28) if counts_all[i, t] >= n]
            if cands:
                tiles[i] = rng.choice(cands)
                ok[i] = True
        idx = np.nonzero(ok)[0]
        if len(idx) == 0:
            continue
        jx = np.asarray(peng_gang_ok_batch(
            jnp.asarray(counts_all[idx]), jnp.asarray(tiles[idx]), n))
        for j, i in enumerate(idx):
            py = _peng_gang_ok(list(counts_all[i]), int(tiles[i]), n)
            total += 1
            if py != bool(jx[j]):
                mismatch += 1
                if mismatch <= 5:
                    print(f"不一致: counts={counts_all[i].tolist()} "
                          f"tile={tiles[i]} n={n} py={py} jax={bool(jx[j])}")
    print(f"对拍 {total} 例, 不一致 {mismatch}")
    print("PASS" if mismatch == 0 else "FAIL")


if __name__ == "__main__":
    main()
