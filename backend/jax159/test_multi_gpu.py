"""多进程多卡 rollout 对拍+性能测试。

主进程用 CUDA_VISIBLE_DEVICES 指定的卡构建 State + 单卡对照 rollout;
worker 用物理卡 0..n-1(由 _worker_main 覆盖)。
用法: CUDA_VISIBLE_DEVICES=5 python backend/jax159/test_multi_gpu.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import numpy as np

from backend.jax159.test_pmap_rollout import _rand_games
from backend.jax159.rollout import build_world_states, rollout_jax
from backend.jax159.jax_net import JaxNet
from backend.jax159.parallel_rollout import (start_workers, rollout_parallel,
                                             stop_workers)


def main():
    n_gpus = int(os.environ.get("TEST_GPUS", "2"))
    snaps = _rand_games(14, seed=7)
    cand_lists = []
    for g, hero in snaps:
        hc = g.players[hero].hand_counts
        cand_lists.append([t for t in range(28) if hc[t] > 0][:4])
    net = JaxNet("models/dqn_shaped_100k_best_jax.npz")

    t0 = time.time()
    sts, meta, _ = build_world_states(snaps, cand_lists, 16, world_seed=99)
    B = sts.hands.shape[0]
    print(f"B={B} 构建 {time.time()-t0:.1f}s", flush=True)

    # 单卡对照
    t0 = time.time()
    r1 = rollout_jax(sts, meta, net, step_penalty=0.02)
    t1 = time.time()
    print(f"单卡 rollout: {t1-t0:.1f}s", flush=True)

    # 多卡 worker
    t0 = time.time()
    in_q, out_q, procs = start_workers(n_gpus, "")
    print(f"worker 启动 {time.time()-t0:.1f}s", flush=True)
    # 先用 n_gpus=1 验证 worker 路径正确性(与单卡相同 B, 排除切分浮点)
    t0 = time.time()
    r_w1 = rollout_parallel(sts, meta, net.params, 0.02, in_q, out_q, 1)
    print(f"worker(n=1, 含编译): {time.time()-t0:.1f}s", flush=True)
    bad1 = sum(1 for si in r1 for tile in r1[si]
               if abs(r1[si][tile] - r_w1[si][tile]) > 1e-4)
    print(f"[n=1 验证 worker 正确性] 不一致={bad1}/{sum(len(d) for d in r1.values())}",
          flush=True)
    # n_gpus 切分
    t0 = time.time()
    r2 = rollout_parallel(sts, meta, net.params, 0.02, in_q, out_q, n_gpus)
    t2 = time.time()
    print(f"多卡 rollout({n_gpus}卡, 含首次编译): {t2-t0:.1f}s", flush=True)
    # 第二次(已编译)
    t0 = time.time()
    r2 = rollout_parallel(sts, meta, net.params, 0.02, in_q, out_q, n_gpus)
    print(f"多卡 rollout({n_gpus}卡, 稳态): {time.time()-t0:.1f}s", flush=True)
    stop_workers(in_q, procs)

    # 对比
    bad = 0
    for si in r1:
        for tile in r1[si]:
            a, b = r1[si][tile], r2[si][tile]
            if abs(a - b) > 1e-4:
                bad += 1
                if bad <= 8:
                    print(f"不一致 si={si} tile={tile}: 单卡={a:.4f} 多卡={b:.4f}")
    print(f"候选数={sum(len(d) for d in r1.values())}, 不一致={bad}")
    print("PASS" if bad == 0 else "FAIL", flush=True)


if __name__ == "__main__":
    main()
