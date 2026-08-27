"""多进程×多卡 JAX 环境吞吐测试: 每进程绑一块 GPU, 各跑独立训练流。

用法:
  python -m backend.jax159.test_multigpu --gpus 0,1,2,3,4,5,6,7 --batch 2048 --steps 100
"""

import argparse
import multiprocessing as mp
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))


def worker(gpu: int, batch: int, steps: int, out_q):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    import jax
    import jax.numpy as jnp
    from backend.jax159.env159 import (reset, step_jit, legal_jit,
                                       load_win_tables)
    load_win_tables()

    assert jax.devices()[0].platform == "gpu", f"GPU{gpu} 不可用"
    dev = jax.devices()[0]
    keys = jax.random.split(jax.random.PRNGKey(gpu), batch)
    vm_reset = jax.jit(jax.vmap(reset))
    vm_step = jax.vmap(step_jit)
    vm_legal = jax.vmap(legal_jit)
    sts = vm_reset(keys)
    # 预热(编译)
    for _ in range(5):
        legal = vm_legal(sts)
        a = jnp.where(sts.phase == 0,
                      jnp.argmax(legal[:, :28].astype(jnp.int32),
                                 axis=-1).astype(jnp.int32),
                      jnp.zeros(batch, dtype=jnp.int32))
        sts = vm_step(sts, a)
    jax.block_until_ready(sts.phase)

    t0 = time.time()
    for _ in range(steps):
        legal = vm_legal(sts)
        a = jnp.where(sts.phase == 0,
                      jnp.argmax(legal[:, :28].astype(jnp.int32),
                                 axis=-1).astype(jnp.int32),
                      jnp.zeros(batch, dtype=jnp.int32))
        sts = vm_step(sts, a)
    jax.block_until_ready(sts.phase)
    dt = time.time() - t0
    rate = batch * steps / dt
    out_q.put((gpu, rate, dt))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", type=str, default="0,1,2,3,4,5,6,7")
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--steps", type=int, default=100)
    args = ap.parse_args()
    gpus = [int(g) for g in args.gpus.split(",")]

    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = [ctx.Process(target=worker,
                         args=(g, args.batch, args.steps, q))
             for g in gpus]
    t0 = time.time()
    for p in procs:
        p.start()
    results = [q.get(timeout=1200) for _ in gpus]
    for p in procs:
        p.join()
    wall = time.time() - t0

    total = sum(r[2] for r in results)  # 各进程耗时之和(并行的 wall 更小)
    print(f"{len(gpus)} 进程并行, batch={args.batch}, {args.steps} 步:")
    for gpu, rate, dt in sorted(results):
        print(f"  GPU{gpu}: {rate:,.0f} env-steps/s ({dt:.2f}s)")
    agg = sum(r[1] for r in results)
    print(f"聚合: {agg:,.0f} env-steps/s, wall {wall:.1f}s "
          f"(理想线性 = 单卡×{len(gpus)})")


if __name__ == "__main__":
    main()
