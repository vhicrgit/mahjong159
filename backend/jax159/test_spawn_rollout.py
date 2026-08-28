"""spawn worker 里直接调 rollout_jax 的最小测试。"""

import multiprocessing as mp
import os
import sys

sys.path.insert(0, ".")

import numpy as np


def worker(gpu, cache_dir):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    if cache_dir:
        os.environ["JAX_COMPILATION_CACHE_DIR"] = cache_dir
    try:
        from backend.jax159.env159 import load_win_tables
        from backend.jax159.shanten import load_front_table
        from backend.jax159.rollout import rollout_jax
        from backend.jax159.jax_net import JaxNet
        load_win_tables()
        load_front_table()
        # 最小 State: 1 局
        import jax.numpy as jnp
        from backend.jax159.env159 import State
        sts = State(
            wall=jnp.zeros(112, dtype=jnp.int8),
            wall_pos=jnp.int16(0), wall_tail=jnp.int16(59),
            hands=jnp.zeros((1, 4, 28), dtype=jnp.int8),
            discards=jnp.zeros((1, 4, 28), dtype=jnp.int16),
            melds_tile=jnp.zeros((1, 4, 4), dtype=jnp.int8),
            melds_kind=jnp.zeros((1, 4, 4), dtype=jnp.int8),
            melds_from=jnp.full((1, 4, 4), -1, dtype=jnp.int8),
            n_melds=jnp.zeros((1, 4), dtype=jnp.int8),
            turn=jnp.zeros(1, dtype=jnp.int8),
            phase=jnp.zeros(1, dtype=jnp.int8),
            last_discard=jnp.full(1, -1, dtype=jnp.int8),
            last_discarder=jnp.full(1, -1, dtype=jnp.int8),
            pend_peng=jnp.zeros((1, 4), bool),
            pend_gang=jnp.zeros((1, 4), bool),
            winner=jnp.full(1, -1, dtype=jnp.int8),
            win_kind=jnp.zeros(1, dtype=jnp.int8),
            n_159=jnp.zeros(1, dtype=jnp.int8),
            scores=jnp.zeros((1, 4), dtype=jnp.int16),
            draws=jnp.zeros(1, dtype=jnp.int16),
        )
        # 随便一个 net
        import numpy as np
        p = {}
        d = 628
        for k in ("input_proj.weight", "input_proj.bias"):
            p[k] = np.random.randn(*(512, d) if "weight" in k else (512,)).astype(np.float32)
        for i in range(12):
            p[f"blocks.{i}.fc1.weight"] = np.random.randn(512, 512).astype(np.float32)
            p[f"blocks.{i}.fc1.bias"] = np.zeros(512, dtype=np.float32)
            p[f"blocks.{i}.fc2.weight"] = np.random.randn(512, 512).astype(np.float32)
            p[f"blocks.{i}.fc2.bias"] = np.zeros(512, dtype=np.float32)
        p["q_head.weight"] = np.random.randn(28, 512).astype(np.float32)
        p["q_head.bias"] = np.zeros(28, dtype=np.float32)
        net = JaxNet.from_dict(p)
        meta = [(0, 0, 0)]
        print("worker: 开始 rollout_jax", flush=True)
        r = rollout_jax(sts, meta, net, 0.02)
        print("worker: rollout 完成", r, flush=True)
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise


def main():
    ctx = mp.get_context("spawn")
    cache = os.environ.get("JAX_COMPILATION_CACHE_DIR", "")
    p = ctx.Process(target=worker, args=(0, cache))
    p.start()
    p.join(timeout=300)
    print("worker exitcode:", p.exitcode)


if __name__ == "__main__":
    main()
