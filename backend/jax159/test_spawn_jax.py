"""最小复现: spawn worker 里 jax GPU 是否可用。"""

import multiprocessing as mp
import os
import sys

sys.path.insert(0, ".")

NVLIBS = os.environ.get("NVLIBS", "")


def worker(gpu):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    import jax
    print(f"worker gpu={gpu}: devices={len(jax.devices())} "
          f"first={jax.devices()[0] if jax.devices() else None}", flush=True)
    import jax.numpy as jnp
    x = jnp.ones((100, 100))
    print("matmul:", float((x @ x).sum()), flush=True)


def main():
    ctx = mp.get_context("spawn")
    p = ctx.Process(target=worker, args=(0,))
    p.start()
    p.join(timeout=120)
    print("worker exitcode:", p.exitcode)


if __name__ == "__main__":
    main()
