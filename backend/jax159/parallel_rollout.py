"""多卡并行 rollout: 常驻 worker 进程, 每进程 CUDA_VISIBLE_DEVICES=i, 分摊世界推演。

用法: rollout_parallel(sts, meta, net_params, step_penalty, n_gpus)
每 iter 通过 Queue 分发分块 State, worker 用自身 GPU 跑 _rollout_jit 后回传奖励。
"""

import multiprocessing as mp
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

_worker_ctx = None


def _worker_main(in_q, out_q, gpu, cache_dir):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    import jax
    import jax.numpy as jnp
    from backend.jax159.env159 import load_win_tables
    from backend.jax159.shanten import load_front_table
    from backend.jax159.rollout import rollout_jax
    from backend.jax159.jax_net import JaxNet
    load_win_tables()
    load_front_table()
    while True:
        task = in_q.get()
        if task is None:
            break
        sts_fields, meta, net_params, step_penalty = task
        net = JaxNet.from_dict(net_params)
        sts = _fields_to_state(sts_fields)
        r_map = rollout_jax(sts, meta, net, step_penalty)
        out_q.put(r_map)
        # 释放大数组
        del sts_fields, sts, net


def _fields_to_state(fields):
    from backend.jax159.env159 import State
    return State(**fields)


def _state_to_fields(sts):
    return {f: np.asarray(getattr(sts, f)) for f in sts._fields}


def start_workers(n_gpus, cache_dir):
    global _worker_ctx
    ctx = mp.get_context("spawn")
    _worker_ctx = ctx
    in_q = ctx.Queue()
    out_q = ctx.Queue()
    procs = [ctx.Process(target=_worker_main,
                         args=(in_q, out_q, i, cache_dir))
             for i in range(n_gpus)]
    for p in procs:
        p.start()
    return in_q, out_q, procs


def rollout_parallel(sts, meta, net_params, step_penalty, in_q, out_q,
                     n_gpus):
    """把 sts 分 n_gpus 块派给 worker, 聚合奖励。"""
    B = sts.hands.shape[0]
    chunk = B // n_gpus
    results = [None] * n_gpus
    for i in range(n_gpus):
        lo, hi = i * chunk, (i + 1) * chunk if i < n_gpus - 1 else B
        fields = _state_to_fields(slice_state_fields(sts, lo, hi))
        in_q.put((fields, meta[lo:hi], net_params, step_penalty))
    for i in range(n_gpus):
        r_map = out_q.get()
        results[i] = r_map
    # 合并 (按 meta 顺序)
    merged = {}
    for i in range(n_gpus):
        merged.update(results[i])
    return merged


def slice_state_fields(sts, lo, hi):
    from backend.jax159.env159 import State
    return State(**{f: getattr(sts, f)[lo:hi] for f in sts._fields})


def stop_workers(in_q, procs):
    for _ in procs:
        in_q.put(None)
    for p in procs:
        p.join(timeout=10)
