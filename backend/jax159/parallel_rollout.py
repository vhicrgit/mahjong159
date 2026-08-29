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
    # 强制限制 JAX 显存(不继承主进程): 给同卡主进程留空间; 单卡 worker 也用不满
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.45"
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
    """把 sts 分 n_gpus 块派给 worker, 聚合奖励。
    net_params: dict(可为 jnp), 统一转 numpy 以便 pickle 传输。

    切分按 snap(si) 对齐: 同一 snap 的所有(候选×世界)局面必须在同一
    worker —— 否则不同 worker 的 XLA kernel 浮点差异会成为候选间的
    系统误差(而非随机噪声), spread 虚高、组内优势方向错误(实测:
    连续切分 spread 0.9 vs 对齐后应回到 0.2, it10 胜率 23% vs 27%)。"""
    net_params = {k: np.asarray(v) for k, v in net_params.items()}
    B = sts.hands.shape[0]
    # 按 si 分组(局面连续): si -> (start, end)
    si_order, si_span = [], {}
    for i, m in enumerate(meta):
        si = m[0]
        if si not in si_span:
            si_order.append(si)
            si_span[si] = [i, i]
        si_span[si][1] = i + 1
    n_si = len(si_order)
    per, rem = divmod(n_si, n_gpus)
    assign = []  # [(lo, hi)] 局面区间
    idx = 0
    for w in range(n_gpus):
        cnt = per + (1 if w < rem else 0)
        if cnt == 0:
            assign.append(None)
            continue
        sis = si_order[idx: idx + cnt]
        idx += cnt
        assign.append((si_span[sis[0]][0], si_span[sis[-1]][1]))
    for w, span in enumerate(assign):
        if span is None:  # 局面数 < worker 数, 空 worker 跳过一个局面
            fields = _state_to_fields(slice_state_fields(sts, 0, 1))
            in_q.put((fields, meta[:1], net_params, step_penalty))
            continue
        lo, hi = span
        fields = _state_to_fields(slice_state_fields(sts, lo, hi))
        in_q.put((fields, meta[lo:hi], net_params, step_penalty))
    results = [None] * n_gpus
    for i in range(n_gpus):
        r_map = out_q.get()
        results[i] = r_map
    merged = {}
    for i, r in enumerate(results):
        if assign[i] is None:
            continue  # 空 worker 的结果是 pad 局面的, 丢弃
        merged.update(r)
    return merged


def slice_state_fields(sts, lo, hi):
    from backend.jax159.env159 import State
    return State(**{f: getattr(sts, f)[lo:hi] for f in sts._fields})


def stop_workers(in_q, procs):
    for _ in procs:
        in_q.put(None)
    for p in procs:
        p.join(timeout=10)
