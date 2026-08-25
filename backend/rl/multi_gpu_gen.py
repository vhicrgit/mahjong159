"""安康159 - 多GPU并行数据生成

将游戏分配到多个GPU进程, 每个进程独立生成数据, 最后合并。
充分利用 8×H20 的算力。

用法:
  python -m backend.rl.multi_gpu_gen --games 10000 --out data_10k.npz
"""

import argparse
import multiprocessing as mp
import os
import time
import numpy as np


def _worker_generate(args):
    """每个 worker 进程在自己 GPU 上生成数据"""
    gpu_id, n_games, seed0, model_path, temperature = args
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    import torch
    from .model import build_model
    from .vec_selfplay import VectorizedSelfPlay

    device = torch.device("cuda")
    if model_path and os.path.exists(model_path):
        ckpt = torch.load(model_path, map_location=device, weights_only=False)
        model = build_model(ckpt["size"]).to(device)
        model.load_state_dict(ckpt["model"])
    else:
        model = build_model("small").to(device)

    engine = VectorizedSelfPlay(model, n_games, device, seed0=seed0)
    results = engine.run(temperature=temperature)

    all_feats, all_acts, all_rets = [], [], []
    for r in results:
        for seat, feat, act, lp, val, regret in r["records"]:
            all_feats.append(feat)
            all_acts.append(act)
            all_rets.append(float(r["scores"][seat]))

    return {
        "feats": np.stack(all_feats).astype(np.float32),
        "acts": np.asarray(all_acts, dtype=np.int64),
        "rets": np.asarray(all_rets, dtype=np.float32),
    }


def generate_multi_gpu(n_games, model_path=None, temperature=0.5,
                        n_gpus=8, out_path=None):
    """多GPU并行生成数据"""
    games_per_gpu = n_games // n_gpus
    remainder = n_games % n_gpus

    tasks = []
    for gpu in range(n_gpus):
        n = games_per_gpu + (1 if gpu < remainder else 0)
        tasks.append((gpu, n, gpu * 100000, model_path, temperature))

    print(f"多GPU生成: {n_games} 局, {n_gpus} 个GPU, 每个 {games_per_gpu} 局")

    t0 = time.time()
    with mp.Pool(n_gpus) as pool:
        results = pool.map(_worker_generate, tasks)
    t1 = time.time()

    # 合并
    all_feats = np.concatenate([r["feats"] for r in results])
    all_acts = np.concatenate([r["acts"] for r in results])
    all_rets = np.concatenate([r["rets"] for r in results])

    total_time = t1 - t0
    total_samples = len(all_acts)
    print(f"  总样本: {total_samples}, 耗时: {total_time:.1f}s, "
          f"{total_samples/total_time:.0f} 样本/s")

    data = {
        "feats": all_feats,
        "acts": all_acts,
        "rets": all_rets,
    }

    if out_path:
        np.savez_compressed(out_path, **data)
        print(f"  已保存: {out_path}")

    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=10000)
    ap.add_argument("--model", type=str, default="")
    ap.add_argument("--temperature", type=float, default=0.5)
    ap.add_argument("--gpus", type=int, default=8)
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()

    generate_multi_gpu(args.games, args.model or None,
                        args.temperature, args.gpus, args.out or None)


if __name__ == "__main__":
    main()
