"""安康159 - 多进程并行评估 (模型 vs 3规则Bot)

评估瓶颈在 CPU 端 encode_state (与自对弈相同), 进程级并行 8x 提速。
用法:
  python -m backend.rl.eval_parallel --model models/dqn_off_v3.pt \
      --games 4000 --procs 8
"""

import argparse
import multiprocessing as mp

import numpy as np
import torch


def _eval_worker(args):
    (model_sd, size, n_games, seed0, feat_dim, feat_version) = args
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from .model import build_model
    from .vec_selfplay import VectorizedSelfPlay
    model = build_model(size, feat_dim=feat_dim).to(device)
    model.load_state_dict(model_sd)
    model.eval()
    engine = VectorizedSelfPlay(model, n_games, device, seed0=seed0,
                                model_seats=[0], feat_version=feat_version)
    results = engine.run(temperature=0.0)
    wins = sum(1 for r in results if r["winner"] == 0)
    scores = [r["scores"][0] for r in results]
    return wins, float(np.sum(scores)), n_games


def evaluate_parallel(model_path: str, games: int = 4000, procs: int = 8):
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    model_sd = ckpt["model"] if "model" in ckpt else ckpt
    size = ckpt.get("size", "base")
    feat_dim = ckpt.get("feat_dim", model_sd["input_proj.weight"].shape[1])
    feat_version = ckpt.get("feat_version",
                            {810: 3, 1146: 4}.get(feat_dim, 2))

    per = games // procs
    tasks = [(model_sd, size, per, 300000 + w * per, feat_dim, feat_version)
             for w in range(procs)]

    ctx = mp.get_context("spawn")
    with ctx.Pool(procs) as pool:
        results = pool.map(_eval_worker, tasks)

    wins = sum(r[0] for r in results)
    total_score = sum(r[1] for r in results)
    n = sum(r[2] for r in results)
    return wins / n, total_score / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--games", type=int, default=4000)
    ap.add_argument("--procs", type=int, default=8)
    args = ap.parse_args()
    wr, avg = evaluate_parallel(args.model, args.games, args.procs)
    print(f"{args.model}: {args.games}局, 胜率 {wr:.1%}, 场均 {avg:+.2f}")


if __name__ == "__main__":
    main()
