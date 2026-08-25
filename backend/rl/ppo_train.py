"""安康159 - PPO 迭代训练主循环

流程:
  1. BC 热身: 用规则Bot数据训练出基本牌效
  2. PPO 迭代: 向量化自对弈 → GAE → PPO更新 → 评估

用法:
  python -m backend.rl.ppo_train --size small --iters 50 --games-per-iter 512
"""

import argparse
import os
import time
import numpy as np
import torch

from .model import build_model
from .selfplay import generate_dataset
from .vec_selfplay import VectorizedSelfPlay
from .train import get_device, train_bc
from .ppo import PPOTrainer, compute_gae_per_trajectory
from .evaluate import play_eval_game


def evaluate_vs_rule(model_path, games=100):
    wins, total = 0, 0.0
    for i in range(games):
        sd, won = play_eval_game(200000 + i, model_path)
        total += sd
        wins += won
    return wins / games, total / games


def collect_ppo_batch(model, n_games, device, temperature, seed0,
                      batch_games=256):
    """收集 PPO 数据, 计算 GAE"""
    engine = VectorizedSelfPlay(model, n_games, device, seed0=seed0)
    results = engine.run(temperature=temperature)

    all_feats, all_acts, all_lps, all_advs, all_rets = [], [], [], [], []
    for r in results:
        if not r["records"]:
            continue
        advs, rets = compute_gae_per_trajectory(r["records"], r["scores"])
        for j, (seat, feat, act, lp, val) in enumerate(r["records"]):
            all_feats.append(feat)
            all_acts.append(act)
            all_lps.append(lp)
            all_advs.append(advs[j])
            all_rets.append(rets[j])

    return {
        "feats": np.stack(all_feats).astype(np.float32),
        "acts": np.asarray(all_acts, dtype=np.int64),
        "old_lps": np.asarray(all_lps, dtype=np.float32),
        "advs": np.asarray(all_advs, dtype=np.float32),
        "rets": np.asarray(all_rets, dtype=np.float32),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=str, default="small")
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--games-per-iter", type=int, default=512)
    ap.add_argument("--bc-games", type=int, default=2000)
    ap.add_argument("--bc-epochs", type=int, default=30)
    ap.add_argument("--ppo-epochs", type=int, default=4)
    ap.add_argument("--batch-games", type=int, default=256,
                    help="向量化自对弈每批并行局数")
    ap.add_argument("--temperature", type=float, default=0.5)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--init", type=str, default="")
    ap.add_argument("--out", type=str, default="model_ppo.pt")
    ap.add_argument("--eval-games", type=int, default=100)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    device = get_device()
    print(f"PPO 训练: {args.size} 模型, {device}, "
          f"温度={args.temperature}, {args.iters} 轮")

    model = build_model(args.size).to(device)
    print(f"参数量: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    # ---- BC 热身 ----
    if args.init and os.path.exists(args.init):
        ckpt = torch.load(args.init, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        print(f"加载初始模型: {args.init}")
    else:
        print(f"BC 热身: {args.bc_games} 局规则Bot数据")
        t0 = time.time()
        bc_data = generate_dataset(args.bc_games, workers=args.workers)
        print(f"  数据: {len(bc_data['acts'])} 样本, {time.time()-t0:.1f}s")
        train_bc(model, bc_data, device, epochs=args.bc_epochs)

    torch.save({"model": model.state_dict(), "size": args.size}, args.out)

    # 评估 BC 基线
    wr, avg = evaluate_vs_rule(args.out, games=args.eval_games)
    print(f"BC 基线: 胜率 {wr:.1%}, 场均 {avg:+.2f}")
    best_score = avg
    torch.save({"model": model.state_dict(), "size": args.size},
               args.out.replace(".pt", "_best.pt"))

    # ---- PPO 迭代 ----
    trainer = PPOTrainer(model, device, lr=args.lr)

    for it in range(args.iters):
        t0 = time.time()
        print(f"\n=== PPO 迭代 {it+1}/{args.iters} ===")

        # 1. 向量化自对弈收集数据
        print(f"  收集 {args.games_per_iter} 局自对弈...")
        data = collect_ppo_batch(model, args.games_per_iter, device,
                                 args.temperature,
                                 seed0=it * 10000,
                                 batch_games=args.batch_games)
        t1 = time.time()
        print(f"  样本: {len(data['acts'])}, 数据耗时 {t1-t0:.1f}s")

        # 2. PPO 更新
        trainer.update(data, epochs=args.ppo_epochs)
        t2 = time.time()
        print(f"  训练耗时 {t2-t1:.1f}s")

        # 3. 保存 + 评估
        torch.save({"model": model.state_dict(), "size": args.size}, args.out)
        wr, avg = evaluate_vs_rule(args.out, games=args.eval_games)
        t3 = time.time()
        print(f"  评估: 胜率 {wr:.1%}, 场均 {avg:+.2f} (耗时 {t3-t2:.1f}s)")

        if avg > best_score:
            best_score = avg
            torch.save({"model": model.state_dict(), "size": args.size},
                       args.out.replace(".pt", "_best.pt"))
            print(f"  ★ 新最优")

        print(f"  本轮总耗时: {t3-t0:.1f}s")

    print(f"\n完成。最优场均: {best_score:+.2f}")


if __name__ == "__main__":
    main()
