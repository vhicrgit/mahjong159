"""安康159 - 在线迭代训练(简化版 Mortal 在线 RL 管线)

流程(每一轮 iteration):
  1. 用当前模型(带温度采样)自对弈生成数据 -> 保证探索性
  2. 与历史数据合并(回放缓冲, 防止灾难性遗忘)
  3. AWR 更新模型
  4. 评估 vs 规则Bot, 保存最优

用法(小规模验证):
  python -m backend.rl.iter_train --size tiny --iters 5 --games-per-iter 500
H20 正式训练建议:
  python -m backend.rl.iter_train --size base --iters 100 --games-per-iter 5000 \
      --workers 32 --out model_base.pt
"""

import argparse
import os
import numpy as np
import torch

from .model import build_model
from .selfplay import generate_dataset
from .train import get_device, train_bc, train_awr
from .evaluate import play_eval_game


def evaluate_vs_rule(model_path, games=100):
    wins, total = 0, 0.0
    for i in range(games):
        sd, won = play_eval_game(200000 + i, model_path)
        total += sd
        wins += won
    return wins / games, total / games


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=str, default="tiny")
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--games-per-iter", type=int, default=500)
    ap.add_argument("--bc-epochs", type=int, default=10)
    ap.add_argument("--awr-epochs", type=int, default=3)
    ap.add_argument("--buffer", type=int, default=200000,
                    help="回放缓冲最大样本数")
    ap.add_argument("--init", type=str, default="",
                    help="初始模型路径(空则先BC热身)")
    ap.add_argument("--out", type=str, default="model_iter.pt")
    ap.add_argument("--eval-games", type=int, default=100)
    args = ap.parse_args()

    device = get_device()
    print(f"设备: {device}, 模型: {args.size}, 迭代 {args.iters} 轮")

    # 初始模型: 空则从规则Bot数据BC热身
    model = build_model(args.size).to(device)
    replay = {"feats": [], "acts": [], "rets": []}

    if args.init and os.path.exists(args.init):
        ckpt = torch.load(args.init, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        print(f"加载初始模型: {args.init}")
    else:
        print("冷启动: 先用规则Bot数据做BC热身")
        bc_data = generate_dataset(args.games_per_iter)
        train_bc(model, bc_data, device, epochs=args.bc_epochs)
        for k in replay:
            replay[k].append(bc_data[k])

    torch.save({"model": model.state_dict(), "size": args.size}, args.out)
    best_score = -1e9

    for it in range(args.iters):
        print(f"\n=== 迭代 {it + 1}/{args.iters} ===")
        # 1. 当前模型自对弈(带探索温度)
        print(f"  生成 {args.games_per_iter} 局(模型温度采样)...")
        from .selfplay import _net_chooser_factory
        chooser = _net_chooser_factory(model, temperature=1.0)
        new_data = generate_dataset(args.games_per_iter,
                                    seed0=500000 + it * 10000,
                                    chooser=chooser)
        for k in replay:
            replay[k].append(new_data[k])
        # 合并 + 截断
        merged = {}
        for k in replay:
            merged[k] = np.concatenate(replay[k])
            if len(merged[k]) > args.buffer:
                merged[k] = merged[k][-args.buffer:]
        data = merged
        print(f"  回放池样本: {len(data['acts'])}")

        # 2. AWR 更新
        train_awr(model, data, device, epochs=args.awr_epochs)
        torch.save({"model": model.state_dict(), "size": args.size}, args.out)

        # 3. 评估
        wr, avg = evaluate_vs_rule(args.out, games=args.eval_games)
        print(f"  评估: 胜率 {wr:.1%}, 场均 {avg:+.2f}")
        if avg > best_score:
            best_score = avg
            torch.save({"model": model.state_dict(), "size": args.size},
                       args.out.replace(".pt", "_best.pt"))
            print(f"  新最优, 已保存 {args.out.replace('.pt', '_best.pt')}")

    print(f"\n完成。最优场均得分: {best_score:+.2f}")


if __name__ == "__main__":
    main()
