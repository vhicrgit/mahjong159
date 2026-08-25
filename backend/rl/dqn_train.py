"""安康159 - DQN+CQL 迭代训练

Mortal 路线:
- Q(s,a) 值函数, MC return (gamma=1), 不用 TD bootstrap
- CQL 保守惩罚: 防止分布外动作 Q 值过估计
- Boltzmann 低温探索 (T=0.05)
- 1v3 评估 (模型 vs 3规则Bot)

用法:
  python -m backend.rl.dqn_train --size small --iters 50
"""

import argparse
import os
import time
import numpy as np
import torch
import torch.nn.functional as F

from .model import build_model, legal_discard_mask, N_ACTIONS
from .selfplay import generate_dataset
from .vec_selfplay import VectorizedSelfPlay
from .train import get_device
from .evaluate import play_eval_game


def dqn_cql_update(model, data, device, epochs=3, batch_size=1024,
                   lr=1e-4, cql_weight=1.0, value_coef=0.5):
    """DQN + CQL 更新

    L = MSE(Q(s,a), R) + cql_weight * (logsumexp(Q(s,·)) - Q(s,a))
        + value_coef * MSE(V(s), R)
    """
    feats = torch.from_numpy(data["feats"]).to(device)
    acts = torch.from_numpy(data["acts"]).to(device)
    rets = torch.from_numpy(data["rets"]).to(device)
    n = len(acts)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for ep in range(epochs):
        perm = torch.randperm(n, device=device)
        total_dqn, total_cql, total_v = 0.0, 0.0, 0.0
        nb = 0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            q_all, v = model(feats[idx])  # (B, 28), (B,)

            # DQN loss: MSE(Q(s,a), R)
            q_a = q_all.gather(1, acts[idx].unsqueeze(1)).squeeze(1)
            dqn_loss = F.mse_loss(q_a, rets[idx])

            # CQL loss: logsumexp(Q(s,·)) - Q(s,a)
            # 惩罚所有动作的Q值, 特别是数据中没有的动作
            cql_loss = (torch.logsumexp(q_all, dim=-1) - q_a).mean()

            # Value loss
            v_loss = F.mse_loss(v, rets[idx])

            loss = dqn_loss + cql_weight * cql_loss + value_coef * v_loss
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            total_dqn += dqn_loss.item()
            total_cql += cql_loss.item()
            total_v += v_loss.item()
            nb += 1
        print(f"  DQN ep {ep+1}/{epochs}  "
              f"dqn={total_dqn/nb:.4f}  cql={total_cql/nb:.4f}  "
              f"v={total_v/nb:.4f}")


def collect_selfplay_q(model, n_games, device, temperature=0.05,
                       seed0=0, batch_games=256):
    """用 Q 网络自对弈收集数据 (Boltzmann 探索)"""
    engine = VectorizedSelfPlay(model, n_games, device, seed0=seed0,
                                 model_seats=[0, 1, 2, 3])
    results = engine.run(temperature=temperature)

    all_feats, all_acts, all_rets = [], [], []
    for r in results:
        for seat, feat, act, lp, val in r["records"]:
            all_feats.append(feat)
            all_acts.append(act)
            all_rets.append(float(r["scores"][seat]))
    return {
        "feats": np.stack(all_feats).astype(np.float32),
        "acts": np.asarray(all_acts, dtype=np.int64),
        "rets": np.asarray(all_rets, dtype=np.float32),
    }


def evaluate_vs_rule(model_path, games=100):
    wins, total = 0, 0.0
    for i in range(games):
        sd, won = play_eval_game(200000 + i, model_path)
        total += sd
        wins += won
    return wins / games, total / games


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=str, default="small")
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--games-per-iter", type=int, default=512)
    ap.add_argument("--bc-games", type=int, default=2000)
    ap.add_argument("--bc-epochs", type=int, default=20)
    ap.add_argument("--dqn-epochs", type=int, default=3)
    ap.add_argument("--batch-games", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.05)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--cql-weight", type=float, default=1.0)
    ap.add_argument("--init", type=str, default="")
    ap.add_argument("--out", type=str, default="model_dqn.pt")
    ap.add_argument("--eval-games", type=int, default=100)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    device = get_device()
    print(f"DQN+CQL 训练: {args.size} 模型, {device}, "
          f"温度={args.temperature}, {args.iters} 轮")

    model = build_model(args.size).to(device)
    print(f"参数量: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    # ---- BC 热身 (用规则Bot数据训练 Q 函数) ----
    if args.init and os.path.exists(args.init):
        ckpt = torch.load(args.init, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        print(f"加载初始模型: {args.init}")
    else:
        print(f"BC 热身: {args.bc_games} 局规则Bot数据")
        t0 = time.time()
        bc_data = generate_dataset(args.bc_games, workers=args.workers)
        print(f"  数据: {len(bc_data['acts'])} 样本, {time.time()-t0:.1f}s")

        # BC: 把规则Bot的选择当作最优动作, 用 Q 学习拟合
        # Q(s, a_rule) → R (局终得分)
        # 同时让其他动作的 Q 值更低
        dqn_cql_update(model, bc_data, device, epochs=args.bc_epochs,
                       lr=args.lr, cql_weight=args.cql_weight)

    torch.save({"model": model.state_dict(), "size": args.size}, args.out)

    # 评估 BC 基线
    wr, avg = evaluate_vs_rule(args.out, games=args.eval_games)
    print(f"BC 基线: 胜率 {wr:.1%}, 场均 {avg:+.2f}")
    best_score = avg
    torch.save({"model": model.state_dict(), "size": args.size},
               args.out.replace(".pt", "_best.pt"))

    # ---- DQN 迭代 ----
    for it in range(args.iters):
        t0 = time.time()
        print(f"\n=== DQN 迭代 {it+1}/{args.iters} ===")

        # 1. 向量化自对弈 (Boltzmann 低温探索)
        print(f"  收集 {args.games_per_iter} 局...")
        data = collect_selfplay_q(model, args.games_per_iter, device,
                                   args.temperature, seed0=it * 10000,
                                   batch_games=args.batch_games)
        t1 = time.time()
        print(f"  样本: {len(data['acts'])}, 收集耗时 {t1-t0:.1f}s")

        # 2. DQN+CQL 更新
        dqn_cql_update(model, data, device, epochs=args.dqn_epochs,
                       lr=args.lr, cql_weight=args.cql_weight)
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
