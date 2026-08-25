"""安康159 - 奖励整形 PPO 训练

核心思路:
- BC 热身提供强基线 (24.5% 胜率, 持平规则Bot)
- PPO 微调, 奖励 = 终局得分 + α * (-regret)
- regret = 规则Bot评分(最优) - 规则Bot评分(所选), 衡量偏离程度
- 低温度 Boltzmann 探索 (T=0.05)

用法:
  python -m backend.rl.shaped_train --init models/bc_base_50k.pt --iters 30
"""

import argparse
import os
import time
import numpy as np
import torch
import torch.nn.functional as F

from .model import build_model
from .vec_selfplay import VectorizedSelfPlay
from .train import get_device
from .evaluate import play_eval_game


def compute_returns(records, scores, gamma=0.99, regret_weight=0.01):
    """计算每个决策点的回报: regret整形 + 终局得分

    records: [(seat, feat, act, log_prob, value, regret), ...]
    scores: [score_0, score_1, score_2, score_3]

    返回: (advantages, returns)
    """
    # 按座位分组
    seat_steps = {}
    for i, (seat, *_rest) in enumerate(records):
        seat_steps.setdefault(seat, []).append(i)

    advantages = np.zeros(len(records), dtype=np.float32)
    returns = np.zeros(len(records), dtype=np.float32)

    for seat, indices in seat_steps.items():
        R_final = scores[seat]
        T = len(indices)

        # 构建每步奖励: -regret * weight (终局时加上终局得分)
        rewards = []
        for t, idx in enumerate(indices):
            r = -records[idx][5] * regret_weight  # regret penalty
            if t == T - 1:
                r += R_final  # 终局奖励
            rewards.append(r)

        # GAE
        vals = [records[i][4] for i in indices]
        last_gae = 0.0
        for t in reversed(range(T)):
            next_val = vals[t + 1] if t < T - 1 else 0.0
            delta = rewards[t] + gamma * next_val - vals[t]
            last_gae = delta + gamma * 0.95 * last_gae
            advantages[indices[t]] = last_gae
            returns[indices[t]] = last_gae + vals[t]

    return advantages, returns


def ppo_update(model, data, device, epochs=4, batch_size=1024,
               lr=3e-4, clip_eps=0.2, value_coef=0.5, entropy_coef=0.01):
    """PPO 更新"""
    feats = torch.from_numpy(data["feats"]).to(device)
    acts = torch.from_numpy(data["acts"]).to(device)
    old_lps = torch.from_numpy(data["old_lps"]).to(device)
    advs = torch.from_numpy(data["advs"]).to(device)
    rets = torch.from_numpy(data["rets"]).to(device)

    advs = (advs - advs.mean()) / (advs.std() + 1e-8)

    n = len(acts)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    for ep in range(epochs):
        perm = torch.randperm(n, device=device)
        total_pl, total_vl, total_ent = 0.0, 0.0, 0.0
        nb = 0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            q_values, values = model(feats[idx])

            # Policy loss (PPO clip on Q-values treated as logits)
            logp = F.log_softmax(q_values, dim=-1)
            new_lp = logp.gather(1, acts[idx].unsqueeze(1)).squeeze(1)
            ratio = torch.exp(new_lp - old_lps[idx])
            surr1 = ratio * advs[idx]
            surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advs[idx]
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value loss
            value_loss = F.mse_loss(values, rets[idx])

            # Entropy bonus
            probs = torch.softmax(q_values, dim=-1)
            entropy = -(probs * logp).sum(-1).mean()

            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            opt.step()

            total_pl += policy_loss.item()
            total_vl += value_loss.item()
            total_ent += entropy.item()
            nb += 1
        print(f"  PPO ep {ep+1}/{epochs}  "
              f"pi={total_pl/nb:.4f}  v={total_vl/nb:.4f}  "
              f"ent={total_ent/nb:.3f}")


def evaluate_vs_rule(model_path, games=100):
    wins, total = 0, 0.0
    for i in range(games):
        sd, won = play_eval_game(200000 + i, model_path)
        total += sd
        wins += won
    return wins / games, total / games


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", type=str, required=True,
                    help="BC 热身模型路径")
    ap.add_argument("--size", type=str, default="base")
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--games-per-iter", type=int, default=256)
    ap.add_argument("--batch-games", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.05)
    ap.add_argument("--regret-weight", type=float, default=0.01)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--ppo-epochs", type=int, default=4)
    ap.add_argument("--out", type=str, default="models/shaped_rl.pt")
    ap.add_argument("--eval-games", type=int, default=100)
    args = ap.parse_args()

    device = get_device()
    print(f"奖励整形 PPO: {args.size} 模型, {device}")
    print(f"  温度={args.temperature}, regret_weight={args.regret_weight}")

    model = build_model(args.size).to(device)
    ckpt = torch.load(args.init, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    print(f"加载 BC 模型: {args.init}")
    print(f"参数量: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    torch.save({"model": model.state_dict(), "size": args.size}, args.out)

    # 评估 BC 基线
    wr, avg = evaluate_vs_rule(args.out, games=args.eval_games)
    print(f"BC 基线: 胜率 {wr:.1%}, 场均 {avg:+.2f}")
    best_score = avg
    torch.save({"model": model.state_dict(), "size": args.size},
               args.out.replace(".pt", "_best.pt"))

    # PPO 迭代
    for it in range(args.iters):
        t0 = time.time()
        print(f"\n=== PPO 迭代 {it+1}/{args.iters} ===")

        # 1. 自对弈 (Boltzmann 低温探索)
        print(f"  收集 {args.games_per_iter} 局...")
        engine = VectorizedSelfPlay(model, args.games_per_iter, device,
                                     seed0=it * 10000,
                                     model_seats=[0, 1, 2, 3])
        results = engine.run(temperature=args.temperature)

        # 2. 计算回报 (regret整形 + 终局得分)
        all_feats, all_acts, all_lps, all_advs, all_rets = [], [], [], [], []
        for r in results:
            if not r["records"]:
                continue
            advs, rets = compute_returns(r["records"], r["scores"],
                                          regret_weight=args.regret_weight)
            for j, (seat, feat, act, lp, val, regret) in enumerate(r["records"]):
                all_feats.append(feat)
                all_acts.append(act)
                all_lps.append(lp)
                all_advs.append(advs[j])
                all_rets.append(rets[j])

        data = {
            "feats": np.stack(all_feats).astype(np.float32),
            "acts": np.asarray(all_acts, dtype=np.int64),
            "old_lps": np.asarray(all_lps, dtype=np.float32),
            "advs": np.asarray(all_advs, dtype=np.float32),
            "rets": np.asarray(all_rets, dtype=np.float32),
        }
        t1 = time.time()
        print(f"  样本: {len(data['acts'])}, 收集耗时 {t1-t0:.1f}s")

        # 3. PPO 更新
        ppo_update(model, data, device, epochs=args.ppo_epochs, lr=args.lr)
        t2 = time.time()
        print(f"  训练耗时 {t2-t1:.1f}s")

        # 4. 保存 + 评估
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
