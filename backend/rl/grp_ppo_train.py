"""安康159 - GRP 奖励整形 PPO 训练

核心机制 (参考 Suphx 的 Global Reward Prediction):
- 每步奖励 r_t = GRP(s_{t+1})[seat] - GRP(s_t)[seat]  (势函数差, 策略不变性)
- 末端奖励 r_T = 真实得分 - GRP(s_T)[seat]           (残差, 含159翻牌噪声但每局仅一次)
- 望远镜求和后总回报 = 真实得分 - GRP(s_0), 与原始目标一致
- GRP 把局终稀疏噪声奖励转为逐步稠密低噪信号, 解决信用分配

流程: BC初始化 -> [向量化自对弈(vs规则Bot) + GRP打奖励 -> GAE -> PPO更新 -> 评估] x N
"""

import argparse
import os
import time

import numpy as np
import torch

from .model import build_model
from .grp import GRPModel
from .vec_selfplay import VectorizedSelfPlay
from .ppo import PPOTrainer


def compute_grp_gae(records, grp_vals, scores, gamma=1.0, lam=0.95):
    """用 GRP 势函数差作为逐步奖励, 计算 GAE

    records:  [(seat, feat, act, lp, val, regret), ...] 本局模型决策点
    grp_vals: [v0, v1, ...] 与 records 平行, 决策玩家自己的 GRP 预测
    scores:   [s0..s3] 各座位最终得分
    """
    n = len(records)
    advs = np.zeros(n, dtype=np.float32)
    rets = np.zeros(n, dtype=np.float32)

    # 按座位分轨迹
    seat_steps = {}
    for i, rec in enumerate(records):
        seat_steps.setdefault(rec[0], []).append(i)

    for seat, idxs in seat_steps.items():
        T = len(idxs)
        vals = [records[i][4] for i in idxs]      # 网络 value 估计
        gvs = [grp_vals[i] for i in idxs]          # GRP 预测

        # 逐步 GRP 势函数差奖励 + 末端残差
        rews = [gvs[t + 1] - gvs[t] for t in range(T - 1)]
        rews.append(scores[seat] - gvs[-1])

        last_gae = 0.0
        for t in reversed(range(T)):
            next_val = vals[t + 1] if t < T - 1 else 0.0
            delta = rews[t] + gamma * next_val - vals[t]
            last_gae = delta + gamma * lam * last_gae
            advs[idxs[t]] = last_gae
            rets[idxs[t]] = last_gae + vals[t]

    return advs, rets


@torch.no_grad()
def evaluate_vec(model, device, n_games=1000, seed0=900000):
    """向量化评估: 模型座位0 vs 3规则Bot, 贪心出牌"""
    model.eval()
    engine = VectorizedSelfPlay(model, n_games, device, seed0=seed0,
                                model_seats=[0])
    results = engine.run(temperature=0.0)
    wins = sum(1 for r in results if r["winner"] == 0)
    avg = float(np.mean([r["scores"][0] for r in results]))
    model.train()
    return wins / len(results), avg


def collect_batch(model, grp, n_games, device, temperature, seed0,
                  grp_version=2):
    engine = VectorizedSelfPlay(model, n_games, device, seed0=seed0,
                                model_seats=[0], grp_model=grp,
                                grp_version=grp_version)
    results = engine.run(temperature=temperature)

    all_feats, all_acts, all_lps, all_advs, all_rets, all_masks = \
        [], [], [], [], [], []
    game_advs = []
    for r in results:
        if not r["records"]:
            continue
        advs, rets = compute_grp_gae(r["records"], r["grp_vals"], r["scores"])
        game_advs.append(float(np.mean(np.abs(advs))))
        for j, rec in enumerate(r["records"]):
            seat, feat, act, lp, val, regret, mask = rec
            all_feats.append(feat)
            all_acts.append(act)
            all_lps.append(lp)
            all_advs.append(advs[j])
            all_rets.append(rets[j])
            all_masks.append(mask)

    data = {
        "feats": np.stack(all_feats).astype(np.float32),
        "acts": np.asarray(all_acts, dtype=np.int64),
        "old_lps": np.asarray(all_lps, dtype=np.float32),
        "advs": np.asarray(all_advs, dtype=np.float32),
        "rets": np.asarray(all_rets, dtype=np.float32),
        "masks": np.stack(all_masks).astype(bool),
    }
    return data, float(np.mean(game_advs))


def _collect_worker(args):
    """spawn 子进程: 独立跑一片向量化自对弈 (避免 CUDA fork 问题)"""
    (model_sd, grp_sd, grp_cfg, size, n_games, seed0, temperature,
     device_id, grp_version) = args
    import torch as _torch
    # CUDA_VISIBLE_DEVICES 下实际可见卡数可能 < 8, 取模防越界
    if device_id >= 0 and _torch.cuda.is_available():
        device_id = device_id % _torch.cuda.device_count()
        device = _torch.device(f"cuda:{device_id}")
    else:
        device = _torch.device("cpu")
    from .model import build_model
    model = build_model(size).to(device)
    model.load_state_dict(model_sd)
    model.eval()
    if grp_version == 3:
        from .grp_train3 import GRP3Model
        grp = GRP3Model(*grp_cfg).to(device)
    else:
        grp = GRPModel(*grp_cfg).to(device)
    grp.load_state_dict(grp_sd)
    grp.eval()
    for p in grp.parameters():
        p.requires_grad_(False)

    engine = VectorizedSelfPlay(model, n_games, device, seed0=seed0,
                                model_seats=[0], grp_model=grp,
                                grp_version=grp_version)
    return engine.run(temperature=temperature)


def collect_batch_parallel(model, grp, n_games, device, temperature, seed0,
                           n_procs=8, size="base",
                           grp_cfg=(512, 6), grp_version=2):
    """多进程并行收集: 每进程一片向量化自对弈 (CPU编码是瓶颈, 进程级并行)"""
    import multiprocessing as mp
    model_sd = {k: v.cpu() for k, v in model.state_dict().items()}
    grp_sd = {k: v.cpu() for k, v in grp.state_dict().items()}

    per = n_games // n_procs
    tasks = []
    for w in range(n_procs):
        dev_id = w % 8 if torch.cuda.is_available() else -1
        tasks.append((model_sd, grp_sd, tuple(grp_cfg), size,
                      per, seed0 + w * per, temperature, dev_id,
                      grp_version))

    ctx = mp.get_context("spawn")
    with ctx.Pool(n_procs) as pool:
        all_results = []
        for res in pool.imap_unordered(_collect_worker, tasks):
            all_results.extend(res)

    return _merge_results(all_results)


def _merge_results(results):
    all_feats, all_acts, all_lps, all_advs, all_rets, all_masks = \
        [], [], [], [], [], []
    game_advs = []
    for r in results:
        if not r["records"]:
            continue
        advs, rets = compute_grp_gae(r["records"], r["grp_vals"], r["scores"])
        game_advs.append(float(np.mean(np.abs(advs))))
        for j, rec in enumerate(r["records"]):
            seat, feat, act, lp, val, regret, mask = rec
            all_feats.append(feat)
            all_acts.append(act)
            all_lps.append(lp)
            all_advs.append(advs[j])
            all_rets.append(rets[j])
            all_masks.append(mask)
    data = {
        "feats": np.stack(all_feats).astype(np.float32),
        "acts": np.asarray(all_acts, dtype=np.int64),
        "old_lps": np.asarray(all_lps, dtype=np.float32),
        "advs": np.asarray(all_advs, dtype=np.float32),
        "rets": np.asarray(all_rets, dtype=np.float32),
        "masks": np.stack(all_masks).astype(bool),
    }
    return data, float(np.mean(game_advs))


def pretrain_value_head(model, device, data_path, epochs=5, lr=1e-3):
    """BC 从未训练 value head —— 用 GRP 数据(决策者本人最终得分)先拟合它。
    目标按 batch 统计归一化, 与 PPO 更新中的 rets_n 归一化保持一致的 scale。"""
    import torch.nn.functional as F
    z = np.load(data_path)
    feats = torch.from_numpy(z["feats"]).to(device)
    seats = torch.from_numpy(z["seats"]).to(device)
    scores = torch.from_numpy(z["scores"]).to(device)
    targets = scores[torch.arange(len(seats)), seats]
    t_mean, t_std = targets.mean(), targets.std() + 1e-8
    targets_n = (targets - t_mean) / t_std

    # 冻结 value_head 以外的所有参数
    for name, p in model.named_parameters():
        p.requires_grad_(name.startswith("value_head"))
    opt = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=lr)
    n = len(feats)
    for ep in range(epochs):
        perm = torch.randperm(n, device=device)
        tot, nb = 0.0, 0
        for i in range(0, n, 4096):
            idx = perm[i:i + 4096]
            _, v = model(feats[idx])
            loss = F.smooth_l1_loss(v, targets_n[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item()
            nb += 1
        print(f"  value 预训练 ep {ep+1}/{epochs} huber={tot/nb:.4f}")
    for p in model.parameters():
        p.requires_grad_(True)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=str, default="base")
    ap.add_argument("--init", type=str, required=True)
    ap.add_argument("--grp", type=str, default="models/grp_v1.pt")
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--games-per-iter", type=int, default=2048)
    ap.add_argument("--ppo-epochs", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=0.5)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--entropy-coef", type=float, default=0.01)
    ap.add_argument("--eval-games", type=int, default=1000)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--value-data", type=str, default="models/grp_data.npz")
    ap.add_argument("--value-epochs", type=int, default=5)
    ap.add_argument("--procs", type=int, default=1,
                    help="数据收集进程数, >1 时用 spawn 多进程+多GPU")
    ap.add_argument("--out", type=str, default="models/grp_ppo.pt")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(args.size).to(device)
    ckpt = torch.load(args.init, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
    print(f"初始化: {args.init}, 参数量 "
          f"{sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    if args.value_epochs > 0 and os.path.exists(args.value_data):
        print("value head 预训练 (BC 检查点未训练过 value head):")
        pretrain_value_head(model, device, args.value_data,
                            epochs=args.value_epochs)

    grp_ckpt = torch.load(args.grp, map_location=device, weights_only=False)
    grp_version = grp_ckpt.get("version", 2)
    if grp_version == 3:
        from .grp_train3 import GRP3Model
        grp = GRP3Model(grp_ckpt.get("hidden", 512),
                        grp_ckpt.get("blocks", 8)).to(device)
    else:
        grp = GRPModel(grp_ckpt.get("hidden", 512),
                       grp_ckpt.get("blocks", 6)).to(device)
    grp.load_state_dict(grp_ckpt["state_dict"])
    grp.eval()
    for p in grp.parameters():
        p.requires_grad_(False)
    print(f"GRP: {args.grp} (v{grp_version})")

    wr, avg = evaluate_vec(model, device, args.eval_games)
    print(f"初始评估: 胜率 {wr:.1%}, 场均 {avg:+.2f}")
    best = avg
    torch.save({"model": model.state_dict(), "size": args.size},
               args.out.replace(".pt", "_best.pt"))

    trainer = PPOTrainer(model, device, lr=args.lr,
                         entropy_coef=args.entropy_coef)

    for it in range(args.iters):
        t0 = time.time()
        model.train()
        if args.procs > 1:
            grp_cfg = (grp_ckpt.get("hidden", 512),
                       grp_ckpt.get("blocks", 6 if grp_version == 2 else 8))
            data, mean_abs_adv = collect_batch_parallel(
                model, grp, args.games_per_iter, device, args.temperature,
                seed0=it * 100000, n_procs=args.procs, size=args.size,
                grp_cfg=grp_cfg, grp_version=grp_version)
        else:
            data, mean_abs_adv = collect_batch(
                model, grp, args.games_per_iter, device, args.temperature,
                seed0=it * 100000, grp_version=grp_version)
        t1 = time.time()

        trainer.update(data, epochs=args.ppo_epochs)
        t2 = time.time()

        print(f"[iter {it+1}] 样本 {len(data['acts'])}, "
              f"|adv|={mean_abs_adv:.3f}, "
              f"收集 {t1-t0:.0f}s, 训练 {t2-t1:.0f}s")

        if (it + 1) % args.eval_every == 0 or it == 0:
            wr, avg = evaluate_vec(model, device, args.eval_games,
                                   seed0=900000 + (it + 1))
            tag = ""
            if avg > best:
                best = avg
                torch.save({"model": model.state_dict(), "size": args.size},
                           args.out.replace(".pt", "_best.pt"))
                tag = " ★新最优"
            print(f"[iter {it+1}] 评估: 胜率 {wr:.1%}, 场均 {avg:+.2f}{tag}")

        torch.save({"model": model.state_dict(), "size": args.size}, args.out)

    print(f"完成。最优场均 {best:+.2f}")


if __name__ == "__main__":
    main()
