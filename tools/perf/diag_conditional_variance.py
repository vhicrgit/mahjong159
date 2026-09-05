"""条件方差分解: 冻结可见历史, 抽多个隐藏世界, 量化"特权 critic"能消掉多少噪声。

为什么要测这个:
  文档里"回报方差 87~90% 是不可约摸牌运气、完美 critic 的 R² 天花板≈0.10"是用
  开局向听/中盘 E 等少量特征做回归得到的 —— 那测的是**这几个预测量**的解释力,
  不是完整历史条件下的方差(Codex 审计 §5 的批评成立)。本脚本用正确口径重测:

    同一个可见局面 → 采样 W 个牌数相容的隐藏世界 → 每个世界用同一策略跑 K 局

  三层嵌套分解(每层都是无偏的方差成分):
    Var_visible   状态之间 E[R|可见局面] 的方差   —— 盲打 critic 能学的部分
    Var_world     同一可见局面下, 世界之间 E[R|世界] 的方差
                                                     —— 特权 critic 额外能学的部分
    Var_policy    同一世界内, 策略采样造成的方差   —— 不可约(除非降温)

  为什么"特权 critic"合法: baseline 只要不依赖动作就不改变策略梯度的期望,
  所以 critic 可以看对手手牌与牌墙(非对称 actor-critic)。若 Var_world 占比大,
  就说明我们此前 RL 的 advantage 噪声主要来自"这一副牌墙本身好不好",
  而不是"这个动作好不好" —— 那正是 PPO 反复中性的机制解释。

口径提醒: sample_world 是**牌数相容的均匀重洗**, 不是按对手历史校准的后验;
真实对手的手牌与其弃牌/鸣牌相关, 所以这里的 Var_world 偏保守还是偏大取决于
历史的信息量, 不能直接当成部署收益。它回答的是"信息值多少方差", 不是"能赢多少分"。

用法:
  python -m tools.perf.diag_conditional_variance --states 256 --worlds 32 --rolls 4
"""

import argparse
import copy
import json
import os
import time

import numpy as np
import torch

from backend.ai.bot_native import NativeV31
from backend.game.engine import Game
from backend.rl import cf_collect
from backend.rl.hidden_worlds import sample_world
from backend.rl.model import build_model


def load_model(path, device):
    ck = torch.load(path, map_location="cpu", weights_only=True)
    m = build_model(ck["size"], feat_dim=ck.get("feat_dim", 628))
    m.load_state_dict(ck["model"])
    m.eval()
    return m.to(device), ck


def collect_states(model, device, n_games, seed0, want, rng):
    """用被测策略自对弈, 对每个弃牌决策点留快照, 再随机下采样到 want 个。

    全采后下采样(而不是按概率采)保证对整局各阶段是均匀覆盖的 —— 旧的 GRPO
    采集器就因为只取整局前 13 次弃牌, 让世界推演全服务于前盘状态。
    """
    n_games = max(n_games, (want * 4) // 45 + 1)
    games = [Game(seed=seed0 + i, human_seat=-1) for i in range(n_games)]
    t0 = time.time()
    games, snaps = cf_collect.run_games(model, device, games, temp=1.0,
                                        snap_p=1.0, rng=rng)
    snaps = [s for s in snaps if not s["game"].bloody
             and s["game"].phase == "discard_wait"
             and s["game"].turn == s["seat"]]
    print(f"自对弈 {n_games} 局 -> 决策点 {len(snaps)} 个 "
          f"({time.time() - t0:.1f}s)", flush=True)
    if len(snaps) > want:
        snaps = [snaps[i] for i in
                 sorted(rng.choice(len(snaps), want, replace=False).tolist())]
    return snaps


def blind_r2(feats, target, groups, alpha=10.0, seed=0):
    """盲打 critic 的现实 R²(不是天花板): 628 维特征 -> E[R|可见局面]。

    用岭回归闭式解而不是 MLP: 状态只有几百个而特征 628 维, 神经网络在这个
    样本量下会直接发散(实测 R² 掉到 -1e7 量级), 数字没有意义。
    按整局划分训练/验证, 避免同局状态跨集合泄漏。
    """
    uniq = np.unique(groups)
    order = np.random.RandomState(seed).permutation(len(uniq))
    hold = set(uniq[order[:max(2, len(uniq) // 5)]].tolist())
    va = np.array([g in hold for g in groups])
    tr = ~va
    x = feats.astype(np.float64)
    y = target.astype(np.float64)
    mu, sd = x[tr].mean(0), x[tr].std(0)
    sd[sd < 1e-12] = 1.0            # 常数特征不缩放, 否则被放大成 1e6
    x = (x - mu) / sd
    y = y - y[tr].mean()
    a = x[tr]
    w = np.linalg.solve(a.T @ a + alpha * np.eye(a.shape[1]), a.T @ y[tr])
    resid = float(np.mean((x[va] @ w - y[va]) ** 2))
    var = float(np.var(y[va], ddof=1))
    return {"r2": 1.0 - resid / var, "val_n": int(va.sum()),
            "val_mse": resid, "val_var": var, "alpha": alpha}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/acnn_latest_best.pt")
    ap.add_argument("--states", type=int, default=256)
    ap.add_argument("--worlds", type=int, default=32)
    ap.add_argument("--rolls", type=int, default=4)
    ap.add_argument("--games", type=int, default=60)
    ap.add_argument("--seed0", type=int, default=206200000)
    ap.add_argument("--seed", type=int, default=20260905)
    ap.add_argument("--rollout-temp", type=float, default=1.0,
                    help="续盘策略温度; 0 = 确定性(此时 Var_policy 只来自世界)")
    ap.add_argument("--procs", type=int, default=1)
    ap.add_argument("--out", default="logs/diag_conditional_variance.json")
    args = ap.parse_args()

    torch.set_num_threads(1)
    device = "cpu"
    rng = np.random.default_rng(args.seed)
    model, ck = load_model(args.model, device)
    print(f"模型 {args.model} (size={ck['size']}, "
          f"feat_dim={ck.get('feat_dim', 628)})")

    snaps = collect_states(model, device, args.games, args.seed0, args.states,
                           np.random.RandomState(args.seed))
    if len(snaps) < 20:
        raise SystemExit("快照太少, 加大 --games")

    feats = np.stack([s["feat"] for s in snaps])   # 采集时已存好盲打特征
    groups = np.array([s["gi"] for s in snaps])

    W, K = args.worlds, args.rolls
    R = np.zeros((len(snaps), W, K), dtype=np.float64)
    t0 = time.time()
    for i, s in enumerate(snaps):
        worlds = []
        for w in range(W):
            base = sample_world(s["game"], s["seat"], rng)
            for _ in range(K):
                worlds.append((copy.deepcopy(base), s["seat"]))
        games = [g for g, _ in worlds]
        seats = [st for _, st in worlds]
        games, _ = cf_collect.run_games(model, device, games,
                                        temp=args.rollout_temp,
                                        rng=np.random.RandomState(
                                            args.seed + 1000 + i))
        for j, g in enumerate(games):
            R[i, j // K, j % K] = cf_collect.default_reward(g, seats[j])
        if (i + 1) % 16 == 0:
            el = time.time() - t0
            print(f"  {i + 1}/{len(snaps)} 状态, {el:.0f}s "
                  f"({el / (i + 1) * (len(snaps) - i - 1):.0f}s 剩余)", flush=True)

    per_state_mean = R.mean(axis=(1, 2))          # E[R | 可见局面]
    world_mean = R.mean(axis=2)                   # E[R | 世界]
    var_visible = float(per_state_mean.var(ddof=1))
    var_world = float(world_mean.var(axis=1, ddof=1).mean())
    within = R.var(axis=2, ddof=1) if K > 1 else np.zeros((len(snaps), W))
    var_policy = float(np.nanmean(within))
    total = var_visible + var_world + var_policy

    blind = blind_r2(feats, per_state_mean, groups)

    out = {
        "model": args.model, "args": vars(args),
        "states": len(snaps), "worlds": W, "rolls": K,
        "rollout_temp": args.rollout_temp,
        "seconds": time.time() - t0,
        "reward": "rank_rewards + 0.25*自己杠分 (cf_collect.default_reward)",
        "variance_components": {
            "var_visible_state": var_visible,
            "var_hidden_world": var_world,
            "var_policy_sampling": var_policy,
            "total": total,
            "share_visible": var_visible / total,
            "share_world": var_world / total,
            "share_policy": var_policy / total,
        },
        "critic_r2": {
            # 分母是"状态间 E[R|可见局面] 的方差"(回归目标自身的方差)
            "blind_fitted_r2_on_visible_mean": blind["r2"],
            # 分母是总方差: 盲打 critic 最多能解释的份额
            "blind_ceiling_share_of_total": var_visible / total,
            # 换算到总方差口径的实际解释份额 = 上面两项相乘
            "blind_effective_share_of_total":
                blind["r2"] * var_visible / total,
            "privileged_ceiling_share_of_total":
                (var_visible + var_world) / total,
            "residual_share_after_privileged": var_policy / total,
        },
        "advantage_noise_reduction_if_privileged":
            (float(np.sqrt(total / var_policy))
             if var_policy > 1e-6 * total else None),
        "per_state_mean_return": {
            "mean": float(per_state_mean.mean()),
            "sd": float(per_state_mean.std(ddof=1)),
        },
        "world_mean_sd_within_state": float(np.sqrt(var_world)),
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)

    v = out["variance_components"]
    print("\n=== 方差分解(每个弃牌决策点的终局回报) ===")
    print(f"  可见局面之间   {var_visible:.4f}  ({v['share_visible']:.1%})"
          f"   <- 盲打 critic 可学")
    print(f"  隐藏世界之间   {var_world:.4f}  ({v['share_world']:.1%})"
          f"   <- 特权 critic 额外可学")
    print(f"  策略采样之内   {var_policy:.4f}  ({v['share_policy']:.1%})"
          f"   <- 不可约(降温可减)")
    print(f"  合计           {total:.4f}")
    c = out["critic_r2"]
    print(f"\n盲打 critic: 拟合 R²={c['blind_fitted_r2_on_visible_mean']:.4f}"
          f"(分母=状态间可见均值方差) -> 折算到总方差 "
          f"{c['blind_effective_share_of_total']:.1%}")
    print(f"盲打天花板(总方差份额) {c['blind_ceiling_share_of_total']:.1%}; "
          f"特权天花板 {c['privileged_ceiling_share_of_total']:.1%}; "
          f"特权后残差 {c['residual_share_after_privileged']:.1%}")
    red = out["advantage_noise_reduction_if_privileged"]
    if red is None:
        print("续盘确定性(temp=0): 策略采样方差为 0, 特权 critic 可解释全部方差, "
              "剩余不确定性只来自选了哪个隐藏世界")
    else:
        print(f"若达到特权天花板, advantage 噪声可降到 1/{red:.2f}")
    print(f"已写 {args.out}")


if __name__ == "__main__":
    main()
