"""安康159 - 离线 DQN + CQL (Mortal 路线, 修复版)

从规则Bot数据学 Q(s,a) -> MC return, Q-greedy 可隐式超越行为策略。
修复历史 DQN 失败的三个问题:
1. CQL logsumexp 施加在合法动作上 (历史版: 全28动作, 非法动作Q被无意义地惩罚)
2. 目标归一化 (历史版: 得分 scale ±20 的 MSE 梯度支配主干)
3. target network 稳定训练

用法:
  python -m backend.rl.dqn_offline --init models/bc_base_50k.pt \
      --data models/offline_data.npz --epochs 20 --out models/dqn_off_v1.pt
"""

import argparse
import time

import numpy as np
import torch
import torch.nn.functional as F

from .model import build_model
from .grp_ppo_train import evaluate_vec


def legal_mask_from_feats(feats: np.ndarray) -> np.ndarray:
    """从 v2 特征恢复合法动作掩码。
    手牌 one-hot 在 [0,112): tile t 的 4 维 = _oh4(count)。
    count>0 iff 4t+1/4t+2/4t+3 任一为 1 (_oh4(0)=[1,0,0,0])"""
    oh = feats[:, :112].reshape(len(feats), 28, 4)
    return (oh[:, :, 1:] > 0.5).any(axis=-1)  # (N, 28) bool


def train_offline_dqn(model, data, device, epochs=20, batch_size=16384,
                      lr=1e-4, cql_weight=0.5, value_coef=0.1,
                      target_sync=200, use_bf16=True):
    """bf16 autocast + 大 batch: H20 上 ~200万样本/s (fp32/bs2048 的 4.8x)"""
    feats = torch.from_numpy(data["feats"]).to(device)
    acts = torch.from_numpy(data["acts"]).to(device)
    rets = torch.from_numpy(data["rets"]).to(device).float()
    masks = torch.from_numpy(data["masks"]).to(device)

    # 全局归一化 MC return
    r_mean, r_std = rets.mean(), rets.std() + 1e-8
    rets_n = (rets - r_mean) / r_std

    import copy
    target = copy.deepcopy(model)
    target.eval()

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    amp_dtype = torch.bfloat16 if use_bf16 else None

    n = len(acts)
    steps = 0
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        tot_dqn, tot_cql, tot_v, nb = 0.0, 0.0, 0.0, 0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            with torch.autocast("cuda", dtype=amp_dtype,
                                enabled=use_bf16):
                q_all, v = model(feats[idx])

                neg_inf = torch.finfo(q_all.dtype).min
                q_masked = q_all.masked_fill(~masks[idx], neg_inf)

                q_a = q_all.gather(1, acts[idx].unsqueeze(1)).squeeze(1)
                dqn_loss = F.smooth_l1_loss(q_a.float(), rets_n[idx])

                # CQL: 压低未见过动作的 Q (只在合法动作上 logsumexp)
                logsumexp_legal = torch.logsumexp(
                    q_masked.float(), dim=-1)
                cql_loss = (logsumexp_legal - q_a.float()).mean()

                v_loss = F.smooth_l1_loss(v.float(), rets_n[idx])

                loss = dqn_loss + cql_weight * cql_loss \
                    + value_coef * v_loss
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            steps += 1
            if steps % target_sync == 0:
                target.load_state_dict(model.state_dict())

            tot_dqn += dqn_loss.item()
            tot_cql += cql_loss.item()
            tot_v += v_loss.item()
            nb += 1
        sched.step()
        print(f"  ep {ep+1}/{epochs}  dqn={tot_dqn/nb:.4f}  "
              f"cql={tot_cql/nb:.4f}  v={tot_v/nb:.4f}")
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", type=str, default="models/bc_base_50k.pt")
    ap.add_argument("--data", type=str, default="models/offline_data.npz")
    ap.add_argument("--size", type=str, default="base")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--cql-weight", type=float, default=0.5)
    ap.add_argument("--eval-games", type=int, default=1000)
    ap.add_argument("--out", type=str, default="models/dqn_off_v1.pt")
    ap.add_argument("--mid-eval-every", type=int, default=20,
                    help="每 N epochs 中间评估一次")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    z = np.load(args.data)
    feats, acts, rets = z["feats"], z["acts"], z["rets"]
    masks = legal_mask_from_feats(feats)
    # sanity: 动作必须合法
    assert masks[np.arange(len(acts)), acts].all(), "数据含非法动作!"
    print(f"数据: {len(acts)} 样本, 合法率验证通过")

    n = len(acts)
    idx = np.random.RandomState(0).permutation(n)
    n_val = min(100000, n // 20)
    val_idx, tr_idx = idx[:n_val], idx[n_val:]
    data = {"feats": feats[tr_idx], "acts": acts[tr_idx],
            "rets": rets[tr_idx], "masks": masks[tr_idx]}

    model = build_model(args.size).to(device)
    if args.init:
        ckpt = torch.load(args.init, map_location=device,
                          weights_only=False)
        if ckpt.get("size", args.size) != args.size:
            print(f"跳过初始化: {args.init} 是 {ckpt.get('size')} 尺寸, "
                  f"当前 {args.size} —— 随机初始化")
        else:
            model.load_state_dict(
                ckpt["model"] if "model" in ckpt else ckpt)
            print(f"初始化: {args.init}")
    else:
        print("随机初始化(从零训练)")
    print(f"{args.size}: "
          f"{sum(p.numel() for p in model.parameters())/1e6:.2f}M 参数")

    wr, avg = evaluate_vec(model, device, args.eval_games)
    print(f"训练前评估: 胜率 {wr:.1%}, 场均 {avg:+.2f}")
    best = avg
    torch.save({"model": model.state_dict(), "size": args.size},
               args.out.replace(".pt", "_best.pt"))

    # 分段训练 + 中间评估: 及时发现最优点, 防止过训退化
    epochs_done = 0
    while epochs_done < args.epochs:
        chunk = min(args.mid_eval_every,
                    args.epochs - epochs_done)
        train_offline_dqn(model, data, device, epochs=chunk,
                          cql_weight=args.cql_weight)
        epochs_done += chunk
        wr, avg = evaluate_vec(model, device, args.eval_games,
                               seed0=950000 + epochs_done)
        tag = ""
        if avg > best:
            best = avg
            tag = " ★"
        torch.save({"model": model.state_dict(), "size": args.size},
                   args.out)
        if tag:
            torch.save({"model": model.state_dict(), "size": args.size},
                       args.out.replace(".pt", "_best.pt"))
        print(f"[{epochs_done}ep] 评估: 胜率 {wr:.1%}, "
              f"场均 {avg:+.2f}{tag}")

    print(f"完成。最优场均 {best:+.2f}")


if __name__ == "__main__":
    main()
