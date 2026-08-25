"""安康159 - PPO 训练器

PPO (Proximal Policy Optimization) 替代 AWR:
- GAE (Generalized Advantage Estimation) 做信用分配
- Clip 目标函数防止大步更新破坏策略
- Value loss + entropy bonus

相比 AWR 的优势:
- GAE 让近期决策获得更准确的 advantage 估计
- Clip 防止策略突变
- Entropy bonus 保持探索
"""

import numpy as np
import torch
import torch.nn.functional as F


def compute_gae_per_trajectory(records, scores, gamma=0.99, lam=0.95):
    """对一局游戏的所有玩家轨迹计算 GAE

    records: [(seat, feat, act, log_prob, value), ...] 一局的所有决策点
    scores:  [score_0, score_1, score_2, score_3] 各座位最终得分

    返回: (advantages, returns) 每个决策点对应的优势和回报
    """
    # 按座位分组轨迹
    seat_steps = {}
    for i, (seat, feat, act, lp, val) in enumerate(records):
        seat_steps.setdefault(seat, []).append(i)

    advantages = np.zeros(len(records), dtype=np.float32)
    returns = np.zeros(len(records), dtype=np.float32)

    for seat, indices in seat_steps.items():
        R = scores[seat]
        T = len(indices)
        vals = [records[i][4] for i in indices]  # value estimates

        # rewards: 中间为0, 最后为R
        rews = [0.0] * (T - 1) + [R]

        # GAE
        last_gae = 0.0
        for t in reversed(range(T)):
            next_val = vals[t + 1] if t < T - 1 else 0.0
            delta = rews[t] + gamma * next_val - vals[t]
            last_gae = delta + gamma * lam * last_gae
            advantages[indices[t]] = last_gae
            returns[indices[t]] = last_gae + vals[t]

    return advantages, returns


class PPOTrainer:
    """PPO 训练器"""

    def __init__(self, model, device, lr=3e-4, clip_eps=0.2,
                 value_coef=0.1, entropy_coef=0.01):
        self.model = model
        self.device = device
        self.opt = torch.optim.Adam(model.parameters(), lr=lr)
        self.clip_eps = clip_eps
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef

    def update(self, data: dict, epochs: int = 4, batch_size: int = 1024):
        """PPO 更新

        data 字段:
          feats:     (N, D)   状态特征
          acts:      (N,)     动作
          old_lps:   (N,)     采样时 log π_old(a|s)
          advs:      (N,)     GAE 优势
          rets:      (N,)     回报目标 (adv + V)
          masks:     (N, 28)  合法动作掩码 (bool) — 必须与采样时一致,
                              否则 ratio 系统性偏移 + 熵泄漏到非法动作
        """
        feats = torch.from_numpy(data["feats"]).to(self.device)
        acts = torch.from_numpy(data["acts"]).to(self.device)
        old_lps = torch.from_numpy(data["old_lps"]).to(self.device)
        advs = torch.from_numpy(data["advs"]).to(self.device)
        rets = torch.from_numpy(data["rets"]).to(self.device)
        masks = torch.from_numpy(data["masks"]).to(self.device)

        # 归一化 advantage
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)
        # value target 归一化: 得分 scale ~±20, 不归一化时 value loss 对
        # 共享主干的梯度比 policy loss 大 >10x, 会拖着策略漂移 (已实测)
        ret_mean, ret_std = rets.mean(), rets.std() + 1e-8
        rets_n = (rets - ret_mean) / ret_std

        n = len(acts)
        for ep in range(epochs):
            perm = torch.randperm(n, device=self.device)
            total_pl, total_vl, total_ent, clip_frac = 0.0, 0.0, 0.0, 0.0
            nb = 0
            for i in range(0, n, batch_size):
                idx = perm[i:i + batch_size]
                logits, values = self.model(feats[idx])
                logits = logits.masked_fill(~masks[idx], -1e9)

                # policy loss (PPO clip)
                logp = F.log_softmax(logits, dim=-1)
                new_lp = logp.gather(1, acts[idx].unsqueeze(1)).squeeze(1)
                ratio = torch.exp(new_lp - old_lps[idx])
                surr1 = ratio * advs[idx]
                surr2 = torch.clamp(ratio, 1 - self.clip_eps,
                                    1 + self.clip_eps) * advs[idx]
                policy_loss = -torch.min(surr1, surr2).mean()
                clip_frac = max(clip_frac, float(
                    ((ratio - 1).abs() > self.clip_eps).float().mean()))

                # value loss (归一化 scale + huber, 梯度有界)
                value_loss = F.smooth_l1_loss(values, rets_n[idx])

                # entropy bonus (合法动作上的分布)
                probs = torch.softmax(logits, dim=-1)
                entropy = -(probs * logp).sum(-1).mean()

                loss = (policy_loss
                        + self.value_coef * value_loss
                        - self.entropy_coef * entropy)

                self.opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.opt.step()

                total_pl += policy_loss.item()
                total_vl += value_loss.item()
                total_ent += entropy.item()
                nb += 1

            print(f"  PPO ep {ep+1}/{epochs}  "
                  f"pi_loss={total_pl/nb:.4f}  v_loss={total_vl/nb:.4f}  "
                  f"ent={total_ent/nb:.3f}  clip={clip_frac:.2f}")
