"""验证: 增强特征 (v2) 的 BC 模型能否真正学到规则Bot策略

对比 v1 (512维) vs v2 (628维) 在相同数据量下的:
1. 训练准确率
2. 真实对局一致率 (fresh games, 不在训练集里)
3. 实战胜率
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import time
import numpy as np
import torch
import torch.nn.functional as F
from backend.rl.model import build_model, legal_discard_mask, N_ACTIONS
from backend.rl.selfplay import generate_dataset
from backend.rl.train import get_device
from backend.game.engine import Game
from backend.ai.bot_v1 import Bot
from backend.rl.features_v2 import encode_state


def train_bc_split(model, data, device, epochs=20, batch_size=512, lr=1e-3):
    """BC训练, 带验证集"""
    n = len(data["acts"])
    idx = np.random.permutation(n)
    split = int(n * 0.9)
    tr_idx, va_idx = idx[:split], idx[split:]

    feats = torch.from_numpy(data["feats"]).to(device)
    acts = torch.from_numpy(data["acts"]).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best_va_acc = 0
    for ep in range(epochs):
        perm = torch.from_numpy(tr_idx).to(device)[torch.randperm(len(tr_idx))]
        total_loss, total_correct = 0.0, 0
        for i in range(0, len(perm), batch_size):
            b = perm[i:i+batch_size]
            logits, _ = model(feats[b])
            loss = F.cross_entropy(logits, acts[b])
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item() * len(b)
            total_correct += (logits.argmax(-1) == acts[b]).sum().item()
        tr_acc = total_correct / len(perm)

        # 验证
        with torch.no_grad():
            va_f = feats[torch.from_numpy(va_idx).to(device)]
            va_a = acts[torch.from_numpy(va_idx).to(device)]
            va_logits, _ = model(va_f)
            va_acc = (va_logits.argmax(-1) == va_a).float().mean().item()
        best_va_acc = max(best_va_acc, va_acc)
        if (ep+1) % 5 == 0 or ep == 0:
            print(f"  ep {ep+1}/{epochs}  loss={total_loss/len(perm):.4f}  "
                  f"train_acc={tr_acc:.3f}  val_acc={va_acc:.3f}")
    return best_va_acc


def test_real_agreement(model, n_games=20):
    """在真实对局中测试与规则Bot的一致率"""
    match, total = 0, 0
    for i in range(n_games):
        g = Game(seed=700000 + i, human_seat=-1)
        bots = {s: Bot(g, s) for s in range(4)}
        guard = 0
        while g.phase != "game_over" and guard < 500:
            guard += 1
            if g.phase == "discard_wait":
                seat = g.turn
                rule_choice = bots[seat].choose_discard()
                feat = encode_state(g, seat)
                x = torch.from_numpy(feat).unsqueeze(0).to(next(model.parameters()).device)
                mask = legal_discard_mask(g.players[seat].hand_counts).unsqueeze(0).to(next(model.parameters()).device)
                with torch.no_grad():
                    probs = model.policy(x, mask)[0]
                model_choice = int(probs.argmax().item())
                total += 1
                if model_choice == rule_choice:
                    match += 1
                g.action_discard(seat, rule_choice)
            elif g.phase == "react_wait":
                s = list(g.pending_actions.keys())[0]
                b = bots[s]
                if g.pending_actions[s].get("gang") and \
                        b.decide_gang(g.last_discard, "ming"):
                    g.action_gang(s)
                elif g.pending_actions[s].get("peng") and \
                        b.decide_peng(g.last_discard):
                    g.action_peng(s)
                else:
                    g.action_pass(s)
    return match / total if total else 0


def eval_vs_rule(model_path, n_games=100):
    """评估模型 vs 3规则Bot"""
    from backend.rl.evaluate import play_eval_game
    wins, total = 0, 0.0
    for i in range(n_games):
        sd, won = play_eval_game(800000 + i, model_path)
        total += sd
        wins += won
    return wins / n_games, total / n_games


if __name__ == "__main__":
    device = get_device()
    print(f"设备: {device}")

    # 生成数据
    print("\n=== 生成 2000 局 BC 数据 ===")
    t0 = time.time()
    data = generate_dataset(2000, workers=16)
    t1 = time.time()
    print(f"样本数: {len(data['acts'])}, 耗时: {t1-t0:.1f}s")

    # 训练 small 模型 (v2 特征)
    print("\n=== BC 训练 small 模型 (v2 特征, 20 epochs) ===")
    model = build_model("small").to(device)
    print(f"参数量: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    best_va = train_bc_split(model, data, device, epochs=20)
    print(f"最佳验证准确率: {best_va:.1%}")

    # 保存
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "..", "..", "models", "bc_v2_small.pt")
    torch.save({"model": model.state_dict(), "size": "small"}, out_path)

    # 真实对局一致率
    print("\n=== 真实对局一致率 (20局) ===")
    agreement = test_real_agreement(model, 20)
    print(f"一致率: {agreement:.1%}")

    # 实战胜率
    print("\n=== 实战胜率 (100局) ===")
    wr, avg = eval_vs_rule(out_path, 100)
    print(f"胜率: {wr:.1%}, 场均: {avg:+.2f}")
