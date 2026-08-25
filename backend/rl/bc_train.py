"""安康159 - 大规模 BC 训练 + 评估

用 10k 局规则Bot数据训练强基线模型。
带验证集监控过拟合。

用法: python -m backend.rl.bc_train --games 10000 --size small --epochs 40
"""

import argparse
import os
import time
import numpy as np
import torch
import torch.nn.functional as F

from .model import build_model, legal_discard_mask, N_ACTIONS
from .selfplay import generate_dataset
from .train import get_device
from .evaluate import play_eval_game


def train_bc_val(model, data, device, epochs=40, batch_size=1024, lr=1e-3):
    """BC训练, 带验证集和early stopping"""
    n = len(data["acts"])
    idx = np.random.permutation(n)
    split = int(n * 0.9)
    tr_idx, va_idx = idx[:split], idx[split:]

    feats = torch.from_numpy(data["feats"]).to(device)
    acts = torch.from_numpy(data["acts"]).to(device)
    va_f = feats[torch.from_numpy(va_idx).to(device)]
    va_a = acts[torch.from_numpy(va_idx).to(device)]

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best_va_acc = 0
    patience = 0
    for ep in range(epochs):
        perm = torch.from_numpy(tr_idx).to(device)[torch.randperm(len(tr_idx))]
        total_loss, total_correct = 0.0, 0
        for i in range(0, len(perm), batch_size):
            b = perm[i:i+batch_size]
            q_values, _ = model(feats[b])
            loss = F.cross_entropy(q_values, acts[b])
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item() * len(b)
            total_correct += (q_values.argmax(-1) == acts[b]).sum().item()
        tr_acc = total_correct / len(perm)
        scheduler.step()

        # 验证
        with torch.no_grad():
            va_q, _ = model(va_f)
            va_acc = (va_q.argmax(-1) == va_a).float().mean().item()

        if va_acc > best_va_acc:
            best_va_acc = va_acc
            patience = 0
        else:
            patience += 1

        if (ep+1) % 5 == 0 or ep == 0:
            print(f"  ep {ep+1}/{epochs}  loss={total_loss/len(perm):.4f}  "
                  f"train={tr_acc:.3f}  val={va_acc:.3f}  "
                  f"lr={scheduler.get_last_lr()[0]:.2e}")

        if patience >= 8:
            print(f"  Early stop at ep {ep+1} (val_acc plateau)")
            break

    return best_va_acc


def evaluate_model(model_path, n_games=200):
    wins, total = 0, 0.0
    for i in range(n_games):
        sd, won = play_eval_game(900000 + i, model_path)
        total += sd
        wins += won
    return wins / n_games, total / n_games


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=10000)
    ap.add_argument("--size", type=str, default="small")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", type=str, default="models/bc_strong.pt")
    ap.add_argument("--eval-games", type=int, default=200)
    args = ap.parse_args()

    device = get_device()
    print(f"BC 训练: {args.size} 模型, {args.games} 局, {args.epochs} epochs")

    # 1. 生成数据
    print(f"\n=== 生成 {args.games} 局 BC 数据 ===")
    t0 = time.time()
    data = generate_dataset(args.games, workers=args.workers)
    t1 = time.time()
    print(f"样本: {len(data['acts'])}, 耗时: {t1-t0:.1f}s")

    # 2. 训练
    print(f"\n=== BC 训练 ({args.epochs} epochs) ===")
    model = build_model(args.size).to(device)
    print(f"参数量: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    best_va = train_bc_val(model, data, device, epochs=args.epochs, lr=args.lr)
    print(f"最佳验证准确率: {best_va:.1%}")

    # 3. 保存
    os.makedirs(os.path.dirname(args.out) if os.path.dirname(args.out) else ".", exist_ok=True)
    torch.save({"model": model.state_dict(), "size": args.size}, args.out)

    # 4. 评估
    print(f"\n=== 评估 ({args.eval_games} 局) ===")
    wr, avg = evaluate_model(args.out, args.eval_games)
    print(f"胜率: {wr:.1%}, 场均: {avg:+.2f}")
    print(f"(随机基线25%, 规则Bot自身约25%)")


if __name__ == "__main__":
    main()
