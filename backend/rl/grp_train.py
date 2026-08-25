"""安康159 - GRP 数据生成 + 训练 + 验证

数据: 规则Bot自对弈, 每个出牌决策点记录
  feats:  (N, 628) float32  决策点局面特征(决策玩家视角)
  seats:  (N,)     int64    决策玩家座位
  scores: (N, 4)   float32  该局4个玩家的最终得分

验证指标:
  - 整体 MSE / MAE / R2
  - 按局进度(progress)分桶的 R2: 随局推进应上升
  - 方差缩减: 直接用局终得分做 per-step 奖励的方差 vs GRP差分奖励的方差
"""

import multiprocessing as mp
import time

import numpy as np
import torch

from ..game.engine import Game
from ..ai.bot import Bot
from .features_v2 import encode_state
from .grp import GRPModel, train_grp


def play_one_game_grp(seed: int):
    """规则Bot自对弈一局, 返回 (feats, seats, scores4)"""
    g = Game(seed=seed, human_seat=-1)
    bots = {i: Bot(g, i) for i in range(4)}
    records = []  # (seat, feat)

    guard = 0
    while g.phase != "game_over" and guard < 500:
        guard += 1
        if g.phase == "discard_wait":
            seat = g.turn
            feat = encode_state(g, seat)
            tile = bots[seat].choose_discard()
            records.append((seat, feat))
            g.action_discard(seat, tile)
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

    scores = np.array([p.score_delta for p in g.players], dtype=np.float32)
    feats = np.stack([r[1] for r in records]).astype(np.float32) \
        if records else np.zeros((0, 628), dtype=np.float32)
    seats = np.array([r[0] for r in records], dtype=np.int64)
    scores4 = np.tile(scores, (len(records), 1)) if records else \
        np.zeros((0, 4), dtype=np.float32)
    return feats, seats, scores4


def _wrapper(seed):
    return play_one_game_grp(seed)


def generate_grp_data(n_games: int, seed0: int = 0, workers: int = 16) -> dict:
    t0 = time.time()
    with mp.Pool(min(workers, n_games)) as pool:
        results = list(pool.imap_unordered(
            _wrapper, range(seed0, seed0 + n_games),
            chunksize=max(1, n_games // workers // 4)))
    feats = np.concatenate([r[0] for r in results])
    seats = np.concatenate([r[1] for r in results])
    scores = np.concatenate([r[2] for r in results])
    dt = time.time() - t0
    print(f"生成 {n_games} 局, {len(seats)} 样本, 耗时 {dt:.1f}s "
          f"({n_games/dt:.0f} 局/s)")
    return {"feats": feats, "seats": seats, "scores": scores}


@torch.no_grad()
def evaluate_grp(model, feats, scores, seats, device, batch_size=8192):
    """整体 MSE / 按进度分桶 / 决策者本人得分的方差缩减"""
    model.eval()
    n = len(feats)
    preds = []
    for i in range(0, n, batch_size):
        x = torch.from_numpy(feats[i:i + batch_size]).to(device)
        preds.append(model(x).cpu().numpy())
    pred = np.concatenate(preds)
    err = pred - scores
    mse = float((err ** 2).mean())

    # 决策者本人的预测 (GRP 奖励只关心这个)
    my_pred = pred[np.arange(n), seats]
    my_true = scores[np.arange(n), seats]
    my_err = my_pred - my_true
    my_mse = float((my_err ** 2).mean())
    my_var = float(my_true.var())

    # progress 特征位于 v1 全局标量区: offset 504=wall_ratio, 506=progress
    progress = feats[:, 506]
    buckets = {}
    for lo, hi in [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6),
                   (0.6, 0.8), (0.8, 1.01)]:
        m = (progress >= lo) & (progress < hi)
        if m.sum() > 0:
            te = my_true[m]
            pe = my_err[m]
            buckets[f"{lo:.1f}-{hi:.1f}"] = {
                "n": int(m.sum()),
                "mse": float((pe ** 2).mean()),
                "var": float(te.var()),
            }
    return {"mse4": mse, "my_mse": my_mse, "my_var": my_var,
            "my_r2": 1 - my_mse / my_var, "buckets": buckets,
            "corr": float(np.corrcoef(my_pred, my_true)[0, 1])}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=50000)
    ap.add_argument("--seed0", type=int, default=1000000)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--blocks", type=int, default=6)
    ap.add_argument("--data", type=str, default="")
    ap.add_argument("--out", type=str, default="models/grp_v2.pt")
    args = ap.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if args.data:
        data = np.load(args.data)
        feats, seats, scores = data["feats"], data["seats"], data["scores"]
        print(f"加载数据 {args.data}: {len(seats)} 样本")
    else:
        data = generate_grp_data(args.games, args.seed0, args.workers)
        feats, seats, scores = data["feats"], data["seats"], data["scores"]
        np.savez_compressed("models/grp_data.npz",
                            feats=feats, seats=seats, scores=scores)
        print("数据已保存 models/grp_data.npz")

    print(f"标签统计: mean={scores.mean():.3f} std={scores.std():.3f} "
          f"var={scores.var():.3f}")

    # train/val split
    n = len(seats)
    idx = np.random.RandomState(0).permutation(n)
    n_val = min(200000, n // 10)
    val_idx, tr_idx = idx[:n_val], idx[n_val:]

    model = GRPModel(args.hidden, args.blocks).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"GRP 参数量: {n_params/1e6:.2f}M, 训练样本 {len(tr_idx)}, "
          f"验证样本 {len(val_idx)}")

    train_data = {"feats": feats[tr_idx], "scores": scores[tr_idx]}
    train_grp(model, train_data, device, epochs=args.epochs)

    val = evaluate_grp(model, feats[val_idx], scores[val_idx],
                       seats[val_idx], device)
    print(f"\n=== GRP 验证 (决策者本人得分) ===")
    print(f"MSE={val['my_mse']:.3f} 标签var={val['my_var']:.3f} "
          f"R2={val['my_r2']:.3f} corr={val['corr']:.3f}")
    for b, s in val["buckets"].items():
        r2 = 1 - s["mse"] / max(s["var"], 1e-9)
        print(f"  进度 {b}: n={s['n']} mse={s['mse']:.3f} "
              f"var={s['var']:.3f} R2={r2:.3f}")

    torch.save({"state_dict": model.state_dict(),
                "hidden": args.hidden, "blocks": args.blocks},
               args.out)
    print(f"GRP 模型已保存 {args.out}")


if __name__ == "__main__":
    main()
