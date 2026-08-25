"""安康159 - GRP v3: 相对目标 + 杠分特征 + 胜者分类辅助头

相对 v2 的关键改进:
1. **相对目标**: v2 的目标是绝对座位得分, 但输入是相对视角(我的手牌/下家副露),
   模型无法推断绝对座位映射(杠分非零时才有线索), 每个输出头信号被稀释 4 倍。
   v3 输出按相对位置: [我, 下家(rel1), 对家(rel2), 上家(rel3)] 的最终得分。
   奖励整形只用输出[0](我的预期得分), 信号不再稀释。
2. 特征加 5 维: 相对视角的 4 家已累计杠分 + 杠总数
3. 胜者分类辅助头(5类: 我/rel1/rel2/rel3/流局) — 85% 方差来自"谁赢",
   分类信号比回归干净, 加速表征学习
4. 数据量 50k → 200k 局

用法:
  python -m backend.rl.grp_train3 --games 200000 --workers 12
"""

import multiprocessing as mp
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..game.engine import Game
from ..ai.bot import Bot
from .features_v2 import encode_state

FEAT_DIM_V2 = 628
GRP3_FEAT_DIM = FEAT_DIM_V2 + 5


def gang_scores_so_far(g: Game) -> list[int]:
    """从 gang_records 算各家当前已产生的杠分(与 _settle 一致的口径)"""
    scores = [0, 0, 0, 0]
    for rec in g.gang_records:
        s = rec["seat"]
        if rec["kind"] == "ming":
            scores[rec["from"]] -= 3
            scores[s] += 3
        else:
            for other in range(4):
                if other != s:
                    scores[other] -= 1
                    scores[s] += 1
    return scores


def encode_state_grp3(game: Game, seat: int) -> np.ndarray:
    """v2 特征 + 相对视角杠分特征"""
    base = encode_state(game, seat)
    gang_abs = gang_scores_so_far(game)
    # 相对位置: rel 0(我), 1(下家), 2(对家), 3(上家)
    gang_rel = [gang_abs[(seat + r) % 4] for r in range(4)]
    extra = [g / 12.0 for g in gang_rel] + [len(game.gang_records) / 8.0]
    return np.concatenate([base, np.asarray(extra, dtype=np.float32)])


def play_one_game_grp3(seed: int):
    g = Game(seed=seed, human_seat=-1)
    bots = {i: Bot(g, i) for i in range(4)}
    records = []  # (seat, feat)

    guard = 0
    while g.phase != "game_over" and guard < 500:
        guard += 1
        if g.phase == "discard_wait":
            seat = g.turn
            feat = encode_state_grp3(g, seat)
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
    n = len(records)
    feats = np.stack([r[1] for r in records]).astype(np.float32) \
        if records else np.zeros((0, GRP3_FEAT_DIM), dtype=np.float32)
    seats = np.array([r[0] for r in records], dtype=np.int64)

    # 相对得分: [我, rel1, rel2, rel3]
    scores_rel = np.zeros((n, 4), dtype=np.float32)
    winners_rel = np.zeros(n, dtype=np.int64)  # 0-3 相对位置, 4=流局
    for i, seat in enumerate(seats):
        for r in range(4):
            scores_rel[i, r] = scores[(seat + r) % 4]
        if g.winner is None:
            winners_rel[i] = 4
        else:
            winners_rel[i] = (g.winner - seat) % 4
    return feats, seats, scores_rel, winners_rel


def _wrapper(seed):
    return play_one_game_grp3(seed)


def generate_grp3_data(n_games, seed0=0, workers=12):
    t0 = time.time()
    with mp.Pool(min(workers, n_games)) as pool:
        results = list(pool.imap_unordered(
            _wrapper, range(seed0, seed0 + n_games),
            chunksize=max(1, n_games // workers // 4)))
    feats = np.concatenate([r[0] for r in results])
    seats = np.concatenate([r[1] for r in results])
    scores = np.concatenate([r[2] for r in results])
    winners = np.concatenate([r[3] for r in results])
    print(f"生成 {n_games} 局, {len(seats)} 样本, "
          f"耗时 {time.time()-t0:.0f}s")
    return {"feats": feats, "seats": seats, "scores": scores,
            "winners": winners}


class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = self.fc2(h)
        return F.relu(x + h)


class GRP3Model(nn.Module):
    """预测相对位置4家最终得分 + 相对胜者分类(5类)。
    输出[0] = 决策者自己的预期得分 (奖励整形只用这个)"""

    def __init__(self, hidden_dim=512, num_blocks=8):
        super().__init__()
        self.input_proj = nn.Linear(GRP3_FEAT_DIM, hidden_dim)
        self.blocks = nn.ModuleList(ResBlock(hidden_dim)
                                    for _ in range(num_blocks))
        self.score_head = nn.Linear(hidden_dim, 4)
        self.winner_head = nn.Linear(hidden_dim, 5)

    def forward(self, x):
        h = F.relu(self.input_proj(x))
        for b in self.blocks:
            h = b(h)
        return self.score_head(h), self.winner_head(h)  # (B,4), (B,5)


def train_grp3(model, data, device, epochs=40, batch_size=2048, lr=1e-3,
               winner_coef=0.5):
    feats = torch.from_numpy(data["feats"]).to(device)
    scores = torch.from_numpy(data["scores"]).to(device)
    winners = torch.from_numpy(data["winners"]).to(device)
    n = len(feats)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        tot_s, tot_w, nb = 0.0, 0.0, 0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            pred_s, pred_w = model(feats[idx])
            loss_s = F.mse_loss(pred_s, scores[idx])
            loss_w = F.cross_entropy(pred_w, winners[idx])
            loss = loss_s + winner_coef * loss_w
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot_s += loss_s.item()
            tot_w += loss_w.item()
            nb += 1
        sched.step()
        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"  GRP3 ep {ep+1}/{epochs}  mse={tot_s/nb:.4f}  "
                  f"ce={tot_w/nb:.4f}")
    return model


@torch.no_grad()
def evaluate_grp3(model, feats, scores, winners, device, batch_size=8192):
    """scores/winners 均为相对视角"""
    model.eval()
    n = len(feats)
    preds_s, preds_w = [], []
    for i in range(0, n, batch_size):
        x = torch.from_numpy(feats[i:i + batch_size]).to(device)
        s, w = model(x)
        preds_s.append(s.cpu())
        preds_w.append(w.cpu())
    pred_s = torch.cat(preds_s).numpy()
    pred_w = torch.cat(preds_w)

    my_pred = pred_s[:, 0]
    my_true = scores[:, 0]
    my_err = my_pred - my_true
    my_mse = float((my_err ** 2).mean())
    my_var = float(my_true.var())

    is_winner = (winners == 0).astype(np.int64)
    p_win = torch.softmax(pred_w, dim=-1).numpy()[:, 0]
    p_win_corr = float(np.corrcoef(p_win, is_winner)[0, 1])
    top1_acc = float((pred_w.argmax(-1).numpy() == winners).mean())

    progress = feats[:, 506]
    buckets = {}
    for lo, hi in [(0.4, 0.6), (0.6, 0.8), (0.8, 1.01)]:
        m = (progress >= lo) & (progress < hi)
        if m.sum() > 0:
            te, pe = my_true[m], my_err[m]
            buckets[f"{lo:.1f}-{hi:.1f}"] = {
                "n": int(m.sum()),
                "mse": float((pe ** 2).mean()),
                "var": float(te.var()),
            }
    return {"my_mse": my_mse, "my_var": my_var,
            "my_r2": 1 - my_mse / my_var, "buckets": buckets,
            "corr": float(np.corrcoef(my_pred, my_true)[0, 1]),
            "p_win_corr": p_win_corr, "top1_acc": top1_acc}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=200000)
    ap.add_argument("--seed0", type=int, default=2000000)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--blocks", type=int, default=8)
    ap.add_argument("--out", type=str, default="models/grp_v3.pt")
    args = ap.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data = generate_grp3_data(args.games, args.seed0, args.workers)
    feats, seats = data["feats"], data["seats"]
    scores, winners = data["scores"], data["winners"]
    print(f"标签统计: var={scores[:, 0].var():.3f}, "
          f"流局率={float((winners == 4).mean()):.3f}")

    n = len(seats)
    idx = np.random.RandomState(0).permutation(n)
    n_val = min(200000, n // 20)
    val_idx, tr_idx = idx[:n_val], idx[n_val:]

    model = GRP3Model(args.hidden, args.blocks).to(device)
    print(f"GRP3 参数量: "
          f"{sum(p.numel() for p in model.parameters())/1e6:.2f}M, "
          f"训练 {len(tr_idx)}, 验证 {len(val_idx)}")

    train_grp3(model, {"feats": feats[tr_idx], "scores": scores[tr_idx],
                       "winners": winners[tr_idx]}, device,
               epochs=args.epochs)

    val = evaluate_grp3(model, feats[val_idx], scores[val_idx],
                        winners[val_idx], device)
    print(f"\n=== GRP3 验证 (相对视角, 决策者本人) ===")
    print(f"MSE={val['my_mse']:.3f} var={val['my_var']:.3f} "
          f"R2={val['my_r2']:.3f} corr={val['corr']:.3f}")
    print(f"P(我赢)与是否胡牌相关性: {val['p_win_corr']:.3f}, "
          f"胜者top1准确率: {val['top1_acc']:.3f}")
    for b, s in val["buckets"].items():
        r2 = 1 - s["mse"] / max(s["var"], 1e-9)
        print(f"  进度 {b}: n={s['n']} mse={s['mse']:.3f} "
              f"var={s['var']:.3f} R2={r2:.3f}")

    torch.save({"state_dict": model.state_dict(),
                "hidden": args.hidden, "blocks": args.blocks,
                "feat_dim": GRP3_FEAT_DIM, "version": 3},
               args.out)
    print(f"已保存 {args.out}")


if __name__ == "__main__":
    main()
