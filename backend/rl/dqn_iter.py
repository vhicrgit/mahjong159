"""安康159 - 迭代离线 DQN (offline Q-iteration)

自提升循环: 当前策略(带探索)自对弈生成数据 -> 混合规则Bot数据 -> 离线DQN再训
原理: Q-greedy 27% 策略产生的数据分布比规则Bot 25% 的更有价值,
      数据质量随策略提升而提升 (expiteration, 但用离线Q而非策略蒸馏)

用法:
  python -m backend.rl.dqn_iter --init models/dqn_off_v2.pt \
      --rounds 5 --games-per-round 50000 --epochs 10
"""

import argparse
import time

import numpy as np
import torch

from .model import build_model
from .vec_selfplay import VectorizedSelfPlay
from .dqn_offline import train_offline_dqn, legal_mask_from_feats


def collect_selfplay(model, n_games, device, temperature, seed0, n_procs=8,
                     size="base"):
    """当前策略 vs 3规则Bot, 带温度探索, 收集 (feat, act, ret)"""
    from .grp_ppo_train import collect_batch_parallel
    grp = None
    # 复用并行收集器但不带 GRP (grp=None)
    import multiprocessing as mp

    model_sd = {k: v.cpu() for k, v in model.state_dict().items()}

    per = n_games // n_procs
    tasks = [(model_sd, None, None, size, per, seed0 + w * per, temperature,
              w % 8 if torch.cuda.is_available() else -1, 2)
             for w in range(n_procs)]

    ctx = mp.get_context("spawn")
    with ctx.Pool(n_procs) as pool:
        results = []
        for res in pool.imap_unordered(_selfplay_worker, tasks):
            results.extend(res)

    feats, acts, rets = [], [], []
    for r in results:
        for rec in r["records"]:
            seat, feat, act, lp, val, regret, mask = rec
            feats.append(feat)
            acts.append(act)
            rets.append(float(r["scores"][seat]))
    return {
        "feats": np.stack(feats).astype(np.float32),
        "acts": np.asarray(acts, dtype=np.int64),
        "rets": np.asarray(rets, dtype=np.float32),
    }


def _selfplay_worker(args):
    (model_sd, _, __, size, n_games, seed0, temperature, device_id,
     _unused) = args
    import torch as _torch
    if device_id >= 0 and _torch.cuda.is_available():
        device_id = device_id % _torch.cuda.device_count()
        device = _torch.device(f"cuda:{device_id}")
    else:
        device = _torch.device("cpu")
    from .model import build_model
    model = build_model(size).to(device)
    model.load_state_dict(model_sd)
    model.eval()
    engine = VectorizedSelfPlay(model, n_games, device, seed0=seed0,
                                model_seats=[0])
    return engine.run(temperature=temperature)


def main():
    from .grp_ppo_train import evaluate_vec

    ap = argparse.ArgumentParser()
    ap.add_argument("--init", type=str, required=True)
    ap.add_argument("--rule-data", type=str,
                    default="models/offline_data.npz")
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--games-per-round", type=int, default=50000)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--temperature", type=float, default=0.5,
                    help="自对弈探索温度")
    ap.add_argument("--cql-weight", type=float, default=0.5)
    ap.add_argument("--size", type=str, default="base")
    ap.add_argument("--eval-games", type=int, default=2000)
    ap.add_argument("--out", type=str, default="models/dqn_iter.pt")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(args.size).to(device)
    ckpt = torch.load(args.init, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
    print(f"初始化: {args.init}")

    z = np.load(args.rule_data)
    rule_feats, rule_acts, rule_rets = z["feats"], z["acts"], z["rets"]
    rule_masks = legal_mask_from_feats(rule_feats)
    print(f"规则Bot数据: {len(rule_acts)} 样本")

    wr, avg = evaluate_vec(model, device, args.eval_games)
    print(f"初始: 胜率 {wr:.1%}, 场均 {avg:+.2f}")
    best = avg

    for rd in range(args.rounds):
        t0 = time.time()
        # 1. 当前策略自对弈 (探索温度)
        sp = collect_selfplay(model, args.games_per_round, device,
                              args.temperature, seed0=8000000 + rd * 100000,
                              size=args.size)
        sp_masks = legal_mask_from_feats(sp["feats"])
        print(f"\n=== 轮 {rd+1}/{args.rounds} === 自对弈 "
              f"{args.games_per_round} 局 -> {len(sp['acts'])} 样本 "
              f"({time.time()-t0:.0f}s), 自对弈胜率 "
              f"{(sp['rets'] > 0).mean():.1%}")

        # 2. 混合: 自对弈数据(新) + 规则Bot数据(下采样 1:1)
        n_sp = len(sp["acts"])
        sel = np.random.RandomState(rd).choice(
            len(rule_acts), size=min(n_sp * 2, len(rule_acts)),
            replace=False)
        feats = np.concatenate([sp["feats"], rule_feats[sel]])
        acts = np.concatenate([sp["acts"], rule_acts[sel]])
        rets = np.concatenate([sp["rets"], rule_rets[sel]])
        masks = np.concatenate([sp_masks, rule_masks[sel]])

        # 3. 离线 DQN 再训 (从当前权重继续)
        data = {"feats": feats, "acts": acts, "rets": rets,
                "masks": masks}
        train_offline_dqn(model, data, device, epochs=args.epochs,
                          cql_weight=args.cql_weight)

        # 4. 评估
        wr, avg = evaluate_vec(model, device, args.eval_games,
                               seed0=960000 + rd)
        tag = ""
        if avg > best:
            best = avg
            tag = " ★"
        print(f"[轮 {rd+1}] 评估: 胜率 {wr:.1%}, 场均 {avg:+.2f}{tag}")
        torch.save({"model": model.state_dict(), "size": args.size},
                   args.out if avg >= best
                   else args.out.replace(".pt", "_prev.pt"))
        if tag:
            torch.save({"model": model.state_dict(), "size": args.size},
                       args.out.replace(".pt", "_best.pt"))

    print(f"\n完成。最优场均 {best:+.2f}")


if __name__ == "__main__":
    main()
