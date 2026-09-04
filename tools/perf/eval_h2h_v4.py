"""v4 模型 vs v2 模型的同种子头对头(CRN 配对, 轮坐四座位, 对手 v31n)。

A 臂: v4 模型(V4Seat, 实时 tracker) 坐 seat s
B 臂: v2 模型(NNSeat) 坐同一 seat s, 同一 seed
两臂碰/杠都用分析器 E 判据(kai=0), 唯一差异是弃牌网络与其输入。

用法: python -m tools.perf.eval_h2h_v4 models/bc_v4_r1.pt models/bc_k0_r2.pt --seeds 300
"""
import argparse
import sys

import numpy as np
import torch

from backend.ai.bot_native import NativeV31
from backend.rl import eval_crn
from backend.rl.model import build_model
from tools.rl_bloody_train import NNSeat
sys.path.insert(0, "tools/perf")
from eval_ckpt_v4 import play_v4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("v4ckpt")
    ap.add_argument("v2ckpt")
    ap.add_argument("--seeds", type=int, default=300)
    ap.add_argument("--seed0", type=int, default=151000000)
    args = ap.parse_args()

    cka = torch.load(args.v4ckpt, map_location="cpu", weights_only=True)
    ma = build_model(cka["size"], feat_dim=cka.get("feat_dim", 718))
    ma.load_state_dict(cka["model"]); ma.eval()
    ckb = torch.load(args.v2ckpt, map_location="cpu", weights_only=True)
    mb = build_model(ckb["size"], feat_dim=ckb.get("feat_dim", 628))
    mb.load_state_dict(ckb["model"]); mb.eval()
    print(f"A: {args.v4ckpt} (v4)   B: {args.v2ckpt} (v2)")

    for bloody, name in ((False, "首胡(线上规则)"), (True, "血战到底")):
        seeds = list(range(args.seed0, args.seed0 + args.seeds))
        rra = np.zeros((args.seeds, 4)); sca = np.zeros((args.seeds, 4))
        rrb = np.zeros((args.seeds, 4)); scb = np.zeros((args.seeds, 4))
        for i, sd in enumerate(seeds):
            # B 臂(便宜)先跑: 四座位轮转
            for s in range(4):
                fac = {k: NativeV31 for k in range(4)}
                fac[s] = lambda g, ss, s=s: NNSeat(g, ss, mb)
                g = eval_crn._play(sd, bloody, fac)
                rrb[i, s] = g.rank_rewards()[s]
                scb[i, s] = eval_crn._adjusted(g, s)
            # A 臂(带 tracker)
            for s in range(4):
                g = play_v4(sd, bloody, ma, s)
                rra[i, s] = g.rank_rewards()[s]
                sca[i, s] = eval_crn._adjusted(g, s)
            if (i + 1) % 50 == 0:
                print(f"  ...{i + 1}/{args.seeds}", flush=True)
        r = eval_crn._stat(rra - rrb)
        sc = eval_crn._stat(sca - scb)
        rho = np.corrcoef((rra - rrb).reshape(-1) * 0 + rra.reshape(-1),
                          rrb.reshape(-1))[0, 1]
        print(f"\n[{name}]  A(v4) - B(v2), 同 seed 同座位配对")
        print(f"  名次奖励差 {r['mean']:+.4f} ± {r['se']:.4f}  t={r['t']:+.2f}  (n={r['n']})")
        print(f"  调整得分差 {sc['mean']:+.4f} ± {sc['se']:.4f}  t={sc['t']:+.2f}")


if __name__ == "__main__":
    main()
