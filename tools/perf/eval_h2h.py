"""两个检查点(或规则Bot)的高精度头对头比较(CRN 配对 + 轮坐四座位)。

为什么不各自跟 v31n 比再相减: 那样两次比较的牌运噪声只共享 0.5%(零和博弈里
共同分量几乎为 0), 合并后 SE 约 0.06 分/局。而两个同源网络在同一副牌墙上直接
配对时 ρ 很高, 降方差 1/(1-ρ) 可达 20 倍以上, SE 能压到 0.01~0.02 —— 这才够
分辨 0.05~0.1 分/局 这个量级的改动。

用法:
  python -m tools.perf.eval_h2h models/ei_b.pt models/bc_k0_r2.pt --seeds 400
  python -m tools.perf.eval_h2h v32 v31n --seeds 4096 --seed0 206300000 \
      --raw-score --first-win-only      # 规则Bot 之间也走这个入口
"""

import argparse

from backend.rl import eval_crn
from tools.perf.eval_ckpt import make_factory


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--seeds", type=int, default=400)
    ap.add_argument("--seed0", type=int, default=47000000)
    ap.add_argument("--claim", choices=["v31", "hv", "deploy"], default="hv")
    ap.add_argument("--claim-a", choices=["v31", "hv", "deploy"], default=None,
                    help="只改 A 臂的碰杠口径(默认与 --claim 相同)")
    ap.add_argument("--claim-b", choices=["v31", "hv", "deploy"], default=None,
                    help="只改 B 臂的碰杠口径(默认与 --claim 相同)")
    ap.add_argument("--raw-score", action="store_true",
                    help="得分列用真实净分(score_delta) —— 验收主指标")
    ap.add_argument("--first-win-only", action="store_true",
                    help="只跑首胡(线上规则), 不跑血战到底")
    args = ap.parse_args()
    fa, ta = make_factory(args.a, args.claim_a or args.claim)
    fb, tb = make_factory(args.b, args.claim_b or args.claim)
    seeds = list(range(args.seed0, args.seed0 + args.seeds))
    score_name = "真实净分差" if args.raw_score else "调整得分差"
    print(f"A: {ta}\nB: {tb}\n{args.seeds} seed × 4 座位, 对手 v31n, CRN 配对, "
          f"{score_name}口径")
    modes = ((False, "首胡(线上规则)"),) if args.first_win_only else \
        ((False, "首胡(线上规则)"), (True, "血战到底"))
    for bloody, name in modes:
        ev = eval_crn.paired_head2head(fa, fb, seeds, bloody=bloody,
                                       raw_score=args.raw_score)
        r, s = ev["rank"], ev["score"]
        print(f"\n[{name}]  A - B")
        print(f"  名次奖励差 {r['mean']:+.4f} ± {r['se']:.4f}  t={r['t']:+.2f}"
              f"  (n={r['n']}, 观测 {r['observations']})")
        print(f"  {score_name} {s['mean']:+.4f} ± {s['se']:.4f}  t={s['t']:+.2f}"
              f"   95%CI [{s['ci95_normal'][0]:+.4f}, {s['ci95_normal'][1]:+.4f}]")


if __name__ == "__main__":
    main()
