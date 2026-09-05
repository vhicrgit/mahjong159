"""按置信度筛选搜索教师的标签。

为什么必须筛: 交叉验证证明"8 次 rollout 的 argmax 比当前策略好 +0.154 分/局
(t=3.88)", 但那是**在整批上平均**成立。单个状态的标签噪声很大 —— 单候选
奖励标准差约 1.8, 8 次 rollout 后 SE≈0.64, 两个候选之差的 SE≈0.9。直接把
softmax(score/0.8) 当目标训练, 学到的主要是噪声: 纯教师那一臂掉到
-0.628 分/局, 教师+锚也只是与起点持平。

筛法: 只保留 (最优 - 次优) 超过 k×SE 的状态, 给硬标签。留下的是 rollout 真的
分出了胜负的局面, 其余丢掉 —— 宁可少而准。

最优那张牌直接取 argmax(target): 生成时 target = softmax(score/τ) 只在候选上
非零, 而 softmax 是分数的单调函数, 所以权重最大的那张就是分数最高的那张。

用法:
  python -m tools.filter_teacher --data "models/teach_*.npz" --k 2.0 \
      --out models/teach_conf.npz
"""

import argparse
import glob

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--rolls", type=int, default=8,
                    help="生成时每候选的 rollout 次数")
    ap.add_argument("--sd", type=float, default=1.8,
                    help="单次 rollout 奖励的标准差(用于估 SE)")
    ap.add_argument("--k", type=float, default=2.0, help="阈值 = k × SE(差)")
    ap.add_argument("--out", default="models/teach_conf.npz")
    args = ap.parse_args()

    paths = []
    for p in args.data:
        paths += sorted(glob.glob(p))
    se1 = args.sd / np.sqrt(args.rolls)
    se_diff = se1 * np.sqrt(2.0)
    thr = args.k * se_diff
    print(f"单候选 SE {se1:.3f}  差的 SE {se_diff:.3f}  阈值 {thr:.3f}")

    F, T, B, M, G = [], [], [], [], []
    have_groups = True
    tot = 0
    gaps = []
    for p in paths:
        d = np.load(p)
        S, P, feats = d["scores"], d["target"], d["feats"]
        paired = d["rollout_returns"] if "rollout_returns" in d.files else None
        if paired is None:
            print(f"WARNING {p}: legacy scores have no per-world returns; fixed-SD threshold is heuristic")
        elif str(d["world_mode"]) != "resample":
            print(f"WARNING {p}: repeated fixed-world returns do not measure hidden-world uncertainty")
        if "game_id" not in d.files:
            have_groups = False
        for i in range(len(S)):
            tot += 1
            s = S[i]
            ok = np.isfinite(s)
            if ok.sum() < 2:
                continue
            order = np.flatnonzero(ok)[np.argsort(s[ok])[::-1]]
            v = s[order]
            gap = v[0] - v[1]
            gaps.append(gap)
            threshold = thr
            if paired is not None:
                delta = paired[i, order[0]] - paired[i, order[1]]
                if len(delta) < 2 or not np.isfinite(delta).all():
                    continue
                threshold = args.k * delta.std(ddof=1) / np.sqrt(len(delta))
            if gap <= 0 or gap < threshold:
                continue
            tile = int(np.argmax(P[i]))
            t = np.zeros(28, dtype=np.float32)
            t[tile] = 1.0
            F.append(feats[i])
            T.append(t)
            B.append(tile)
            M.append(feats[i, :112].reshape(28, 4)[:, 0] == 0)
            G.append(int(d["game_id"][i]) if "game_id" in d.files else -1)
    if not F:
        raise SystemExit("没有通过阈值的样本, 调小 --k 或加大 --rolls")
    gaps = np.array(gaps)
    print(f"分差分布: 均 {gaps.mean():.3f} 中位 {np.median(gaps):.3f} "
          f"90分位 {np.percentile(gaps, 90):.3f}")
    extra = {"game_id": np.array(G)} if have_groups else {}
    print("筛选使用同批选出的赢家/次优: 存在选择偏差，不构成独立验证的置信保证")
    np.savez_compressed(args.out, feats=np.stack(F), target=np.stack(T),
                        bests=np.array(B, dtype=np.int8),
                        legal_mask=np.stack(M), value_valid=np.zeros(len(B), dtype=bool),
                        labels=np.zeros(len(B), dtype=np.float32), **extra)
    print(f"已存 {args.out}  保留 {len(B)}/{tot} ({len(B) / tot:.1%})")


if __name__ == "__main__":
    main()
