"""AC 二阶段的信号质量诊断: 四座位回报的相关性、critic 解释力、有效样本量。

回答两件事:
  1. 同一局四个座位的样本相关到什么程度 -> 名义 batch 有多少有效样本
  2. 单次弃牌的 advantage 里有多少是"这一手打得好", 多少是运气

用法: python -m tools.perf.diag_ac_signal --games 128
"""

import argparse

import numpy as np
import torch

from tools.rl_ac_train import ACVectorizedSelfPlay, adjusted_scores
from backend.rl.model import build_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=128)
    ap.add_argument("--model", default="models/acnn_v2.pt")
    ap.add_argument("--seed0", type=int, default=7000000)
    args = ap.parse_args()

    ck = torch.load(args.model, map_location="cpu", weights_only=True)
    model = build_model(ck["size"])
    model.load_state_dict(ck["model"])
    model.eval()

    vec = ACVectorizedSelfPlay(model, args.games, "cpu", seed0=args.seed0)
    results = vec.run(temperature=1.0)

    # ---- 1. 四座位回报的局内结构 ----
    S = np.array([adjusted_scores(r) for r in results], dtype=float)  # (G,4)
    print(f"局数 {len(S)}  座位回报: 均 {S.mean():+.3f} 标准差 {S.std():.3f}")
    print(f"  局内四座位之和: 均 {S.sum(1).mean():+.3f} 标准差 {S.sum(1).std():.3f}"
          "   (≈0 说明近似零和)")
    off = [np.corrcoef(S[:, i], S[:, j])[0, 1]
           for i in range(4) for j in range(i + 1, 4)]
    print(f"  座位两两相关: 均 {np.mean(off):+.3f}  区间 "
          f"[{min(off):+.3f}, {max(off):+.3f}]")

    # ---- 2. 逐决策样本 ----
    feats, rets, gids, seats = [], [], [], []
    for gi, res in enumerate(results):
        adj = adjusted_scores(res)
        for (seat, feat, tile, logp, val, regret, mask) in res["records"]:
            feats.append(feat)
            rets.append(float(adj[seat]))
            gids.append(gi)
            seats.append(seat)
    X = torch.from_numpy(np.stack(feats))
    G = np.array(rets)
    gids = np.array(gids)
    seats = np.array(seats)
    print(f"\n名义样本 {len(G)}  独立单元(局×座位) {len(S) * 4}  "
          f"每单元重复 {len(G) / (len(S) * 4):.1f} 次(同一回报被复用)")

    with torch.no_grad():
        _, v = model(X)
    v = v.numpy().ravel() * 10.0          # 训练里 Gs=G/10, 换算回分

    var_tot = G.var()
    resid = G - v
    print(f"\ncritic 解释力: var(G) {var_tot:.2f}  var(G-v) {resid.var():.2f}"
          f"  R² {1 - resid.var() / var_tot:+.3f}   MAE {np.abs(resid).mean():.2f}分")
    print(f"  critic 自身预测的离散度: var(v) {v.var():.3f} "
          f"(标准差 {v.std():.2f}分) vs 回报标准差 {G.std():.2f}分")

    # G 在一个(局,座位)单元内恒定, 所以 adv 的单元内差异只可能来自 v(s)。
    # 这个量决定了梯度能不能区分"同一局里哪一张打得好"。
    adv = G - v
    units, w_spread = 0, []
    for g in range(len(S)):
        m = gids == g
        if not m.any():
            continue
        for s in range(4):
            mm = m & (seats == s)
            if mm.sum() >= 2:
                w_spread.append(adv[mm].std())
                units += 1
    print(f"\n独立单元 {units} 个, 单元内 advantage 标准差 均 "
          f"{np.mean(w_spread):.3f}分  vs 单元间 {G.std():.2f}分")
    print(f"  信噪比(单元内可区分度 / 单元间噪声) = "
          f"{np.mean(w_spread) / G.std():.3f}")
    print(f"  名义样本 {len(G)} -> 策略梯度的有效样本约 {units} "
          f"(缩水 {len(G) / units:.1f}x)")


if __name__ == "__main__":
    main()
