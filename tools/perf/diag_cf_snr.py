"""同墙反事实 advantage 的信噪比 —— 值不值得付那次重放的代价。

共同标尺: 用牌型分析器的 ΔE = E(打替代张) - E(打选中张) 当"已知的质量差"
(正值 = 选中的那张更好)。看两种 advantage 各能解释 ΔE 的多少方差:

  反事实 adv = R_主 - R_替      (同一副牌墙, 只差这一张)
  整局   adv = R - v(s)         (现在训练在用的; 注意它对"打哪张"没有依赖,
                                 只能通过采样到的动作影响 R 来间接相关)

corr² 直接可比, 再按每个样本的重放代价折算成"每单位算力的信号"。

用法: python -m tools.perf.diag_cf_snr --games 300
"""

import argparse
import time

import numpy as np
import torch

from backend.analysis import hv_native
from backend.rl import cf_collect
from backend.rl.model import build_model
from backend.rules.tiles import TILE_COUNT


def e_of(g, seat, tile, kai=0):
    """打出 tile 后的期望巡数。"""
    h = list(g.players[seat].hand_counts)
    if sum(h) % 3 != 2 or h[tile] <= 0:
        return float("nan")
    v = [0] * TILE_COUNT
    for q in g.players:
        for t in q.discards:
            v[t] += 1
        for m in q.melds:
            v[m["tile"]] += 3 if m["type"] == "peng" else 4
    for t, n in enumerate(h):
        v[t] += n
    hv_native.set_hand(h, v, 1.0, kai > 0, 2, kai, 6)
    return hv_native.e_after_discard(tile)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=300)
    ap.add_argument("--model", default="models/hv_value_pretrained.pt")
    ap.add_argument("--snap-p", type=float, default=0.10)
    ap.add_argument("--seed0", type=int, default=50000000)
    args = ap.parse_args()

    ck = torch.load(args.model, map_location="cpu", weights_only=True)
    model = build_model(ck["size"])
    model.load_state_dict(ck["model"], strict=False)
    model.eval()

    # 反事实采集(顺便量一下代价)
    t0 = time.perf_counter()
    rng = np.random.default_rng(7)
    from backend.game.engine import Game
    main_games = [Game(seed=args.seed0 + i, human_seat=-1, bloody=True)
                  for i in range(args.games)]
    main_games, snaps = cf_collect.run_games(model, "cpu", main_games, 1.0,
                                             args.snap_p, rng)
    t_main = time.perf_counter() - t0
    print(f"主线 {args.games} 局 {t_main:.1f}s, 快照 {len(snaps)} 个")

    # 在重放前先算 ΔE(需要快照当时的局面)
    dE = []
    for sn in snaps:
        e_c = e_of(sn["game"], sn["seat"], sn["tile"])
        e_a = e_of(sn["game"], sn["seat"], sn["alt"])
        dE.append(e_a - e_c)          # >0: 选中的那张更好
    dE = np.array(dE)

    t0 = time.perf_counter()
    alt = []
    for sn in snaps:
        g = sn["game"]
        g.action_discard(sn["seat"], sn["alt"])
        alt.append(g)
    alt, _ = cf_collect.run_games(model, "cpu", alt, 1.0)
    t_alt = time.perf_counter() - t0
    print(f"反事实重放 {len(alt)} 次 {t_alt:.1f}s")

    adv_cf = np.array([cf_collect.default_reward(main_games[sn["gi"]],
                                                 sn["seat"])
                       - cf_collect.default_reward(ga, sn["seat"])
                       for sn, ga in zip(snaps, alt)])
    # 整局 advantage: R - v(s), 同一批决策点
    X = torch.from_numpy(np.stack([sn["feat"] for sn in snaps]))
    with torch.no_grad():
        _, v = model(X)
    R = np.array([cf_collect.default_reward(main_games[sn["gi"]], sn["seat"])
                  for sn in snaps])
    adv_ep = R - v.numpy() * 0.0      # 预训练 v 尺度不对, 用批内均值当基线
    adv_ep = R - R.mean()

    ok = np.isfinite(dE) & np.isfinite(adv_cf)
    print(f"\n有效样本 {ok.sum()}")

    def rep(name, a, cost_per_sample):
        c = np.corrcoef(a[ok], dE[ok])[0, 1]
        print(f"  {name:16s} 标准差 {a[ok].std():6.3f}  corr(adv,ΔE) "
              f"{c:+.4f}  corr² {c ** 2:.4f}  每样本代价 {cost_per_sample:5.1f} "
              f"决策  单位算力信号 {c ** 2 / cost_per_sample:.5f}")

    nd_main = sum(len(p.discards) for g in main_games for p in g.players)
    # 整局 advantage 的"每样本代价": 一个座位一局约 nd/(4*games) 次决策,
    # 但同一回报被这些决策共享 -> 一个独立样本的代价就是这一整段
    per_unit = nd_main / (4 * args.games)
    nd_alt = sum(len(p.discards) for g in alt for p in g.players)
    print(f"主线总决策 {nd_main}, 重放总决策 {nd_alt}")
    rep("整局 R-baseline", adv_ep, per_unit)
    rep("同墙反事实", adv_cf, nd_alt / max(1, len(alt)))

    # 附: ΔE 本身的分布(反事实要区分的"真实差异"有多大)
    print(f"\nΔE(替代 - 选中): 均 {dE[ok].mean():+.3f} 巡  "
          f"标准差 {dE[ok].std():.3f}  |ΔE|中位 "
          f"{np.median(np.abs(dE[ok])):.3f}")


if __name__ == "__main__":
    main()
