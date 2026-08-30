"""复式(四座位轮转)相比"多开 seed"是否真的更省 —— 检验 var(D) < 4·var(d)。

估计量:
  d_s(seed) = score_A(seed, A 坐 s 位) - score_B(seed, B 坐 s 位)
  D(seed)   = Σ_{s=0..3} d_s(seed)          <- 复式: 同一副牌四个座位都玩
等算力比较(旋转一次 = 4 局):
  var(D)/16  vs  4·var(d)/16   ->  比值 var(D)/(4·var(d))
  < 1  说明复式真的更省; ≈ 1 说明只是把算力换了个花样; > 1 说明更差。

基线臂用全 v31n 一局即可拿到四个座位的分, 所以每 seed 只要 4+1 局。

用法: python -m tools.perf.diag_duplicate --games 500 --test nn
"""

import argparse

import numpy as np

from tools.perf.diag_crn import adjusted, make_seat
from backend.ai.bot_native import NativeV31
from backend.game.engine import Game


def run(seed, spec, seat, rng):
    """spec 坐 seat, 其余 v31n。返回四个座位的调整得分。"""
    g = Game(seed=seed, human_seat=-1)
    bots = {s: NativeV31(g, s) for s in range(4)}
    if spec is not None:
        bots[seat] = make_seat(spec, g, seat, rng)
    guard = 0
    while g.phase != "game_over" and guard < 500:
        guard += 1
        if g.phase == "discard_wait":
            g.action_discard(g.turn, bots[g.turn].choose_discard())
        else:
            s = list(g.pending_actions.keys())[0]
            pend = g.pending_actions[s]
            b = bots[s]
            if pend.get("gang") and b.decide_gang(g.last_discard, "ming"):
                g.action_gang(s)
            elif pend.get("peng") and b.decide_peng(g.last_discard):
                g.action_peng(s)
            else:
                g.action_pass(s)
    return [adjusted(g, s) for s in range(4)]


def report(d):
    D = d.sum(1)
    n = len(d)
    vd = d.reshape(-1).var()
    vD = D.var()
    print(f"\nseed 数 {n}   逐座位差分 d: 均 {d.mean():+.4f}  方差 {vd:.3f}")
    for s in range(4):
        print(f"    座{s}: 均 {d[:, s].mean():+.4f} 方差 {d[:, s].var():.3f}")
    C = np.corrcoef(d.T)
    off = [C[i, j] for i in range(4) for j in range(i + 1, 4)]
    print(f"  座位间差分的相关: 均 {np.mean(off):+.4f}  区间 "
          f"[{min(off):+.4f}, {max(off):+.4f}]")
    print(f"\n复式和 D: 均 {D.mean():+.4f}  方差 {vD:.3f}")
    print(f"**var(D) / (4·var(d)) = {vD / (4 * vd):.3f}**  "
          "(<1 复式更省, ≈1 白换, >1 更差)")
    se_rot = (vD / n) ** .5 / 4
    se_flat = (vd / (4 * n)) ** .5
    print(f"\n等算力下(总局数固定): 复式 SE={se_rot:.4f}  "
          f"平铺 SE={se_flat:.4f}   比值 {se_rot / se_flat:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=500, help="seed 数")
    ap.add_argument("--test", default="eps5", help="被测策略(基线=v31n)")
    ap.add_argument("--seed0", type=int, default=9800000)
    ap.add_argument("--save")
    ap.add_argument("--load", nargs="+")
    args = ap.parse_args()

    if args.load:
        report(np.vstack([np.load(p)["d"] for p in args.load]))
        return
    print(f"被测 {args.test} 轮坐四个座位, 基线 v31n, {args.games} 个 seed")

    ds = []          # (seed, 4) 逐座位差分
    for i in range(args.games):
        seed = args.seed0 + i
        base = run(seed, None, 0, None)          # 全 v31n, 一局拿四座位
        row = []
        for s in range(4):
            a = run(seed, args.test, s, np.random.default_rng(seed * 4 + s))
            row.append(a[s] - base[s])
        ds.append(row)
        if (i + 1) % 100 == 0:
            print(f"  ...{i + 1}/{args.games}")

    d = np.array(ds)                 # (n, 4)
    if args.save:
        np.savez(args.save, d=d)
        print(f"已存 {args.save}")
    report(d)


if __name__ == "__main__":
    main()
