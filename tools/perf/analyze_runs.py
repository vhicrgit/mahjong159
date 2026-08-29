"""系统性分析 GRPO 训练结果: 有没有可检出的学习信号?

核心问题不是"哪个 checkpoint 最好", 而是:
  1. eval 的噪声有多大? 400 局的胜率标准误是多少?
  2. 各 run 的 eval 序列与 baseline 的差, 在噪声尺度下是否显著?
  3. 策略到底动没动(KL), 动了但不涨说明梯度方向和目标不一致
  4. ★(最优 checkpoint)选择本身是不是在选噪声
"""

import argparse
import glob
import math
import os
import re

RE_IT = re.compile(r"^\[it (\d+)\] snaps=(\d+) rollouts=(\d+) "
                   r"spread=([-+][\d.]+) pg=([-+]?[\d.]+) kl=([\d.]+) (\d+)s")
RE_EV = re.compile(r"^\[it (\d+)\] eval: 胜率 ([\d.]+)%, 场均 ([-+][\d.]+)")
RE_BASE = re.compile(r"^\[it 0\] init baseline: 胜率 ([\d.]+)%, 场均 ([-+][\d.]+)")


def parse(path):
    """只取最后一次 run(日志会被多次 append)。"""
    lines = open(path, encoding="utf-8", errors="replace").read().splitlines()
    starts = [i for i, l in enumerate(lines) if RE_BASE.match(l)]
    if starts:
        lines = lines[starts[-1]:]
    base = None
    its, evs = [], []
    for l in lines:
        m = RE_BASE.match(l)
        if m:
            base = (float(m.group(1)), float(m.group(2)))
            continue
        m = RE_IT.match(l)
        if m:
            its.append(dict(it=int(m.group(1)), spread=float(m.group(4)),
                            pg=float(m.group(5)), kl=float(m.group(6)),
                            sec=int(m.group(7))))
            continue
        m = RE_EV.match(l)
        if m:
            evs.append((int(m.group(1)), float(m.group(2)),
                        float(m.group(3))))
    return base, its, evs


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def sd(xs):
    if len(xs) < 2:
        return float("nan")
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def slope(xy):
    """最小二乘斜率 + 其标准误 + t 值。"""
    n = len(xy)
    if n < 3:
        return float("nan"), float("nan"), float("nan")
    mx = mean([x for x, _ in xy])
    my = mean([y for _, y in xy])
    sxx = sum((x - mx) ** 2 for x, _ in xy)
    sxy = sum((x - mx) * (y - my) for x, y in xy)
    if sxx == 0:
        return float("nan"), float("nan"), float("nan")
    b = sxy / sxx
    resid = [y - (my + b * (x - mx)) for x, y in xy]
    s2 = sum(r * r for r in resid) / (n - 2)
    seb = math.sqrt(s2 / sxx) if s2 > 0 else 0.0
    return b, seb, (b / seb if seb > 0 else float("nan"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="logs/v12_v*_s*.log")
    ap.add_argument("--eval-games", type=int, default=400)
    args = ap.parse_args()

    paths = sorted(glob.glob(args.glob))
    print(f"{'run':34s} {'it':>4s} {'base%':>6s} {'eval均值%':>9s} "
          f"{'差(pp)':>7s} {'z':>6s} {'最好%':>6s} {'KL末':>6s} {'s/it':>6s}")
    print("-" * 96)
    all_diffs = {}
    for p in paths:
        base, its, evs = parse(p)
        if base is None or not evs:
            continue
        wr = [w for _, w, _ in evs]
        se = math.sqrt(0.25 * 0.75 / args.eval_games) * 100
        # 各 eval 与 baseline 的差, 合并后的标准误
        diff = mean(wr) - base[0]
        se_diff = se * math.sqrt(1.0 / len(wr) + 1.0)
        z = diff / se_diff
        tag = os.path.basename(p).replace("v12_", "").replace(".log", "")
        cfg = tag.rsplit("_s", 1)[0]
        all_diffs.setdefault(cfg, []).append(diff)
        print(f"{tag:34s} {len(its):4d} {base[0]:6.1f} {mean(wr):9.2f} "
              f"{diff:+7.2f} {z:+6.2f} {max(wr):6.1f} "
              f"{its[-1]['kl'] if its else float('nan'):6.3f} "
              f"{mean([i['sec'] for i in its]):6.0f}")

    print(f"\n单次 eval({args.eval_games} 局)胜率标准误 = "
          f"{math.sqrt(0.25*0.75/args.eval_games)*100:.2f} pp")
    print(f"两次 eval 之差的标准误 = "
          f"{math.sqrt(2*0.25*0.75/args.eval_games)*100:.2f} pp")
    print(f"=> 95% 置信区间要检出的最小差异 ≈ "
          f"{1.96*math.sqrt(2*0.25*0.75/args.eval_games)*100:.1f} pp")

    print("\n按配置合并(各种子的 diff):")
    for cfg, ds in all_diffs.items():
        m, s = mean(ds), sd(ds)
        n = len(ds)
        t = m / (s / math.sqrt(n)) if n > 1 and s > 0 else float("nan")
        print(f"  {cfg:22s} n={n} diff={m:+.2f} pp (种子间 sd={s:.2f}) "
              f"t={t:+.2f}")

    print("\n趋势检验(胜率 ~ 迭代数, 各 run 独立最小二乘):")
    for p in paths:
        base, its, evs = parse(p)
        if not evs or len(evs) < 3:
            continue
        b, seb, t = slope([(i, w) for i, w, _ in evs])
        tag = os.path.basename(p).replace("v12_", "").replace(".log", "")
        print(f"  {tag:34s} 斜率 {b:+.4f} pp/迭代 (se {seb:.4f}, t={t:+.2f})")

    print("\n策略是否在动(KL 相对初始参考策略):")
    for p in paths:
        base, its, evs = parse(p)
        if not its:
            continue
        kl = [i["kl"] for i in its]
        tag = os.path.basename(p).replace("v12_", "").replace(".log", "")
        print(f"  {tag:34s} KL 首 {kl[0]:.4f} -> 末 {kl[-1]:.4f} "
              f"(峰值 {max(kl):.4f}), spread 均 "
              f"{mean([i['spread'] for i in its]):.3f}")


if __name__ == "__main__":
    main()
