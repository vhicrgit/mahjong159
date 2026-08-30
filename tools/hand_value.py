"""牌型价值分析器: 估算当前牌型"期望还要几巡才能胡"。

模型:
- 信息集: 自己手牌 + 所有人弃牌 + 所有副露(明碰/杠)。对手手牌不推测。
  unseen[t] = 4 - visible[t]
- 两个进张通道:
  * 自摸: 摸到有效张, 速率 = 池内剩余张数
  * 碰:   手里有对子(h[t]==2)且碰后最优向听严格降低时, 对手喂牌也是一条进张路径;
          速率 = u[t] × 3ρ, 其中 ρ = 对手摸到该牌后打出来的概率(ρ=1 → 总共 ×4,
          即"自摸+三家喂碰"的最乐观口径)。碰是独立状态转移: 对子变副露、
          暗手 -2, 再打出最优一张 —— 不只是速率加成
- 期望巡数的主估计用等待时间递推: 有效事件总权重 U、池子 N 张时,
  等到下一个有效事件的期望 = (N+1)/(U+1)(无放回首次命中的精确期望),
  命中后按 v10 牌效打出最优一张, 递归到新牌型。听牌后有效张=胡牌张,
  收敛到 (N+1)/(W+1)。ρ=0(纯自摸)时与 MC 单人模拟同口径, 已验证。
- 换型层(默认开, --no-kaizen 关闭): 摸到不降向听但让有效张变宽 ≥2 的牌
  也算一次进展。预算按整条路径计(kai_max 次, 不随降向听重置), 每状态只保留
  进张净增最多的 kai_topk 个分支 —— 旧版"连续次数重置"语义会让高向听手牌的
  DP 状态爆炸到 64 万(单手 280s), 截断后 C 引擎整表秒级。修正"改良盲"偏差:
  无此层时系统性偏高 ~+0.6 巡(tools/perf/feas_kaizen.py)。
- 计算引擎: 默认走 C(libmjcore.so, 由 mobile/wasm/mjcore.c 编译, 与 Python
  逐位一致, 对拍见 tools/perf/test_hv_c_parity.py); --engine py 回退 Python;
  --procs N 按候选多核并行(kai_max=2 全表 18s -> ~7s)。
- 牌型枚举: DFS n 进张内可达的和牌型, 按所需组合聚合展示(含独立期望)。

输出 E[带碰] 与 E[纯自摸] 双口径, Δ碰 表示这副牌多依赖碰牌。

用法:
  python -m tools.hand_value --seed 961000 --turn 4 --seat 0
  python -m tools.hand_value --hand "2条 2条 ..." --discards "..." --rho 0.6 --rho-sweep
"""

import argparse
import math
import re
import sys

import numpy as np

from backend.native import native

RED = 27
SUITS = "条饼万"


def tile_id(s: str) -> int:
    s = s.strip()
    if s in ("红中", "红", "中"):
        return RED
    m = re.match(r"^(\d)(条|饼|万)$", s)
    if not m:
        raise ValueError(f"无法识别的牌: {s!r}")
    r, su = int(m.group(1)), m.group(2)
    return SUITS.index(su) * 9 + (r - 1)


def tile_name(t: int) -> str:
    if t == RED:
        return "红中"
    return f"{t % 9 + 1}{SUITS[t // 9]}"


def parse_tiles(s: str) -> list[int]:
    if not s or not s.strip():
        return []
    return [tile_id(x) for x in re.split(r"[\s,/，、]+", s.strip()) if x]


def counts_of(tiles) -> list[int]:
    c = [0] * 28
    for t in tiles:
        c[t] += 1
    return c



from backend.analysis.hand_value import (  # noqa: 核心已移入 backend
    HandAnalyzer, useful_set, _USEFUL_SET_CACHE)

def build_from_args(args):
    """返回 (hand_counts, visible_counts, 描述)。"""
    if args.seed is not None:
        from tools.perf.arbitrate_961000 import replay_to
        g = replay_to(args.seed, args.seat, args.turn)
        p = g.players[args.seat]
        visible = [0] * 28
        for q in g.players:
            for t in q.discards:
                visible[t] += 1
            for mm in q.melds:
                visible[mm["tile"]] += 3 if mm["type"] == "peng" else 4
        for t, n in enumerate(p.hand_counts):
            visible[t] += n
        desc = f"seed={args.seed} 座{args.seat} 第{args.turn}巡 墙剩{g.wall_remaining()}"
        return list(p.hand_counts), visible, desc
    hand = counts_of(parse_tiles(args.hand))
    visible = list(hand)
    for t in parse_tiles(args.discards):
        visible[t] += 1
    for grp in (args.opp_discards or "").split("/"):
        for t in parse_tiles(grp):
            visible[t] += 1
    for t in parse_tiles(args.meld_tiles):
        visible[t] += 3   # 碰按3张计; 杠请把该牌写两次
    return hand, visible, "手动输入"


def _c_worker(payload):
    """多核 worker: 一个候选牌的 (rho, 0) 双口径 E。spawn 子进程各自加载 C 库,
    memo 互不相通(共享 memo 无法跨进程, 各候选独享一份)。"""
    (hand, visible, d, rho, kaizen, kai_margin, kai_max, kai_topk) = payload
    from backend.analysis import hv_native
    hv_native.set_hand(hand, visible, rho=rho, kaizen=kaizen,
                       kai_margin=kai_margin, kai_max=kai_max,
                       kai_topk=kai_topk)
    e1 = hv_native.e_after_discard(d)
    hv_native.set_hand(hand, visible, rho=0.0, kaizen=kaizen,
                       kai_margin=kai_margin, kai_max=kai_max,
                       kai_topk=kai_topk)
    e0 = hv_native.e_after_discard(d)
    return d, e0, e1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--turn", type=int, default=4)
    ap.add_argument("--seat", type=int, default=0)
    ap.add_argument("--hand", type=str, default="")
    ap.add_argument("--discards", type=str, default="", help="自己的弃牌")
    ap.add_argument("--opp-discards", type=str, default="",
                    help="其他家弃牌, 用 / 分隔三家")
    ap.add_argument("--meld-tiles", type=str, default="",
                    help="副露里的牌(碰写1次按3张计)")
    ap.add_argument("--rho", type=float, default=1.0,
                    help="对手摸到你要碰的牌后打出来的概率; 0=纯自摸, 1=一定喂")
    ap.add_argument("--rho-sweep", action="store_true",
                    help="对弃牌表做 ρ∈{0,0.3,0.6,1.0} 敏感性分析")
    ap.add_argument("--draws", type=int, default=0,
                    help="牌型枚举的进张深度(0=自动: 向听+2, 封顶4)")
    ap.add_argument("--no-peng-x4", action="store_true",
                    help="[兼容旧用法] 等价于 --rho 0")
    ap.add_argument("--kaizen", dest="kaizen", action="store_true",
                    default=True, help="换型层(默认开)")
    ap.add_argument("--no-kaizen", dest="kaizen", action="store_false",
                    help="关掉换型层")
    ap.add_argument("--kaizen-margin", type=int, default=2,
                    help="换型判定的进张放宽阈值")
    ap.add_argument("--kaizen-max", type=int, default=1,
                    help="每条路径的换型预算(全局, 不重置); 2 更准但慢 ~10 倍")
    ap.add_argument("--kaizen-topk", type=int, default=6,
                    help="每状态最多保留的换型分支数(按进张净增排序); 0=不截断")
    ap.add_argument("--engine", choices=["auto", "c", "py"], default="auto",
                    help="E 的计算引擎: auto=优先 C(libmjcore, 毫秒级), py=纯 Python")
    ap.add_argument("--procs", type=int, default=1,
                    help="C 引擎下按候选并行计算的进程数(换型层下全表可 4-5 倍提速)")
    ap.add_argument("--mc", type=int, default=5000, help="MC 模拟局数, 0=跳过")
    ap.add_argument("--max-pattern-time", type=int, default=12)
    args = ap.parse_args()

    rho = 0.0 if args.no_peng_x4 else args.rho
    hand, visible, desc = build_from_args(args)
    az = HandAnalyzer(hand, visible, rho=rho, kaizen=args.kaizen,
                      kai_margin=args.kaizen_margin, kai_max=args.kaizen_max,
                      kai_topk=args.kaizen_topk)
    az0 = HandAnalyzer(hand, visible, rho=0.0, kaizen=args.kaizen,
                       kai_margin=args.kaizen_margin, kai_max=args.kaizen_max,
                       kai_topk=args.kaizen_topk)

    print(f"=== {desc} ===")
    print(f"手牌({sum(hand)}张): {' '.join(tile_name(t) for t in range(28) for _ in range(hand[t]))}")
    tot_u = sum(az.u0)
    print(f"未见牌共 {tot_u} 张  ρ={rho:.2f} "
          f"(对子成刻速率 ×{1+3*rho:.1f}; ρ=0 为纯自摸口径)")

    ntile = sum(hand)
    if ntile % 3 == 2:
        # 待出牌状态: 每张候选的期望巡数, 碰/不碰双口径
        # C 引擎(libmjcore): 换型层下整表亚秒; 不可用时回退 Python
        c_es = None
        if args.engine in ("auto", "c"):
            from backend.analysis import hv_native
            if args.procs > 1:
                # 多核: 只探测库可用性, 计算全在子进程
                if hv_native.lib() is not None:
                    import multiprocessing as mp
                    tasks = [(hand, visible, d, rho, args.kaizen,
                              args.kaizen_margin, args.kaizen_max,
                              args.kaizen_topk)
                             for d in range(28) if hand[d] > 0]
                    ctx = mp.get_context("spawn")
                    with ctx.Pool(args.procs) as pool:
                        rets = pool.map(_c_worker, tasks)
                    e0s = {d: e0 for d, e0, _e1 in rets}
                    e1s = {d: e1 for d, _e0, e1 in rets}
                    c_es = (e0s, e1s)
            elif hv_native.set_hand(hand, visible, rho=rho, kaizen=args.kaizen,
                                    kai_margin=args.kaizen_margin,
                                    kai_max=args.kaizen_max,
                                    kai_topk=args.kaizen_topk):
                e1s = {d: hv_native.e_after_discard(d)
                       for d in range(28) if hand[d] > 0}
                # rho 一切换就清 C 侧 memo, 所以按 rho 分两遍算
                hv_native.set_hand(hand, visible, rho=0.0,
                                   kaizen=args.kaizen,
                                   kai_margin=args.kaizen_margin,
                                   kai_max=args.kaizen_max,
                                   kai_topk=args.kaizen_topk)
                e0s = {d: hv_native.e_after_discard(d)
                       for d in range(28) if hand[d] > 0}
                c_es = (e0s, e1s)
        print("\n=== 各弃牌候选的期望巡数(越小越好) ==="
              + ("  [C 引擎]" if c_es else "  [Python 引擎]"))
        print(f"{'打出':>5s} {'向听':>4s} {'有效张':>6s} "
              f"{'E[无碰]':>8s} {'E[带碰]':>8s} {'Δ碰':>7s}")
        rows = []
        h = list(hand)
        for d in range(28):
            if h[d] <= 0:
                continue
            hd = list(hand)
            hd[d] -= 1
            key = (tuple(hd), az.u0)
            if c_es is not None:
                e0, e1 = c_es[0][d], c_es[1][d]
            else:
                e0 = az0.E(*key)
                e1 = az.E(*key)
            s = native.shanten(hd)
            _, uf = az._useful(key[0], az.u0)
            rows.append((d, s, sum(uf.values()), e0, e1))
        rows.sort(key=lambda r: r[4])
        for d, s, uw, e0, e1 in rows:
            print(f"{tile_name(d):>5s} {s:4d} {uw:6d} "
                  f"{e0:8.2f} {e1:8.2f} {e0-e1:+7.2f}")
        best = rows[0][0]
        hand_after = list(hand)
        hand_after[best] -= 1
        hand13 = hand_after
        print(f"\n(以下按带碰口径最优的打出 {tile_name(best)} 后的牌型展开)")

        if args.rho_sweep:
            print("\n=== ρ 敏感性(排序稳定性) ===")
            rhos = [0.0, 0.3, 0.6, 1.0]
            azs = [HandAnalyzer(hand, visible, rho=r, kaizen=args.kaizen,
                                kai_margin=args.kaizen_margin,
                                kai_max=args.kaizen_max) for r in rhos]
            print(f"{'打出':>5s} " + " ".join(f"ρ={r:<4.1f}" for r in rhos))
            for d in range(28):
                if hand[d] <= 0:
                    continue
                hd = list(hand)
                hd[d] -= 1
                key = (tuple(hd), az.u0)
                es = [a.E(*key) for a in azs]
                print(f"{tile_name(d):>5s} " +
                      " ".join(f"{e:6.2f}" for e in es))
            # 各 ρ 下的最优选择
            winners = []
            for a in azs:
                best_d, best_e = None, 1e18
                for d in range(28):
                    if hand[d] <= 0:
                        continue
                    hd = list(hand)
                    hd[d] -= 1
                    e = a.E(tuple(hd), a.u0)
                    if e < best_e:
                        best_d, best_e = d, e
                winners.append(tile_name(best_d))
            print("各 ρ 最优: " + "  ".join(
                f"ρ={r}:{w}" for r, w in zip(rhos, winners)))
    else:
        hand13 = hand

    s0 = native.shanten(list(hand13))
    _, uf = az._useful(tuple(hand13), az.u0)
    pengs = az._peng_transitions(tuple(hand13), az.u0)
    print(f"\n=== 当前牌型(出牌后 {sum(hand13)} 张): 向听 {s0} ===")
    print("有效张(自摸): " + " ".join(
        f"{tile_name(t)}×{az.u0[t]}" for t in sorted(uf)))
    if pengs:
        print("可碰对子:      " + " ".join(
            f"{tile_name(t)}×{az.u0[t]}(权重×{1+3*rho:.0f}→碰后打{tile_name(d)})"
            for t, (w, d, bs) in sorted(pengs.items())))
    e_tot = az.E(tuple(hand13), az.u0)
    e_np = az0.E(tuple(hand13), az.u0)
    print(f"\n>>> 期望巡数到胡: {e_tot:.2f} (带碰) / {e_np:.2f} (纯自摸) "
          f"| Δ碰 = {e_np - e_tot:+.2f}")

    nd = args.draws or min(s0 + 2, 4)
    pats, nodes = az.enum_patterns(tuple(hand13), nd)
    print(f"\n=== {nd} 进张内可达的和牌型(按所需组合聚合, 枚举节点 {nodes}) ===")
    print(f"{'需要摸的牌':<28s} {'张数':>4s} {'牌型数':>6s} {'独立期望':>8s}")
    for p in pats[:args.max_pattern_time]:
        ts = " ".join(f"{tile_name(t)}×{c}(剩{az.u0[t]})"
                      for t, c in p["tiles"])
        et = az.pattern_time(p["tiles"])
        ets = f"{et:.2f}" if et is not None else "-"
        print(f"{ts:<28s} {p['draws']:>4d} {p['n_pat']:>6d} {ets:>8s}")
    if len(pats) > args.max_pattern_time:
        print(f"... 共 {len(pats)} 种组合")

    if args.mc > 0:
        mcv = az.mc(tuple(hand13), args.mc)
        print(f"\nMC 单人摸打模拟({args.mc} 次, v10 牌效弃牌, 无碰): "
              f"{mcv:.2f} 巡 | 递推(无碰) {e_np:.2f} | 差 {e_np - mcv:+.2f}"
              f"  —— MC 不含碰, 校验的是 ρ=0 口径")


if __name__ == "__main__":
    main()
