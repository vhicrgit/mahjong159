"""血战到底 + 名次奖励到底把训练信号变干净了多少 —— 真规则测量。

之前用"首胡时按剩余向听排序"当名次代理, 测出 R² 0.103 -> 0.281 (2.7x),
那是乐观上界(假设后续名次由当前手牌进度决定)。这里跑真规则给出实数。

三个奖励口径, 同一批 seed:
  首胡·得分     现在训练用的 (基准)
  血战·得分     血战到底下的调整得分
  血战·名次奖励 +3/+1/-1/-3, 未胡者并列平分
指标: R²(巡k 状态E -> 奖励) —— 奖励里有多少能被状态解释。越高越可学。

顺带报"首胡那一刻另外三家的向听分布", 即停在首胡扔掉了多少名次信息。

用法: python -m tools.perf.diag_bloody --games 600
"""

import argparse

import numpy as np

from backend.ai.bot_native import NativeV31
from backend.analysis import hv_native
from backend.game.engine import Game
from backend.native import native
from backend.rules.tiles import TILE_COUNT


def visible(g, s):
    v = [0] * TILE_COUNT
    for q in g.players:
        for t in q.discards:
            v[t] += 1
        for m in q.melds:
            v[m["tile"]] += 3 if m["type"] == "peng" else 4
    for t, n in enumerate(g.players[s].hand_counts):
        v[t] += n
    return v


def state_e(g, s, kai=0):
    """当前 14 张手牌的最优弃牌后 E(期望巡数); 非 3n+2 返回 nan。"""
    h = list(g.players[s].hand_counts)
    if sum(h) % 3 != 2:
        return float("nan")
    vv = visible(g, s)
    best = float("inf")
    for t in range(TILE_COUNT):
        if h[t]:
            hv_native.set_hand(h, vv, 1.0, kai > 0, 2, kai, 6)
            best = min(best, hv_native.e_after_discard(t))
    return best


def adjusted(g, s):
    """放杠不罚(与训练奖励口径一致)。"""
    v = float(g.players[s].score_delta)
    for rec in g.gang_records:
        if rec["kind"] == "ming" and rec["from"] == s:
            v += 3
    return v


def gang_own(g, s):
    """自己杠的收入(明杠+3, 暗杠/补杠向当时在场的每人各收1); 放杠不罚。"""
    v = 0.0
    for rec in g.gang_records:
        if rec["seat"] != s:
            continue
        if rec["kind"] == "ming":
            v += 3.0
        else:
            v += float(len([x for x in rec.get("active", [0, 1, 2, 3])
                            if x != s]))
    return v


def rollout(seed, bloody, mid, probe_first_hu=False):
    g = Game(seed=seed, human_seat=-1, bloody=bloody)
    bots = {s: NativeV31(g, s) for s in range(4)}
    nd = {s: 0 for s in range(4)}
    me = {s: float("nan") for s in range(4)}
    first_hu_sh = None
    guard = 0
    while g.phase != "game_over" and guard < 800:
        guard += 1
        if g.phase == "discard_wait":
            s = g.turn
            nd[s] += 1
            if nd[s] == mid:
                me[s] = state_e(g, s)
            g.action_discard(s, bots[s].choose_discard())
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
        if probe_first_hu and first_hu_sh is None and g.finished:
            w = g.finished[0]
            first_hu_sh = sorted(native.shanten(list(g.players[x].hand_counts))
                                 for x in range(4) if x != w)
    return g, me, first_hu_sh


def r2(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 30 or x[ok].std() == 0 or y[ok].std() == 0:
        return float("nan")
    return float(np.corrcoef(x[ok], y[ok])[0, 1] ** 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=600)
    ap.add_argument("--seed0", type=int, default=14000000)
    ap.add_argument("--mid", type=int, default=5, help="第几次弃牌取状态 E")
    ap.add_argument("--save")
    ap.add_argument("--load", nargs="+")
    args = ap.parse_args()

    if args.load:
        cols = {}
        for p in args.load:
            d = np.load(p)
            for k in d.files:
                cols.setdefault(k, []).append(d[k])
        report({k: np.concatenate(v) for k, v in cols.items()}, args.mid)
        return

    rows = {k: [] for k in ("e_f", "g_f", "rr_f", "e_b", "g_b", "rr_b",
                            "rank_b", "gw_b")}
    hu_sh = []
    for i in range(args.games):
        seed = args.seed0 + i
        gf, mef, _ = rollout(seed, False, args.mid)
        gb, meb, fh = rollout(seed, True, args.mid, probe_first_hu=True)
        rrf, rrb = gf.rank_rewards(), gb.rank_rewards()
        if fh is not None:
            hu_sh.append(fh)
        for s in range(4):
            rows["e_f"].append(mef[s])
            rows["g_f"].append(adjusted(gf, s))
            rows["rr_f"].append(rrf[s])
            rows["e_b"].append(meb[s])
            rows["g_b"].append(adjusted(gb, s))
            rows["rr_b"].append(rrb[s])
            rows["rank_b"].append(float(gb.ranks[s]))
            rows["gw_b"].append(gang_own(gb, s))
        if (i + 1) % 100 == 0:
            print(f"  ...{i + 1}/{args.games}", flush=True)

    cols = {k: np.array(v, float) for k, v in rows.items()}
    if hu_sh:
        cols["hu_sh"] = np.array(hu_sh, float).reshape(-1)
    if args.save:
        np.savez(args.save, **cols)
        print(f"已存 {args.save}")
    report(cols, args.mid)


def report(c, mid):
    n = len(c["g_f"]) // 4
    print(f"\n局数 {n}  单元 {len(c['g_f'])}")
    if "hu_sh" in c:
        h = c["hu_sh"].reshape(-1, 3)
        print(f"\n首胡那一刻另外三家的向听: 最好 {h[:, 0].mean():.2f}  "
              f"中 {h[:, 1].mean():.2f}  最差 {h[:, 2].mean():.2f}   "
              f"至少一家已听牌 {(h[:, 0] <= 0).mean():.1%}")

    print(f"\n=== R²(巡{mid} 状态E -> 奖励) ===")
    base = r2(c["e_f"], c["g_f"])
    tab = [("首胡·得分 (当前基准)", c["e_f"], c["g_f"]),
           ("首胡·名次奖励(+3/-1/-1/-1)", c["e_f"], c["rr_f"]),
           ("血战·得分", c["e_b"], c["g_b"]),
           ("血战·名次奖励(+3/+1/-1/-3)", c["e_b"], c["rr_b"]),
           ("血战·名次(数值)", c["e_b"], c["rank_b"])]
    if "gw_b" in c:
        for w in (0.25, 0.5, 1.0):
            tab.append((f"血战·名次 + {w:g}×自己杠分",
                        c["e_b"], c["rr_b"] + w * c["gw_b"]))
    for name, x, y in tab:
        v = r2(x, y)
        y_ok = np.asarray(y, float)
        print(f"  {name:28s} R² {v:.4f}  ({v / base:5.2f}x)   "
              f"奖励标准差 {y_ok.std():.2f}")

    print("\n=== 等效样本量(信号/噪声比 = R²/(1-R²), 相对基准) ===")
    b = base / (1 - base)
    for name, x, y in tab:
        v = r2(x, y)
        print(f"  {name:28s} {(v / (1 - v)) / b:5.2f}x")

    print(f"\n奖励之间的相关: 血战名次 vs 首胡得分 "
          f"{np.corrcoef(c['rr_b'], c['g_f'])[0, 1]:+.3f}   "
          f"血战名次 vs 血战得分 {np.corrcoef(c['rr_b'], c['g_b'])[0, 1]:+.3f}")


if __name__ == "__main__":
    main()
