"""对手牌型分析器(OppTracker)的对局验证: 学者 bot 作对手, 对照真实手牌算精度。

每局: 座0 = v31n(观察者), 座1..3 = 学者(bot_hv)。跟踪器只看公开信息+座0手牌。
每个对手出牌事件后记录:
  brier_hold   每牌"持有概率" vs 真值(0/1) 的 Brier
  count_mae    每牌期望持有数 vs 真实拷贝数 的平均绝对误差
  shanten_mae  后验期望向听 vs 真实向听
  tenpai_brier 听牌概率 vs 真实是否听牌
  wait_mass    真实听牌时, 后验听口分布落在真实听口上的总概率
  cover        真实手牌是否在粒子集中(诊断 beam 是否丢真手)

--policy / --no-policy: 对照组关掉策略反演(纯计数基线)。
--show: 只打一局并逐巡打印跟踪器输出与真实手牌。

用法:
  python -m tools.perf.eval_opp_model --games 30 --procs 10
  python -m tools.perf.eval_opp_model --show --seed0 961000
"""

import argparse
import math
import multiprocessing as mp
import time

import numpy as np

from backend.ai.bot_hv import Bot as HVBot
from backend.ai.bot_native import NativeV31
from backend.analysis.opp_model import OppTracker, _counts_str
from backend.game.engine import Game
from backend.native import native
from backend.rules.ting import waiting_tiles
from backend.rules.tiles import tile_name

HERO = 0


def run_game(seed, args, verbose=False):
    """打一局并收集全部跟踪指标。返回 (rows, diag)。"""
    g = Game(seed=seed, human_seat=-1)
    hv_memos = {i: {} for i in range(1, 4)}
    bots = {HERO: NativeV31(g, HERO)}
    for i in range(1, 4):
        bots[i] = HVBot(g, i, memo=hv_memos[i])
    trackers = {s: OppTracker(s, g.players[HERO].hand_counts,
                              n_init=args.n_init, beam=args.beam,
                              policy=args.policy, seed=seed * 10 + s)
                for s in (1, 2, 3)}
    for tr in trackers.values():
        tr.notify_deal(g.dealer)

    rows = []
    t_filter = 0.0
    opp_turn = {1: 0, 2: 0, 3: 0}

    def feed_draw(seat, tile):
        for tr in trackers.values():
            tr.notify_draw(seat, tile if seat == HERO else None,
                           g.wall_remaining())

    def feed_discard(seat, tile):
        nonlocal t_filter
        for s, tr in trackers.items():
            t0 = time.perf_counter()
            tr.notify_discard(seat, tile, g.wall_remaining())
            if s == seat:
                t_filter += time.perf_counter() - t0

    def feed_claim(seat, action, tile, discarder):
        for tr in trackers.values():
            tr.notify_claim(seat, action, tile, discarder)

    def wall_errors():
        """牌墙组成预测: 先验 u_eff vs 均匀 unseen, 对照真实牌墙。"""
        wall_true = [0] * 28
        for t in g.wall:
            wall_true[t] += 1
        vis = [0] * 28
        for q in g.players:
            for t in q.discards:
                vis[t] += 1
            for m in q.melds:
                vis[m["tile"]] += 3 if m["type"] == "peng" else 4
        for t, n in enumerate(g.players[HERO].hand_counts):
            vis[t] += n
        unseen = [max(0, 4 - vis[t]) for t in range(28)]
        out = {}
        for beta in (1.0, 3.0):
            held = [0.0] * 28
            for tr2 in trackers.values():
                ec = tr2.expected_counts(beta=beta)
                for t in range(28):
                    held[t] += ec[t]
            u_eff = [max(0.0, unseen[t] - held[t]) for t in range(28)]
            out[beta] = float(np.mean([
                abs(u_eff[t] - wall_true[t]) for t in range(28)]))
        unif = float(np.mean([abs(unseen[t] - wall_true[t])
                              for t in range(28)]))
        return out[1.0], out[3.0], unif

    def snapshot(opp):
        """对手(opp)刚出完牌, 对照真手记录一行指标。"""
        tr = trackers[opp]
        truth = tuple(g.players[opp].hand_counts)
        hp = tr.hold_probs()
        ec = tr.expected_counts()
        sh_dist = tr.shanten_dist()
        e_sh = sum(k * v for k, v in sh_dist.items())
        t_sh = native.shanten(list(truth))
        tp = tr.tenpai_prob()
        wm = None
        if t_sh == 0:
            tw = set(waiting_tiles(list(truth)))
            wp = tr.wait_probs()
            wm = sum(wp.get(t, 0.0) for t in tw)
        we = wall_errors()
        rows.append({
            "opp": opp, "turn": opp_turn[opp],
            "n_par": len(tr.particles),
            "brier_hold": float(np.mean([
                (hp[t] - (1 if truth[t] > 0 else 0)) ** 2
                for t in range(28)])),
            "count_mae": float(np.mean([
                abs(ec[t] - truth[t]) for t in range(28)])),
            "shanten_mae": abs(e_sh - t_sh),
            "tenpai_brier": (tp - (1 if t_sh == 0 else 0)) ** 2,
            "wait_mass": wm,
            "cover": truth in tr.particles,
            "wall_err_prior": we[0], "wall_err_beta3": we[1],
            "wall_err_unif": we[2],
        })
        if verbose:
            print(f"\n--- 座{opp} 第{opp_turn[opp]}巡打出后 "
                  f"(真实手牌: {_counts_str(truth)}"
                  f"{' 副露:' + str(len(g.players[opp].melds)) if g.players[opp].melds else ''})")
            print(tr.summary())

    guard = 0
    while g.phase != "game_over" and guard < 500:
        guard += 1
        if g.phase == "discard_wait":
            s = g.turn
            d = bots[s].choose_discard()
            feed_discard(s, d)
            ev = g.action_discard(s, d)
            if s in trackers:
                opp_turn[s] += 1
                snapshot(s)
        else:
            s = list(g.pending_actions.keys())[0]
            pend = g.pending_actions[s]
            tile, discarder = g.last_discard, g.last_discarder
            b = bots[s]
            if pend.get("gang") and b.decide_gang(tile, "ming"):
                feed_claim(s, "gang", tile, discarder)
                ev = g.action_gang(s)
            elif pend.get("peng") and b.decide_peng(tile):
                feed_claim(s, "peng", tile, discarder)
                ev = g.action_peng(s)
            else:
                feed_claim(s, None, tile, discarder)
                ev = g.action_pass(s)
        # 摸牌事件透传(含杠后补牌)
        if ev.get("event") == "draw":
            feed_draw(ev["seat"], ev["tile"])
        elif ev.get("event") == "gang_draw":
            feed_draw(ev["seat"], ev["tile"])

    diag = {"t_filter": t_filter,
            "winner": g.winner}
    if verbose:
        print(f"\n结局: winner={g.winner} kind={g.win_kind} "
              f"过滤耗时={t_filter:.1f}s")
    return rows, diag


def work(payload):
    seed, args = payload
    rows, diag = run_game(seed, args)
    return rows, diag


def bucket(turn):
    if turn <= 2:
        return "1-2 "
    if turn <= 5:
        return "3-5 "
    if turn <= 8:
        return "6-8 "
    return "9+  "


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--seed0", type=int, default=980000)
    ap.add_argument("--procs", type=int, default=10)
    ap.add_argument("--beam", type=int, default=300)
    ap.add_argument("--n-init", type=int, default=2000)
    ap.add_argument("--policy", action="store_true", default=True)
    ap.add_argument("--no-policy", dest="policy", action="store_false")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    if args.show:
        t0 = time.time()
        rows, diag = run_game(args.seed0, args, verbose=True)
        print(f"\n总耗时 {time.time() - t0:.1f}s")
        return

    seeds = [args.seed0 + i for i in range(args.games)]
    t0 = time.time()
    if args.procs > 1:
        with mp.Pool(args.procs) as pool:
            rets = pool.map(work, [(s, args) for s in seeds], chunksize=1)
    else:
        rets = [work((s, args)) for s in seeds]
    all_rows = [r for rows, _ in rets for r in rows]
    tf = sum(d["t_filter"] for _, d in rets)
    wall = time.time() - t0

    print(f"局数={len(rets)} policy={args.policy} beam={args.beam} "
          f"n_init={args.n_init}  墙钟={wall:.0f}s "
          f"(过滤总耗时 {tf:.0f}s, 均 {tf / max(1, len(rets)):.1f}s/局)")

    def agg(rows):
        n = len(rows)
        if n == 0:
            return None
        wm = [r["wait_mass"] for r in rows if r["wait_mass"] is not None]
        return (f"n={n:4d} brier={np.mean([r['brier_hold'] for r in rows]):.4f} "
                f"cnt_mae={np.mean([r['count_mae'] for r in rows]):.3f} "
                f"向听mae={np.mean([r['shanten_mae'] for r in rows]):.3f} "
                f"听牌brier={np.mean([r['tenpai_brier'] for r in rows]):.4f} "
                f"听口命中={np.mean(wm) if wm else float('nan'):.3f}({len(wm)}) "
                f"覆盖={np.mean([r['cover'] for r in rows]):.1%} "
                f"粒子={np.mean([r['n_par'] for r in rows]):.0f} "
                f"墙err(先验/β3/均匀)="
                f"{np.mean([r['wall_err_prior'] for r in rows]):.3f}/"
                f"{np.mean([r['wall_err_beta3'] for r in rows]):.3f}/"
                f"{np.mean([r['wall_err_unif'] for r in rows]):.3f}")

    print("\n按巡段:")
    for b in ("1-2 ", "3-5 ", "6-8 ", "9+  "):
        sub = [r for r in all_rows if bucket(r["turn"]) == b]
        a = agg(sub)
        if a:
            print(f"  巡{b}: {a}")
    print(f"  全部 : {agg(all_rows)}")


if __name__ == "__main__":
    main()
