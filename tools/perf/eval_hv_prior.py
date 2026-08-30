"""集成评估: 学者+对手先验(HV-prior) vs 学者均匀假设(HV), 同种子配对。

先验臂: 座0 的 bot 在每次决策前, 用 3 个 OppTracker(软似然模式) 的后验构造
  held_exp[t] = Σ_对手 E[持有 t 的拷贝数]
  u_eff[t]    = max(0, 不可见[t] - held_exp[t])   (牌墙期望组成)
  然后走与学者完全相同的 argmin-E 决策(bot_hv 模块函数)。
基线臂: 座0 = 学者原版(均匀假设)。两臂同种子(同一牌墙)配对。

报告: 胜率/场均分差 + McNemar 精确检验。

用法: python -m tools.perf.eval_hv_prior --games 40 --procs 10
"""

import argparse
import math
import multiprocessing as mp
import time

import numpy as np

from backend.ai import bot_hv
from backend.ai.bot_hv import Bot as HVBot
from backend.analysis.opp_model import OppTracker
from backend.game.engine import Game

HERO = 0


class PriorBot:
    """学者 + 对手先验。trackers 由 harness 同步喂事件。"""

    def __init__(self, game, seat, trackers, beam, n_init):
        self.game = game
        self.seat = seat
        self.trackers = trackers
        self.memo = {}

    def _visible(self):
        visible = [0] * 28
        for q in self.game.players:
            for t in q.discards:
                visible[t] += 1
            for m in q.melds:
                visible[m["tile"]] += 3 if m["type"] == "peng" else 4
        for t, n in enumerate(self.game.players[self.seat].hand_counts):
            visible[t] += n
        return visible

    def _prior(self):
        held = [0.0] * 28
        for tr in self.trackers.values():
            ec = tr.expected_counts()
            for t in range(28):
                held[t] += ec[t]
        visible = self._visible()
        u_eff = [0.0] * 28
        for t in range(28):
            u_eff[t] = max(0.0, 4 - visible[t] - held[t])
        return u_eff, held

    def choose_discard(self):
        p = self.game.players[self.seat]
        u_eff, held = self._prior()
        t = bot_hv.choose_discard(p.hand_counts, self._visible(), 1.0,
                                  self.memo, u_eff=u_eff, held_exp=held)
        return t if t is not None else p.hand[-1]

    def decide_peng(self, tile):
        p = self.game.players[self.seat]
        u_eff, held = self._prior()
        return bot_hv.decide_peng(p.hand_counts, self._visible(), tile, 1.0,
                                  self.memo, u_eff=u_eff, held_exp=held)

    def decide_gang(self, tile, kind):
        return bot_hv.decide_gang(self.game.players[self.seat].hand_counts,
                                  tile, kind)


def play(seed, use_prior, beam=600, n_init=3000):
    g = Game(seed=seed, human_seat=-1)
    trackers = {}
    if use_prior:
        trackers = {s: OppTracker(s, g.players[HERO].hand_counts,
                                  n_init=n_init, beam=beam, policy=True,
                                  seed=seed * 10 + s)
                    for s in (1, 2, 3)}
        for tr in trackers.values():
            tr.notify_deal(g.dealer)
        hero = PriorBot(g, HERO, trackers, beam, n_init)
    else:
        hero = HVBot(g, HERO)
    bots = {HERO: hero}
    for i in range(1, 4):
        bots[i] = HVBot(g, i, memo={})

    def feed_draw(seat, tile):
        for tr in trackers.values():
            tr.notify_draw(seat, tile if seat == HERO else None,
                           g.wall_remaining())

    def feed_discard(seat, tile):
        for tr in trackers.values():
            tr.notify_discard(seat, tile, g.wall_remaining())

    def feed_claim(seat, action, tile, discarder):
        for tr in trackers.values():
            tr.notify_claim(seat, action, tile, discarder)

    guard = 0
    while g.phase != "game_over" and guard < 500:
        guard += 1
        if g.phase == "discard_wait":
            s = g.turn
            d = bots[s].choose_discard()
            feed_discard(s, d)
            ev = g.action_discard(s, d)
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
        if ev.get("event") in ("draw", "gang_draw"):
            feed_draw(ev["seat"], ev["tile"])
    return (1 if g.winner == HERO else 0, g.players[HERO].score_delta)


def work(payload):
    seed, beam, n_init = payload
    t0 = time.time()
    a = play(seed, True, beam, n_init)     # 先验臂
    b = play(seed, False)                  # 均匀臂
    return a, b, time.time() - t0


def mcnemar(wins_a, wins_b):
    b = sum(1 for x, y in zip(wins_a, wins_b) if x == 1 and y == 0)
    c = sum(1 for x, y in zip(wins_a, wins_b) if x == 0 and y == 1)
    n = b + c
    if n == 0:
        return b, c, 1.0
    k = min(b, c)
    p = 2 * min(1.0, sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n)
    return b, c, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=40)
    ap.add_argument("--seed0", type=int, default=995000)
    ap.add_argument("--procs", type=int, default=10)
    ap.add_argument("--beam", type=int, default=600)
    ap.add_argument("--n-init", type=int, default=3000)
    args = ap.parse_args()

    seeds = [(args.seed0 + i, args.beam, args.n_init)
             for i in range(args.games)]
    t0 = time.time()
    if args.procs > 1:
        with mp.Pool(args.procs) as pool:
            rets = pool.map(work, seeds, chunksize=1)
    else:
        rets = [work(s) for s in seeds]
    wall = time.time() - t0

    A = [r[0] for r in rets]   # prior
    B = [r[1] for r in rets]   # uniform
    t_pair = np.mean([r[2] for r in rets])

    def stat(rows):
        w = np.array([r[0] for r in rows], dtype=float)
        s = np.array([r[1] for r in rows], dtype=float)
        n = len(w)
        return (w.mean() * 100,
                1.96 * math.sqrt(w.mean() * (1 - w.mean()) / n) * 100,
                s.mean())

    for name, rows in (("学者+先验", A), ("学者(均匀)", B)):
        wr, ci, sc = stat(rows)
        print(f"{name:12s} 胜率 {wr:.1f}% (±{ci:.1f})  场均 {sc:+.2f}")
    b_, c_, p_ = mcnemar([r[0] for r in A], [r[0] for r in B])
    diff = np.array([a[1] for a in A]) - np.array([b[1] for b in B])
    print(f"配对: 分歧对 b={b_} c={c_}, McNemar p={p_:.4f}; "
          f"分差均值差 {diff.mean():+.2f} (±{1.96 * diff.std() / math.sqrt(len(diff)):.2f})")
    print(f"局数={len(A)} 墙钟={wall:.0f}s 均时对(两臂)={t_pair:.1f}s")


if __name__ == "__main__":
    main()
