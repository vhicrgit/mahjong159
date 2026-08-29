"""对拍: 原生 cheat_full vs 纯 Python cheat_full。

两级检查:
1. 逐决策: 同一批快照上, 两个 bot 的 choose_discard 必须同一张
2. 整局: 1 个 cheat_full + 3 个 v1 打完整局, 日志与得分必须完全一致
   (纯 Python 侧一次决策 ~6.6s, 所以局数不能多)
"""

import argparse
import copy
import random
import time

from ..game.engine import Game
from ..rules.ting import discard_options
from .bot_cheat import Bot as PyCheat
from .bot_cheat_native import NativeCheatFull
from .bot_native import NativeV1
from .bot_v1 import Bot as PyV1

CHEAT_FULL = dict(wall_lookahead=-1, see_opponents=True, rollout=True)


def make_snaps(n, seed0=771000):
    out = []
    gi = 0
    from .bot_native import NativeV10
    while len(out) < n and gi < n * 30:
        seed = seed0 + gi
        gi += 1
        g = Game(seed=seed, human_seat=-1)
        bots = {i: NativeV10(g, i) for i in range(4)}
        rng = random.Random(seed ^ 0xBEEF)
        target = rng.randint(3, 12)
        tc, guard = 0, 0
        while g.phase != "game_over" and guard < 500:
            guard += 1
            if g.phase == "discard_wait":
                tc += 1
                seat = g.turn
                if tc == target and len(discard_options(
                        list(g.players[seat].hand_counts))) >= 3:
                    g.log = []
                    out.append((copy.deepcopy(g), seat))
                    break
                g.action_discard(seat, bots[seat].choose_discard())
            else:
                s = list(g.pending_actions.keys())[0]
                b = bots[s]
                if g.pending_actions[s].get("gang") and \
                        b.decide_gang(g.last_discard, "ming"):
                    g.action_gang(s)
                elif g.pending_actions[s].get("peng") and \
                        b.decide_peng(g.last_discard):
                    g.action_peng(s)
                else:
                    g.action_pass(s)
    return out


def per_decision(n, seed0):
    snaps = make_snaps(n, seed0)
    bad = 0
    t_py = t_na = 0.0
    for g, seat in snaps:
        t0 = time.process_time()
        a = PyCheat(g, seat, **CHEAT_FULL).choose_discard()
        t_py += time.process_time() - t0
        t0 = time.process_time()
        b = NativeCheatFull(g, seat).choose_discard()
        t_na += time.process_time() - t0
        if a != b:
            bad += 1
            if bad <= 3:
                print(f"  DISCARD 分歧 seat={seat} py={a} native={b} "
                      f"hand={g.players[seat].hand} wall={len(g.wall)}")
    print(f"  逐决策: {bad}/{len(snaps)} 分歧")
    print(f"  cpu: py {t_py:.1f}s vs native {t_na:.2f}s -> "
          f"{t_py/max(t_na,1e-9):.0f}x "
          f"({t_py/max(len(snaps),1):.2f} vs {t_na/max(len(snaps),1):.3f} s/决策)")
    return bad


def play(g, hero, hero_bot, opp_cls):
    opp = {i: opp_cls(g, i) for i in range(4) if i != hero}
    guard = 0
    while g.phase != "game_over" and guard < 500:
        guard += 1
        if g.phase == "discard_wait":
            b = hero_bot if g.turn == hero else opp[g.turn]
            g.action_discard(g.turn, b.choose_discard())
        else:
            s = list(g.pending_actions.keys())[0]
            b = hero_bot if s == hero else opp[s]
            if g.pending_actions[s].get("gang") and \
                    b.decide_gang(g.last_discard, "ming"):
                g.action_gang(s)
            elif g.pending_actions[s].get("peng") and \
                    b.decide_peng(g.last_discard):
                g.action_peng(s)
            else:
                g.action_pass(s)
    return g


def whole_game(n, seed0):
    bad = 0
    t_py = t_na = 0.0
    for gi in range(n):
        seed = seed0 + gi
        g1 = Game(seed=seed, human_seat=-1)
        t0 = time.process_time()
        play(g1, 0, PyCheat(g1, 0, **CHEAT_FULL), PyV1)
        t_py += time.process_time() - t0
        g2 = Game(seed=seed, human_seat=-1)
        t0 = time.process_time()
        play(g2, 0, NativeCheatFull(g2, 0), NativeV1)
        t_na += time.process_time() - t0
        d1 = [p.score_delta for p in g1.players]
        d2 = [p.score_delta for p in g2.players]
        if g1.log != g2.log or d1 != d2:
            bad += 1
            for i, (x, y) in enumerate(zip(g1.log, g2.log)):
                if x != y:
                    print(f"  整局分歧 seed={seed} 第{i}步 py={x!r} "
                          f"native={y!r}")
                    break
            print(f"   得分 py={d1} native={d2}")
    print(f"  整局(1×cheat_full + 3×v1): {bad}/{n} 局不一致")
    print(f"  cpu: py {t_py:.1f}s vs native {t_na:.2f}s -> "
          f"{t_py/max(t_na,1e-9):.0f}x "
          f"({t_na/max(n,1):.2f} s/局 原生)")
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--decisions", type=int, default=8)
    ap.add_argument("--games", type=int, default=2)
    ap.add_argument("--seed0", type=int, default=771000)
    args = ap.parse_args()
    bad = 0
    if args.decisions:
        bad += per_decision(args.decisions, args.seed0)
    if args.games:
        bad += whole_game(args.games, args.seed0 + 300000)
    print("PARITY", "OK" if bad == 0 else "FAIL")
    raise SystemExit(0 if bad == 0 else 1)


if __name__ == "__main__":
    main()
