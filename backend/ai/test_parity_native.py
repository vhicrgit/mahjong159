"""对拍: 原生 Bot 的每个决策 vs 纯 Python Bot。

做法: 由 Python Bot 驱动整局(保证走到真实分布, 含副露后的短手牌),
每个决策点同时问原生 Bot, 逐个比较 choose_discard / decide_peng / decide_gang。
再做一次"原生 Bot 独立驱动"的整局对比, 校验日志与得分完全一致。
"""

import argparse
import random
import time

from ..game.engine import Game
from .bot_native import NativeV1, NativeV10, NativeV31
from .bot_v1 import Bot as PyV1
from .bot_v10 import Bot as PyV10
from .bot_v31 import Bot as PyV31

PAIRS = {
    "v1": (PyV1, NativeV1),
    "v10": (PyV10, NativeV10),
    "v31": (PyV31, NativeV31),
}


def per_decision(kind: str, n_games: int, seed0: int):
    PyCls, NaCls = PAIRS[kind]
    bad = {"discard": 0, "peng": 0, "gang": 0}
    cnt = {"discard": 0, "peng": 0, "gang": 0}
    melded_discards = 0
    for gi in range(n_games):
        g = Game(seed=seed0 + gi, human_seat=-1)
        py = {i: PyCls(g, i) for i in range(4)}
        na = {i: NaCls(g, i) for i in range(4)}
        guard = 0
        while g.phase != "game_over" and guard < 500:
            guard += 1
            if g.phase == "discard_wait":
                s = g.turn
                a = py[s].choose_discard()
                b = na[s].choose_discard()
                cnt["discard"] += 1
                if g.players[s].melds:
                    melded_discards += 1
                if a != b:
                    bad["discard"] += 1
                    if bad["discard"] <= 3:
                        print(f"  DISCARD 分歧 seed={seed0+gi} seat={s} "
                              f"py={a} native={b} melds={g.players[s].melds} "
                              f"hand={g.players[s].hand}")
                g.action_discard(s, a)
            else:
                s = list(g.pending_actions.keys())[0]
                pend = g.pending_actions[s]
                if pend.get("gang"):
                    a = py[s].decide_gang(g.last_discard, "ming")
                    b = na[s].decide_gang(g.last_discard, "ming")
                    cnt["gang"] += 1
                    if a != b:
                        bad["gang"] += 1
                        if bad["gang"] <= 3:
                            print(f"  GANG 分歧 seed={seed0+gi} seat={s} "
                                  f"py={a} native={b}")
                    if a:
                        g.action_gang(s)
                        continue
                if pend.get("peng"):
                    a = py[s].decide_peng(g.last_discard)
                    b = na[s].decide_peng(g.last_discard)
                    cnt["peng"] += 1
                    if a != b:
                        bad["peng"] += 1
                        if bad["peng"] <= 3:
                            print(f"  PENG 分歧 seed={seed0+gi} seat={s} "
                                  f"py={a} native={b} "
                                  f"hand={g.players[s].hand}")
                    if a:
                        g.action_peng(s)
                        continue
                g.action_pass(s)
    return bad, cnt, melded_discards


def play(g, bots):
    guard = 0
    while g.phase != "game_over" and guard < 500:
        guard += 1
        if g.phase == "discard_wait":
            g.action_discard(g.turn, bots[g.turn].choose_discard())
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
    return g


def whole_game(kind: str, n_games: int, seed0: int):
    PyCls, NaCls = PAIRS[kind]
    bad = 0
    t_py = t_na = 0.0
    for gi in range(n_games):
        g1 = Game(seed=seed0 + gi, human_seat=-1)
        t0 = time.process_time()
        play(g1, {i: PyCls(g1, i) for i in range(4)})
        t_py += time.process_time() - t0
        g2 = Game(seed=seed0 + gi, human_seat=-1)
        t0 = time.process_time()
        play(g2, {i: NaCls(g2, i) for i in range(4)})
        t_na += time.process_time() - t0
        d1 = [p.score_delta for p in g1.players]
        d2 = [p.score_delta for p in g2.players]
        if g1.log != g2.log or d1 != d2 or g1.winner != g2.winner:
            bad += 1
            if bad <= 2:
                for i, (x, y) in enumerate(zip(g1.log, g2.log)):
                    if x != y:
                        print(f"  整局分歧 seed={seed0+gi} 第{i}步: "
                              f"py={x!r} native={y!r}")
                        break
                print(f"   得分 py={d1} native={d2}")
    return bad, t_py, t_na


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bots", default="v1,v10,v31")
    ap.add_argument("--games", type=int, default=30)
    ap.add_argument("--whole-games", type=int, default=10)
    ap.add_argument("--seed0", type=int, default=880000)
    args = ap.parse_args()

    ok = True
    for kind in args.bots.split(","):
        print(f"=== {kind}")
        bad, cnt, melded = per_decision(kind, args.games, args.seed0)
        print(f"  逐决策: discard {bad['discard']}/{cnt['discard']} 分歧"
              f"(其中副露手 {melded} 次), peng {bad['peng']}/{cnt['peng']}, "
              f"gang {bad['gang']}/{cnt['gang']}")
        wbad, t_py, t_na = whole_game(kind, args.whole_games,
                                      args.seed0 + 500000)
        print(f"  整局: {wbad}/{args.whole_games} 局日志或得分不一致")
        print(f"  整局 CPU: py {t_py:.2f}s vs native {t_na:.2f}s "
              f"-> {t_py/max(t_na,1e-9):.0f}x  "
              f"({args.whole_games/max(t_na,1e-9):.1f} 局/s/核)")
        if sum(bad.values()) or wbad:
            ok = False
    print("PARITY", "OK" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
