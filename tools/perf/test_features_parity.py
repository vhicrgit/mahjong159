"""features_v2.encode_state 换 C 向听后的逐位对拍 + 提速测量。

encode_state 原来对每个候选弃牌重算一遍 Python DFS 向听, profile 显示占
自对弈采集 86% 的时间。换成 mj_discard_shanten 一次拿全 + 只对听牌候选补
一次 waits_ukeire。特征值必须逐位不变, 否则已有的预训练权重全部作废。

用法: python -m tools.perf.test_features_parity [--games 120]
"""

import argparse
import sys
import time

import numpy as np

from backend.ai.bot_native import NativeV31
from backend.game.engine import Game
from backend.rl import features_v2 as F


def states(n_games, seed0, bloody=False):
    """跑真实对局, 产出 (game, seat) 决策点。"""
    out = []
    for i in range(n_games):
        g = Game(seed=seed0 + i, human_seat=-1, bloody=bloody)
        bots = {s: NativeV31(g, s) for s in range(4)}
        guard = 0
        while g.phase != "game_over" and guard < 800:
            guard += 1
            if g.phase == "discard_wait":
                s = g.turn
                out.append(F.encode_state(g, s, _derive_fn=F._derive_py))
                out.append(F.encode_state(g, s, _derive_fn=F._derive_c))
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
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=120)
    ap.add_argument("--seed0", type=int, default=22000000)
    args = ap.parse_args()

    print("== 逐位对拍 (首胡 + 血战两种规则) ==")
    bad = 0
    tot = 0
    for bloody in (False, True):
        v = states(args.games // 2, args.seed0 + (0 if not bloody else 500000),
                   bloody)
        for k in range(0, len(v), 2):
            tot += 1
            if not np.array_equal(v[k], v[k + 1]):
                bad += 1
                if bad <= 3:
                    d = np.nonzero(v[k] != v[k + 1])[0]
                    print(f"  不一致 维度 {d[:8]}  py={v[k][d[:8]]} "
                          f"c={v[k + 1][d[:8]]}")
    print(f"  {tot} 个决策点: {'全部逐位一致' if bad == 0 else f'{bad} 处不一致'}")

    print("\n== 提速 ==")
    g = Game(seed=args.seed0 + 900000, human_seat=-1)
    bots = {s: NativeV31(g, s) for s in range(4)}
    for _ in range(12):                       # 推进到中盘再测
        if g.phase == "discard_wait":
            g.action_discard(g.turn, bots[g.turn].choose_discard())
        elif g.phase == "react_wait":
            g.action_pass(list(g.pending_actions.keys())[0])
    for name, fn in (("Python DFS", F._derive_py), ("C LUT", F._derive_c)):
        n = 300 if fn is F._derive_py else 3000
        t0 = time.perf_counter()
        for _ in range(n):
            F.encode_state(g, g.turn, _derive_fn=fn)
        dt = (time.perf_counter() - t0) / n
        print(f"  encode_state({name:10s}) {dt * 1e6:8.1f} us/次")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
