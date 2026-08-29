"""校验 grpo_train 里另外两处 v10 调用能否安全换成原生实现:

1) _collect_worker 用 BotV10 自对弈采样快照 -> NativeV10 是否给出同一局进程
2) v10_pick = argmax(_v10_scores(g, seat)) -> NativeV10.choose_discard() 是否同一张

两者都必须逐位一致, 否则训练分布会变。
"""

import argparse
import copy
import random

from ..ai.bot_native import NativeV10
from ..ai.bot_v10 import Bot as PyV10
from ..game.engine import Game
from ..rules.ting import discard_options
from .world_grpo import _v10_scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=6)
    ap.add_argument("--seed0", type=int, default=930000)
    args = ap.parse_args()

    bad_pick = 0
    n_pick = 0
    bad_log = 0
    for gi in range(args.games):
        seed = args.seed0 + gi
        # (1) 整局: Python 驱动, 每个决策点顺便比 v10_pick
        g = Game(seed=seed, human_seat=-1)
        bots = {i: PyV10(g, i) for i in range(4)}
        nat = {i: NativeV10(g, i) for i in range(4)}
        guard = 0
        while g.phase != "game_over" and guard < 500:
            guard += 1
            if g.phase == "discard_wait":
                seat = g.turn
                if len(discard_options(list(g.players[seat].hand_counts))) >= 3:
                    pick_py = max(_v10_scores(g, seat).items(),
                                  key=lambda kv: kv[1])[0]
                    pick_na = nat[seat].choose_discard()
                    n_pick += 1
                    if pick_py != pick_na:
                        bad_pick += 1
                        if bad_pick <= 3:
                            print(f"  v10_pick 分歧 seed={seed} seat={seat} "
                                  f"py={pick_py} native={pick_na}")
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

        # (2) _collect_worker 等价性: 两种 bot 各自独立跑同一 seed 的整局
        logs = []
        snaps_sig = []
        for cls in (PyV10, NativeV10):
            g2 = Game(seed=seed, human_seat=-1)
            bb = {i: cls(g2, i) for i in range(4)}
            rng = random.Random(seed ^ 0xC0FFEE)
            take = set(rng.sample(range(2, 14), 4))
            tc, guard = 0, 0
            sig = []
            while g2.phase != "game_over" and guard < 500:
                guard += 1
                if g2.phase == "discard_wait":
                    tc += 1
                    seat = g2.turn
                    if tc in take and len(discard_options(
                            list(g2.players[seat].hand_counts))) >= 3:
                        sig.append((seat, tuple(g2.players[seat].hand),
                                    len(g2.wall)))
                    g2.action_discard(seat, bb[seat].choose_discard())
                else:
                    s = list(g2.pending_actions.keys())[0]
                    b = bb[s]
                    if g2.pending_actions[s].get("gang") and \
                            b.decide_gang(g2.last_discard, "ming"):
                        g2.action_gang(s)
                    elif g2.pending_actions[s].get("peng") and \
                            b.decide_peng(g2.last_discard):
                        g2.action_peng(s)
                    else:
                        g2.action_pass(s)
            logs.append(g2.log)
            snaps_sig.append(sig)
        if logs[0] != logs[1] or snaps_sig[0] != snaps_sig[1]:
            bad_log += 1
            print(f"  采样分歧 seed={seed}")

    print(f"v10_pick: {bad_pick}/{n_pick} 分歧")
    print(f"_collect_worker 等价: {bad_log}/{args.games} 局分歧")
    ok = bad_pick == 0 and bad_log == 0
    print("PARITY", "OK" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
