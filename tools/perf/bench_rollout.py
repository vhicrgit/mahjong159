"""基准: 训练里真实的世界推演路径 (_rollout_worker_rules) 在 v31 / v31n 下的吞吐。

口径与 grpo_train 完全一致: 同一批快照 × 同一批候选 × n 个共享世界,
每个 (候选) 一个 task。用 process_time 记 CPU 时间(机器负载高, wall 不可比)。

用法:
  python -m tools.perf.bench_rollout --bot v31n --worlds 128 --states 4
  python -m tools.perf.bench_rollout --bot v31  --worlds 8   --states 1
"""

import argparse
import copy
import random
import time

from backend.game.engine import Game
from backend.rl.grpo_train import _rollout_worker_rules
from backend.rules.ting import discard_options


def make_snaps(n, seed0=770000):
    from backend.ai.bot_native import NativeV10
    out = []
    gi = 0
    while len(out) < n and gi < n * 20:
        seed = seed0 + gi
        gi += 1
        g = Game(seed=seed, human_seat=-1)
        bots = {i: NativeV10(g, i) for i in range(4)}
        rng = random.Random(seed ^ 0xC0FFEE)
        target = rng.randint(4, 12)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bot", default="v31n")
    ap.add_argument("--worlds", type=int, default=128)
    ap.add_argument("--states", type=int, default=4)
    ap.add_argument("--top-m", type=int, default=4)
    ap.add_argument("--step-penalty", type=float, default=0.02)
    args = ap.parse_args()

    snaps = make_snaps(args.states)
    print(f"快照 {len(snaps)} 个")
    tasks = []
    for si, (g, seat) in enumerate(snaps):
        opts = discard_options(list(g.players[seat].hand_counts))
        tiles = [o["tile"] for o in opts][:args.top_m]
        for t in tiles:
            tasks.append((g, seat, t, 7919 + si, args.worlds,
                          args.step_penalty, args.bot))
    n_games = len(tasks) * args.worlds
    print(f"任务 {len(tasks)} 个候选 × {args.worlds} 世界 = {n_games} 局")

    t0 = time.process_time()
    tw = time.time()
    rets = [_rollout_worker_rules(t) for t in tasks]
    cpu = time.process_time() - t0
    wall = time.time() - tw
    print(f"bot={args.bot} cpu={cpu:.2f}s wall={wall:.2f}s "
          f"-> {n_games/cpu:.1f} 局/s/核 ({cpu/n_games*1000:.2f} ms/局)")
    print(f"  回报样例: {[round(r,3) for r in rets[:6]]}")

    # 换算成一次训练迭代(32 快照 × 4 候选 × worlds)
    per_it = 32 * 4 * args.worlds
    for procs in (8, 10):
        print(f"  推算: 32快照×4候选×{args.worlds}世界 = {per_it} 局, "
              f"{procs} 进程满载 ≈ {per_it/(n_games/cpu)/procs:.0f}s/迭代")


if __name__ == "__main__":
    main()
