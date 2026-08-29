"""Profile a v31 (=BotV10 in registry) world rollout: where does the time go?

用 process_time 度量 CPU 时间(机器负载高, wall time 不可比)。
"""
import argparse
import copy
import cProfile
import io
import pstats
import random
import time

from backend.game.engine import Game
from backend.rl.gen_offline import _shaped_scores
from backend.rl.world_grpo import sample_world
from backend.rules.ting import discard_options


def make_snap(seed: int):
    from backend.ai.bot_v10 import Bot as BotV10
    g = Game(seed=seed, human_seat=-1)
    bots = {i: BotV10(g, i) for i in range(4)}
    rng = random.Random(seed ^ 0xC0FFEE)
    target = rng.randint(4, 12)
    tc = 0
    guard = 0
    while g.phase != "game_over" and guard < 500:
        guard += 1
        if g.phase == "discard_wait":
            tc += 1
            if tc == target and len(discard_options(
                    list(g.players[g.turn].hand_counts))) >= 3:
                g.log = []
                return copy.deepcopy(g), g.turn
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
    return None, None


def one_rollout(snap, seat, tile, hands, wall, bot_cls, step_penalty=0.02):
    g = copy.deepcopy(snap)
    for s2, h in hands.items():
        g.players[s2].hand = list(h)
    g.wall = list(wall)
    bots = {i: bot_cls(g, i) for i in range(4)}
    g.action_discard(seat, tile)
    guard = 0
    steps = 0
    while g.phase != "game_over" and guard < 500:
        guard += 1
        if g.phase == "discard_wait":
            steps += 1
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
    return float(_shaped_scores(g, step_penalty)[seat]), steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bot", default="v10", choices=["v1", "v10", "v31"])
    ap.add_argument("--worlds", type=int, default=8)
    ap.add_argument("--states", type=int, default=2)
    ap.add_argument("--profile", action="store_true")
    args = ap.parse_args()

    if args.bot == "v1":
        from backend.ai.bot_v1 import Bot as BotCls
    elif args.bot == "v10":
        from backend.ai.bot_v10 import Bot as BotCls
    else:
        from backend.ai.bot_v31 import Bot as BotCls

    from backend.rules import win as W
    from backend.ai import bot_v10 as B10

    snaps = []
    for i in range(args.states):
        s, seat = make_snap(760000 + i)
        if s is not None:
            snaps.append((s, seat))
    print(f"snaps={len(snaps)}")

    def workload():
        tot_steps = 0
        n_games = 0
        for snap, seat in snaps:
            opts = discard_options(list(snap.players[seat].hand_counts))
            tiles = [o["tile"] for o in opts][:4]
            rng = random.Random(12345)
            worlds = [sample_world(snap, rng, hero_seat=seat)
                      for _ in range(args.worlds)]
            for t in tiles:
                for hands, wall in worlds:
                    _, st = one_rollout(snap, seat, t, hands, wall, BotCls)
                    tot_steps += st
                    n_games += 1
        return n_games, tot_steps

    # warm caches lightly? no - measure cold-ish then warm
    t0 = time.process_time()
    if args.profile:
        pr = cProfile.Profile()
        pr.enable()
        n_games, tot_steps = workload()
        pr.disable()
    else:
        n_games, tot_steps = workload()
    dt = time.process_time() - t0

    print(f"bot={args.bot} games={n_games} discard_steps={tot_steps} "
          f"cpu={dt:.2f}s  -> {n_games/dt:.2f} games/s/core, "
          f"{tot_steps/dt:.0f} decisions/s/core")
    print("cache:", "shanten_cached", W.shanten_cached.cache_info())
    print("cache:", "_dfs_cached", W._dfs_cached.cache_info())
    print("cache:", "_all_melds_cached", W._all_melds_cached.cache_info())
    print("cache:", "_ukeire", B10._ukeire.cache_info())
    print("cache:", "_second_step", B10._second_step_value.cache_info())

    if args.profile:
        s = io.StringIO()
        pstats.Stats(pr, stream=s).sort_stats("tottime").print_stats(22)
        print(s.getvalue())


if __name__ == "__main__":
    main()
