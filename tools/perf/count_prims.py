"""统计一局 v10x4 推演里 shanten/is_win 的调用次数与去重数, 用于估算 C 版收益。"""
import copy
import random
import time

from backend.game.engine import Game
from backend.rl.world_grpo import sample_world
from backend.rules import win as W
from backend.rules.ting import discard_options

_orig_shanten = W.shanten
_orig_is_win = W.is_win

STAT = {"sh": 0, "win": 0, "sh_set": set(), "win_set": set(),
        "dfs": 0, "dfs_set": set()}


def shanten_counted(tiles_counts):
    STAT["sh"] += 1
    STAT["sh_set"].add(tuple(tiles_counts))
    return _orig_shanten(tiles_counts)


def is_win_counted(tiles_counts):
    STAT["win"] += 1
    STAT["win_set"].add(tuple(tiles_counts))
    return _orig_is_win(tiles_counts)


def install():
    W.shanten = shanten_counted
    W.is_win = is_win_counted
    import backend.rules.ting as T
    T.shanten = shanten_counted
    T.is_win = is_win_counted
    import backend.ai.bot_v10 as B10
    # bot_v10 imports shanten / shanten_cached directly
    B10.shanten = shanten_counted
    import backend.ai.bot_v1 as B1
    if hasattr(B1, "shanten"):
        B1.shanten = shanten_counted


def make_snap(seed):
    from backend.ai.bot_v10 import Bot as BotV10
    g = Game(seed=seed, human_seat=-1)
    bots = {i: BotV10(g, i) for i in range(4)}
    rng = random.Random(seed ^ 0xC0FFEE)
    target = rng.randint(4, 12)
    tc, guard = 0, 0
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


def main():
    from backend.ai.bot_v10 import Bot as BotCls
    snap, seat = make_snap(760000)
    assert snap is not None
    install()
    rng = random.Random(999)
    hands, wall = sample_world(snap, rng, hero_seat=seat)
    tile = discard_options(list(snap.players[seat].hand_counts))[0]["tile"]

    t0 = time.process_time()
    g = copy.deepcopy(snap)
    for s2, h in hands.items():
        g.players[s2].hand = list(h)
    g.wall = list(wall)
    bots = {i: BotCls(g, i) for i in range(4)}
    g.action_discard(seat, tile)
    guard, steps = 0, 0
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
    dt = time.process_time() - t0
    print(f"1 game(4x v10): cpu={dt:.2f}s discard_steps={steps}")
    print(f"  shanten() calls={STAT['sh']}  distinct={len(STAT['sh_set'])}")
    print(f"  is_win()  calls={STAT['win']} distinct={len(STAT['win_set'])}")
    print(f"  _dfs_cached {W._dfs_cached.cache_info()}")
    print(f"  _all_melds  {W._all_melds_cached.cache_info()}")
    print(f"  per decision: shanten={STAT['sh']/max(steps,1):.0f} "
          f"is_win={STAT['win']/max(steps,1):.0f}")


if __name__ == "__main__":
    main()
