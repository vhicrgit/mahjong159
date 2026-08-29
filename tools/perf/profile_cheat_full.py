"""拆解 cheat_full 一次决策的成本, 定位该把哪部分搬进 C。

cheat_full = BotCheat(wall_lookahead=-1, see_opponents=True, rollout=True)
  choose_discard
    -> _ranked_oracle_discards            (含 beam search)
    -> 对 root_width 个候选各 _root_rollout_score
         -> deepcopy(game) + _rollout_to_end(depth=search_depth-1)
              -> 每个 hero 回合 _choose_rollout_discard(depth)
                   -> _ranked_oracle_discards + search_width 个子 rollout
"""

import argparse
import copy
import random
import time

from backend.game.engine import Game
from backend.rules.ting import discard_options

CHEAT_FULL = dict(wall_lookahead=-1, see_opponents=True, rollout=True)


def make_snap(seed=770000):
    from backend.ai.bot_native import NativeV10
    g = Game(seed=seed, human_seat=-1)
    bots = {i: NativeV10(g, i) for i in range(4)}
    rng = random.Random(seed ^ 0xC0FFEE)
    target = rng.randint(4, 10)
    tc, guard = 0, 0
    while g.phase != "game_over" and guard < 500:
        guard += 1
        if g.phase == "discard_wait":
            tc += 1
            seat = g.turn
            if tc == target and len(discard_options(
                    list(g.players[seat].hand_counts))) >= 3:
                g.log = []
                return copy.deepcopy(g), seat
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
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full-decision", action="store_true",
                    help="真跑一次完整 choose_discard(可能几分钟)")
    args = ap.parse_args()

    from backend.ai.bot_cheat import Bot as BotCheat
    from backend.ai.bot_oracle import search_first_discard_detail

    snap, seat = make_snap()
    assert snap is not None
    bot = BotCheat(snap, seat, **CHEAT_FULL)
    counts14 = list(snap.players[seat].hand_counts)
    opts = discard_options(counts14)
    future = bot._future_draws(snap)
    print(f"座位 {seat}, 候选 {len(opts)} 张, future_draws {len(future)} 张, "
          f"牌墙 {len(snap.wall)}")
    print(f"参数: depth={bot.search_depth} width={bot.search_width} "
          f"root_width={bot.root_width} beam={bot.beam}")

    # 1) 一次 beam search
    t0 = time.process_time()
    search_first_discard_detail(counts14, future, bot.beam)
    t_beam = time.process_time() - t0
    print(f"\n[1] search_first_discard_detail 1 次: {t_beam*1000:.1f} ms")

    # 2) 一次 _ranked_oracle_discards
    t0 = time.process_time()
    ranked = bot._ranked_oracle_discards()
    t_rank = time.process_time() - t0
    print(f"[2] _ranked_oracle_discards 1 次: {t_rank*1000:.1f} ms")

    # 3) 一次 deepcopy(game)
    t0 = time.process_time()
    for _ in range(20):
        copy.deepcopy(snap)
    t_dc = (time.process_time() - t0) / 20
    print(f"[3] deepcopy(Game) 1 次: {t_dc*1000:.2f} ms")

    # 4) 一次 depth=0 的整局 rollout(hero 每回合 1 次 beam search)
    g0 = copy.deepcopy(snap)
    g0.action_discard(seat, ranked[0])
    t0 = time.process_time()
    bot._rollout_to_end(g0, 0)
    t_r0 = time.process_time() - t0
    print(f"[4] _rollout_to_end(depth=0) 1 次: {t_r0:.2f} s")

    # 5) 一次 depth=1 的整局 rollout(= _root_rollout_score 的主体)
    t0 = time.process_time()
    v = bot._root_rollout_score(ranked[0])
    t_r1 = time.process_time() - t0
    print(f"[5] _root_rollout_score(depth=1) 1 次: {t_r1:.2f} s (v={v:.1f})")

    n_root = min(len(ranked), bot.root_width)
    print(f"\n推算 choose_discard = {n_root} 个 root 候选 × [5] "
          f"≈ {n_root*t_r1:.0f} s/决策")
    print(f"推算 一局(约 29 次 hero 决策, 但 cheat_full 只在 hero 回合用) "
          f"≈ {n_root*t_r1*8:.0f} s/局 (hero 约 8 次出牌)")
    print(f"推算 一次迭代(32×4×32 世界 = 4096 局) "
          f"≈ {n_root*t_r1*8*4096/3600:.0f} 核·小时")

    if args.full_decision:
        t0 = time.process_time()
        t = bot.choose_discard()
        print(f"\n实测 choose_discard 完整 1 次: "
              f"{time.process_time()-t0:.1f} s -> 打 {t}")


if __name__ == "__main__":
    main()
