"""重测 v10_jax(JAX 版 v10 规则推演)在真正的 GPU 上的编译时间与吞吐。

背景: 之前给出的"编译 978s / 缓存后 >7min"是在 jaxlib 静默退回 CPU 的情况下测的
(见 backend/jax159/__init__.py 的根因说明), 那组数字不成立, 必须在 CudaDevice 上重测。
"""

import argparse
import copy
import random
import time

from backend.game.engine import Game
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
    ap.add_argument("--states", type=int, default=4)
    ap.add_argument("--worlds", type=int, default=32)
    ap.add_argument("--top-m", type=int, default=4)
    ap.add_argument("--chunk", type=int, default=512)
    args = ap.parse_args()

    import backend.jax159  # noqa: 触发 nvjitlink 预载
    import jax
    print("jax devices:", jax.devices(), flush=True)

    snaps = make_snaps(args.states)
    cand_lists = []
    for g, seat in snaps:
        opts = discard_options(list(g.players[seat].hand_counts))
        cand_lists.append([o["tile"] for o in opts][:args.top_m])
    n_games = sum(len(c) for c in cand_lists) * args.worlds
    print(f"快照 {len(snaps)}, 候选 {sum(len(c) for c in cand_lists)}, "
          f"世界 {args.worlds} -> {n_games} 局", flush=True)

    from backend.jax159.fast_inject import build_world_states
    from backend.jax159.v10_rollout import rollout_v10_jax
    t0 = time.time()
    sts, meta, _ = build_world_states(snaps, cand_lists, args.worlds, 7919)
    print(f"注入 {time.time()-t0:.1f}s", flush=True)

    t0 = time.time()
    r1 = rollout_v10_jax(sts, meta, 0.02, feat_chunk=args.chunk)
    t_first = time.time() - t0
    print(f"首次(含编译) {t_first:.1f}s", flush=True)
    t0 = time.time()
    r2 = rollout_v10_jax(sts, meta, 0.02, feat_chunk=args.chunk)
    t_cached = time.time() - t0
    print(f"缓存后 {t_cached:.1f}s -> {n_games/t_cached:.1f} 局/s (单卡)",
          flush=True)
    print("样例回报:", {k: {t: round(v, 3) for t, v in d.items()}
                        for k, d in list(r2.items())[:2]})


if __name__ == "__main__":
    main()
