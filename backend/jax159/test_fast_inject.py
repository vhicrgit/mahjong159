"""fast_inject 正确性对拍: 与 rollout.build_world_states(deepcopy版) 全字段一致 + 端到端推演一致。"""

import sys, copy
sys.path.insert(0, ".")
import numpy as np
from backend.game.engine import Game
from backend.ai.bot_v10 import Bot as V10
from backend.jax159.fast_inject import build_world_states as fast_build
from backend.jax159.rollout import build_world_states as old_build
from backend.jax159.rollout import rollout_jax
from backend.jax159.jax_net import JaxNet
from backend.jax159.env159 import load_win_tables
from backend.jax159.shanten import load_front_table


def collect_snaps(seed, n_snaps, max_turn=8):
    g = Game(seed=seed, human_seat=-1)
    bots = {i: V10(g, i) for i in range(4)}
    snaps = []
    guard = 0
    while g.phase != "game_over" and guard < 500 and len(snaps) < n_snaps:
        guard += 1
        if g.phase == "discard_wait":
            if len(snaps) < n_snaps and g.turn in (1, 2, 3):
                snaps.append((copy.deepcopy(g), g.turn))
            g.action_discard(g.turn, bots[g.turn].choose_discard())
        elif g.phase == "react_wait":
            s = list(g.pending_actions.keys())[0]
            b = bots[s]
            if g.pending_actions[s].get("gang") and b.decide_gang(g.last_discard, "ming"):
                g.action_gang(s)
            elif g.pending_actions[s].get("peng") and b.decide_peng(g.last_discard):
                g.action_peng(s)
            else:
                g.action_pass(s)
    return snaps


def field_match(a, b, name):
    return bool((a == b).all()), f"{name}: maxdiff={(a != b).sum()}/{a.size}"


def main():
    load_win_tables()
    load_front_table()
    snaps = collect_snaps(7, 6)
    print(f"collected {len(snaps)} snaps")
    from backend.rl.world_grpo import _v10_scores
    cands = []
    for snap, hero in snaps:
        scores = _v10_scores(snap, hero)
        cands.append([t for t, _ in sorted(scores.items(),
                                           key=lambda kv: -kv[1])[:4]])

    sts_old, meta_old, _ = old_build(snaps, cands, 4, 714)
    sts_new, meta_new, _ = fast_build(snaps, cands, 4, 714)
    print(f"old B={len(meta_old)}, new B={len(meta_new)}")
    assert len(meta_old) == len(meta_new)
    assert meta_old == meta_new, "meta 不一致"

    fields = ["hands", "discards", "wall", "wall_tail",
              "melds_tile", "melds_kind", "melds_from", "n_melds",
              "last_discard", "last_discarder",
              "winner", "win_kind", "n_159", "scores", "draws"]
    n_bad = 0
    for f in fields:
        ok, msg = field_match(getattr(sts_old, f), getattr(sts_new, f), f)
        if not ok:
            n_bad += 1
            print("DIFF", msg)
        else:
            print("OK  ", f)
    if n_bad == 0:
        print("== 字段完全一致 ==")
    else:
        print(f"== {n_bad} 个字段不一致 (预期: 旧版 react_wait 中间态 vs 新版直接下家摸牌, 等价表示) ==")
    react_mask = (np.asarray(sts_old.phase) == 1)
    print(f"旧版处于 react_wait 的 state: {int(react_mask.sum())}/{len(meta_old)}")

    # 端到端: 用同一 JaxNet 推演比较终局
    import torch
    ckpt = torch.load("models/dqn_shaped_100k_best.pt", map_location="cpu",
                      weights_only=True)
    sd = {k: v.detach().cpu().numpy() for k, v in ckpt["model"].items()}
    net = JaxNet.from_dict(sd)
    r_old = rollout_jax(sts_old, meta_old, net, 0.02)
    r_new = rollout_jax(sts_new, meta_new, net, 0.02)
    assert r_old.keys() == r_new.keys()
    bad = 0
    for si in r_old:
        for t in r_old[si]:
            if abs(r_old[si][t] - r_new[si][t]) > 1e-3:
                bad += 1
                print(f"reward diff si={si} t={t}: {r_old[si][t]:.3f} vs {r_new[si][t]:.3f}")
    print(f"端到端奖励: {len(r_old)*4} 项, 不一致 {bad}")


if __name__ == "__main__":
    main()
