"""pmap 多卡 rollout 与单卡 rollout_jax 的结果一致性对拍。

用法: CUDA_VISIBLE_DEVICES=0,3,5 python -m backend.jax159.test_pmap_rollout
"""
import random
import numpy as np

from backend.game.engine import Game
from .convert import state_from_game, batch_states
from .jax_net import JaxNet
from .rollout import rollout_jax
from .pmap_rollout import rollout_jax_pmap


def _rand_games(n, seed):
    """构造 n 个随机中途局面快照"""
    rng = random.Random(seed)
    snaps = []
    for i in range(n):
        g = Game(seed=seed * 1000 + i, human_seat=-1)
        for _ in range(rng.randint(5, 55)):
            if g.phase == "discard_wait":
                hc = g.players[g.turn].hand_counts
                g.action_discard(g.turn,
                                 rng.choice([t for t in range(28) if hc[t] > 0]))
            elif g.phase == "react_wait":
                g.action_pass(list(g.pending_actions.keys())[0])
            else:
                break
        if g.phase == "discard_wait":
            snaps.append((g, g.turn))
    return snaps


def main():
    n_snap = 16
    n_worlds = 8
    top_m = 4
    snaps = _rand_games(n_snap, seed=7)
    print(f"snaps={len(snaps)}")
    # 每局面候选: 手里有的牌取 top_m
    cand_lists = []
    for g, hero in snaps:
        hc = g.players[hero].hand_counts
        tiles = [t for t in range(28) if hc[t] > 0][:top_m]
        cand_lists.append(tiles)

    net = JaxNet("models/dqn_shaped_100k_best_jax.npz")

    # 构建世界状态(单卡版与 pmap 版用同一份注入)
    from .rollout import build_world_states
    sts, meta, _ = build_world_states(snaps, cand_lists, n_worlds,
                                      world_seed=99)
    print(f"B={sts.hands.shape[0]}")

    r1 = rollout_jax(sts, meta, net, step_penalty=0.02)
    print("单卡 rollout 完成")
    r2 = rollout_jax_pmap(sts, meta, net, step_penalty=0.02)
    print("pmap rollout 完成")

    # 对比
    bad = 0
    for si in r1:
        for tile in r1[si]:
            a, b = r1[si][tile], r2[si][tile]
            if abs(a - b) > 1e-4:
                bad += 1
                if bad <= 8:
                    print(f"不一致 si={si} tile={tile}: 单卡={a:.4f} pmap={b:.4f}")
    print(f"候选总数={sum(len(d) for d in r1.values())}, 不一致={bad}")
    print("PASS" if bad == 0 else "FAIL")


if __name__ == "__main__":
    main()
