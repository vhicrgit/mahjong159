"""特征对拍: jax159.encode_obs vs backend.rl.features_v2.encode_state

从 v10 自对弈收集真实局面(含副露), 双端编码逐位比较。
运行: .venv-jax/bin/python -m backend.jax159.test_features --states 30
"""

import argparse
import sys

import numpy as np

sys.path.insert(0, ".")

from backend.game.engine import Game
from backend.ai.bot_v1 import Bot as BotV1
from backend.ai.bot_v10 import Bot as BotV10
from backend.rl.features_v2 import encode_state as py_encode, FEAT_DIM

import jax.numpy as jnp
from backend.jax159.convert import state_from_game, batch_states
from backend.jax159.features import encode_obs


def collect_state(seed: int, target_turn: int = 6):
    g = Game(seed=seed, human_seat=-1)
    bots = {i: BotV10(g, i) for i in range(4)}
    turns = 0
    guard = 0
    while g.phase != "game_over" and guard < 500:
        guard += 1
        if g.phase == "discard_wait":
            turns += 1
            if turns == target_turn:
                return g, g.turn
            g.action_discard(g.turn, bots[g.turn].choose_discard())
        elif g.phase == "react_wait":
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
    ap.add_argument("--states", type=int, default=30)
    ap.add_argument("--seed0", type=int, default=910000)
    args = ap.parse_args()

    max_diff = 0.0
    n = 0
    for i in range(args.states):
        g, seat = collect_state(args.seed0 + i, 6)
        if g is None:
            continue
        py_f = py_encode(g, seat)
        js = batch_states([state_from_game(g, hero_seat=seat)])
        js_f = np.asarray(
            encode_obs(js, jnp.array([seat], dtype=jnp.int8))[0])
        d = np.abs(py_f - js_f).max()
        max_diff = max(max_diff, d)
        n += 1
        if d > 1e-5:
            idx = np.argmax(np.abs(py_f - js_f))
            print(f"seed{args.seed0+i} seat{seat} 不一致@{idx}: "
                  f"py={py_f[idx]} js={js_f[idx]}")
    print(f"{n} 局面, 最大绝对差 {max_diff:.2e}")


if __name__ == "__main__":
    main()
