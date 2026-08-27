"""双引擎对拍: Python engine.py vs JAX env159

同一副牌墙 + 同一动作序列(共享随机索引选择合法动作), 逐步比对:
手牌计数/弃牌计数/副露/阶段/回合/胜者/分数/159张数。
任何一步不一致立即打印现场并报错退出。

运行(需在 .venv-jax 中, 纯CPU即可):
  LD_LIBRARY_PATH=$(cat .venv-jax/nvlibs.txt | sed 's|^|'$PWD'/|') \
    .venv-jax/bin/python -m backend.jax159.test_parity --games 200
"""

import argparse
import os
import random
import sys

import numpy as np

sys.path.insert(0, ".")

# Python 引擎 (纯 stdlib, 可在 venv-jax 中导入)
from backend.game.engine import Game
from backend.rules.tiles import counts_from_tiles

import jax
import jax.numpy as jnp
from backend.jax159 import env159


def py_legal(g: Game) -> list[int]:
    """与 env159.legal_actions 同编码的 Python 版合法动作列表。"""
    acts = []
    if g.phase == "discard_wait":
        p = g.players[g.turn]
        hc = p.hand_counts
        acts += [t for t in range(28) if hc[t] > 0]
        acts += [28 + t for t in range(27) if hc[t] == 4]
        for m_ in p.melds:
            if m_["type"] == "peng" and hc[m_["tile"]] >= 1:
                acts.append(55 + m_["tile"])
    elif g.phase == "react_wait":
        acts.append(0)
        seat = list(g.pending_actions.keys())[0]
        if g.pending_actions[seat].get("peng"):
            acts.append(1)
        if g.pending_actions[seat].get("gang"):
            acts.append(2)
    return acts


def py_step(g: Game, a: int):
    if g.phase == "discard_wait":
        if a < 28:
            g.action_discard(g.turn, a)
        elif a < 55:
            g.action_gang(g.turn, a - 28)
        else:
            g.action_gang(g.turn, a - 55)
    elif g.phase == "react_wait":
        seat = list(g.pending_actions.keys())[0]
        if a == 1:
            g.action_peng(seat)
        elif a == 2:
            g.action_gang(seat)
        else:
            g.action_pass(seat)


def compare(py: Game, js, ctx: str) -> list[str]:
    errs = []
    js = jax.device_get(js)
    for s in range(4):
        py_h = counts_from_tiles(py.players[s].hand)
        if not np.array_equal(np.array(js.hands[s]), np.array(py_h)):
            errs.append(f"{ctx} 座位{s}手牌不一致")
        py_d = counts_from_tiles(py.players[s].discards)
        if not np.array_equal(np.array(js.discards[s]), np.array(py_d)):
            errs.append(f"{ctx} 座位{s}弃牌不一致")
        py_nm = len(py.players[s].melds)
        if int(js.n_melds[s]) != py_nm:
            errs.append(f"{ctx} 座位{s}副露数不一致 {js.n_melds[s]} vs {py_nm}")
    phase_map = {"discard_wait": 0, "react_wait": 1, "game_over": 2}
    if int(js.phase) != phase_map[py.phase]:
        errs.append(f"{ctx} phase 不一致 {int(js.phase)} vs {py.phase}")
    if int(js.phase) != 2 and int(js.turn) != py.turn:
        errs.append(f"{ctx} turn 不一致 {int(js.turn)} vs {py.turn}")
    py_winner = -1 if py.winner is None else py.winner
    if int(js.winner) != py_winner:
        errs.append(f"{ctx} winner 不一致 {int(js.winner)} vs {py_winner}")
    if py.phase == "game_over" and py.winner is not None:
        if int(js.n_159) != py.n_159:
            errs.append(f"{ctx} n159 不一致 {int(js.n_159)} vs {py.n_159}")
        for s in range(4):
            if int(js.scores[s]) != py.players[s].score_delta:
                errs.append(f"{ctx} 座位{s}分数不一致 "
                            f"{int(js.scores[s])} vs {py.players[s].score_delta}")
    return errs


def play_paired(seed: int, rng: random.Random) -> list[str]:
    g = Game(seed=seed, human_seat=-1)
    # 复现引擎的洗牌得到原始牌墙 (引擎 random.Random(seed).shuffle 确定性)
    from backend.rules.tiles import build_wall
    wr = random.Random(seed)
    full_wall = build_wall()
    wr.shuffle(full_wall)
    env159.load_win_tables()
    js = env159.reset_from_wall(jnp.array(full_wall, dtype=jnp.int8))
    errs = compare(g, js, f"seed{seed} init")
    if errs:
        return errs
    step_i = 0
    while g.phase != "game_over" and step_i < 500:
        step_i += 1
        acts_py = py_legal(g)
        legal_js = np.array(env159.legal_jit(js))
        acts_js = list(np.nonzero(legal_js)[0])
        if acts_py != acts_js:
            return [f"seed{seed} step{step_i} 合法动作不一致 "
                    f"py={acts_py} js={acts_js} phase={g.phase}"]
        a = acts_py[rng.randrange(len(acts_py))]
        py_step(g, a)
        js = env159.step_jit(js, jnp.int32(a))
        errs = compare(g, js, f"seed{seed} step{step_i} act{a}")
        if errs:
            return errs
    return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--seed0", type=int, default=880000)
    args = ap.parse_args()
    env159.load_win_tables()
    rng = random.Random(0)
    total_err = 0
    for i in range(args.games):
        errs = play_paired(args.seed0 + i, rng)
        if errs:
            total_err += len(errs)
            for e in errs:
                print("MISMATCH:", e)
            if total_err > 20:
                break
        if (i + 1) % 50 == 0:
            print(f"{i+1}/{args.games} 局对拍通过", flush=True)
    print(f"完成: {args.games} 局, 不一致 {total_err}")


if __name__ == "__main__":
    main()
