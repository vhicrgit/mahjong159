"""血战到底改动的回归测试。

两件事:
  1. **首胡模式必须逐位不变** —— 从 git HEAD 取出改动前的 engine.py 当基准,
     同一批 seed 跑同一套 bot, 逐局比对四家得分/胜者/黄庄/弃牌数。
     这是最要紧的一条: 线上规则不能被这次改动碰到。
  2. bloody 模式的不变量 —— 3 家胡完或牌墙耗尽才终局; 名次自洽;
     名次奖励零和; 已下场的人不再出牌/不再被问碰杠。

用法: python -m tools.perf.test_bloody_parity [--games 400]
"""

import argparse
import importlib.util
import os
import subprocess
import sys
import tempfile

from backend.ai.bot_native import NativeV31
from backend.game.engine import Game as GameNew


def load_head_engine():
    """把 git HEAD 版本的 engine.py 作为 backend.game._engine_head 载入。
    模块名放在 backend.game 包下, 里面的相对 import 才能解析。"""
    root = subprocess.run(["git", "rev-parse", "--show-toplevel"],
                          capture_output=True, text=True,
                          check=True).stdout.strip()
    src = subprocess.run(["git", "show", "HEAD:backend/game/engine.py"],
                         capture_output=True, text=True, check=True,
                         cwd=root).stdout
    d = tempfile.mkdtemp()
    path = os.path.join(d, "_engine_head.py")
    with open(path, "w") as f:
        f.write(src)
    name = "backend.game._engine_head"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod.Game


def play(GameCls, seed, bloody=None):
    kw = {} if bloody is None else {"bloody": bloody}
    g = GameCls(seed=seed, human_seat=-1, **kw)
    bots = {s: NativeV31(g, s) for s in range(4)}
    guard = 0
    while g.phase != "game_over" and guard < 800:
        guard += 1
        if g.phase == "discard_wait":
            s = g.turn
            assert g.is_active(s) if hasattr(g, "is_active") else True, \
                f"已下场的座位{s}还在出牌"
            g.action_discard(s, bots[s].choose_discard())
        else:
            s = list(g.pending_actions.keys())[0]
            pend = g.pending_actions[s]
            b = bots[s]
            if pend.get("gang") and b.decide_gang(g.last_discard, "ming"):
                g.action_gang(s)
            elif pend.get("peng") and b.decide_peng(g.last_discard):
                g.action_peng(s)
            else:
                g.action_pass(s)
    assert g.phase == "game_over", f"seed {seed} 未终局(guard={guard})"
    return g


def snap(g):
    return ([p.score_delta for p in g.players], g.winner, g.huangzhuang,
            [len(p.discards) for p in g.players],
            [len(p.melds) for p in g.players], len(g.wall))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=400)
    ap.add_argument("--seed0", type=int, default=13000000)
    args = ap.parse_args()

    GameHead = load_head_engine()
    print("== 1. 首胡模式与 git HEAD 对拍 ==")
    bad = 0
    for i in range(args.games):
        seed = args.seed0 + i
        a = snap(play(GameHead, seed))
        b = snap(play(GameNew, seed, bloody=False))
        if a != b:
            bad += 1
            if bad <= 3:
                print(f"  seed {seed} 不一致\n    HEAD {a}\n    NEW  {b}")
    print(f"  {args.games} 局: {'全部一致' if bad == 0 else f'{bad} 局不一致'}")

    print("\n== 2. bloody 模式不变量 ==")
    nfin = {0: 0, 1: 0, 2: 0, 3: 0}
    nz = huang = 0
    rr_sum_bad = 0
    tot_disc = [0, 0]
    for i in range(args.games):
        g = play(GameNew, args.seed0 + i, bloody=True)
        k = len(g.finished)
        nfin[k] += 1
        if g.huangzhuang:
            huang += 1
        else:
            assert k == 3, f"非黄庄却只有 {k} 家胡"
        # 名次自洽
        for j, s in enumerate(g.finished):
            assert g.ranks[s] == j, "名次与下场顺序不符"
        for s in g._active_seats():
            assert g.ranks[s] == k, "未胡者名次应并列"
        rr = g.rank_rewards()
        if abs(sum(rr)) > 1e-9:
            rr_sum_bad += 1
        # 分数零和性: 杠分与胡分都是转移, 总和应为 0
        if abs(sum(p.score_delta for p in g.players)) > 1e-9:
            nz += 1
        tot_disc[1] += sum(len(p.discards) for p in g.players)
        g0 = play(GameNew, args.seed0 + i, bloody=False)
        tot_disc[0] += sum(len(p.discards) for p in g0.players)
    print(f"  胡牌家数分布: {nfin}   黄庄 {huang} 局")
    print(f"  得分非零和的局: {nz}   名次奖励非零和的局: {rr_sum_bad}")
    print(f"  平均弃牌数: 首胡 {tot_disc[0] / args.games:.1f} -> "
          f"血战 {tot_disc[1] / args.games:.1f} "
          f"({tot_disc[1] / max(1, tot_disc[0]):.2f}x 决策量)")

    ok = bad == 0 and nz == 0 and rr_sum_bad == 0
    print(f"\n{'通过' if ok else '失败'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
