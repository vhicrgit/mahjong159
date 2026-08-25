"""安康159 - 模拟评估训练 (Expert Iteration)

对每个决策状态, 尝试每种可能的出牌, 用规则Bot模拟后续对局,
评估哪种出牌实际赢最多, 然后训练模型选择最优出牌。

这比 RL 更稳定: 直接优化胜率, 不受信用分配问题困扰。
"""

import argparse
import copy
import os
import time
import multiprocessing as mp
import numpy as np
import torch
import torch.nn.functional as F

from .model import build_model, legal_discard_mask, N_ACTIONS
from .selfplay import generate_dataset
from .vec_selfplay import VectorizedSelfPlay, _rule_score
from .train import get_device, train_bc
from .evaluate import play_eval_game
from .features_v2 import encode_state
from ..game.engine import Game
from ..ai.bot import Bot
from ..rules.win import shanten
from ..rules.ting import discard_options


def _simulate_from_state(game_state_dict, discard_tile, n_sim=5, seed0=0):
    """从某个局面模拟打出 tile 后的平均得分"""
    scores = []
    for sim in range(n_sim):
        g = Game.__new__(Game)
        g.__dict__.update(copy.deepcopy(game_state_dict))
        g.rng = __import__('random').Random(seed0 + sim)

        # 打出候选牌
        seat = g.turn
        g.players[seat].hand.remove(discard_tile)
        g.players[seat].discards.append(discard_tile)
        g.last_discard = discard_tile
        g.last_discarder = seat

        # 检查碰/杠
        g.pending_actions = {}
        if discard_tile != 27:
            for other in range(4):
                if other == seat:
                    continue
                op = g.players[other]
                cnt = op.hand.count(discard_tile)
                can_peng = cnt >= 2
                can_gang = cnt >= 3
                if can_peng or can_gang:
                    g.pending_actions[other] = {"peng": can_peng, "gang": can_gang}

        if g.pending_actions:
            g.phase = "react_wait"
        else:
            g._next_draw()

        # 用规则Bot完成剩余对局
        bots = {s: Bot(g, s) for s in range(4)}
        guard = 0
        while g.phase != "game_over" and guard < 500:
            guard += 1
            if g.phase == "discard_wait":
                s = g.turn
                g.action_discard(s, bots[s].choose_discard())
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
        scores.append(g.players[seat].score_delta)
    return np.mean(scores)


def _evaluate_actions_worker(args):
    """Worker: 评估一个状态的所有候选出牌"""
    state_dict, hand_counts, feat, seat, n_sim, seed0 = args
    legal = [t for t in range(28) if hand_counts[t] > 0]
    scores = {}
    for t in legal:
        scores[t] = _simulate_from_state(state_dict, t, n_sim, seed0)
    best_tile = max(scores, key=scores.get)
    return feat, best_tile, scores


def generate_expert_data(model, n_states, device, n_sim=5, workers=16):
    """生成专家数据: 对每个状态, 模拟评估所有候选出牌"""
    # 1. 用当前模型自对弈收集状态
    print(f"  收集 {n_states} 个状态...")
    n_games = max(64, n_states // 35)  # 每局约35个决策
    engine = VectorizedSelfPlay(model, n_games, device, seed0=0,
                                 model_seats=[0])
    results = engine.run(temperature=0.5)

    # 2. 收集状态
    states = []
    for r in results:
        g_idx = results.index(r)
        for seat, feat, act, lp, val, regret in r["records"]:
            # 需要重建游戏状态用于模拟
            # 但我们没有保存完整状态... 需要改引擎
            pass

    # 这个方案需要保存完整游戏状态, 太复杂了
    # 换一种方式: 直接在自对弈过程中记录每个决策的规则Bot评分
    # 然后用评分差异作为训练信号
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", type=str, required=True)
    ap.add_argument("--size", type=str, default="base")
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--games-per-iter", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.05)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--out", type=str, default="models/expert_iter.pt")
    ap.add_argument("--eval-games", type=int, default=100)
    args = ap.parse_args()

    device = get_device()
    print(f"Expert Iteration 训练: {args.size} 模型, {device}")

    model = build_model(args.size).to(device)
    ckpt = torch.load(args.init, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    print(f"加载 BC 模型: {args.init}")

    torch.save({"model": model.state_dict(), "size": args.size}, args.out)

    # 评估基线
    from .evaluate import play_eval_game
    wins, total = 0, 0.0
    for i in range(args.eval_games):
        sd, won = play_eval_game(200000 + i, args.out)
        total += sd
        wins += won
    print(f"BC 基线: 胜率 {wins/args.eval_games:.1%}, 场均 {total/args.eval_games:+.2f}")

    # TODO: 实现 expert iteration
    print("Expert iteration 训练尚未完全实现, 请先使用 BC 模型")


if __name__ == "__main__":
    main()
