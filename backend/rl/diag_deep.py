"""诊断: 排查 BC 模型胜率低的根因

测试1: 4个规则Bot对战, seat 0 的胜率(排除位置偏差)
测试2: 在相同局面下, 模型 argmax vs 规则Bot 选择的逐手对比
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import torch
from backend.game.engine import Game
from backend.ai.bot import Bot
from backend.rl.model import build_model, legal_discard_mask
from backend.rl.features import encode_state
from backend.rl.net_bot import NetBot


def test_pure_rule_bots(n_games=200):
    """4个规则Bot对战, 统计各座位胜率"""
    print("=== 测试1: 4个规则Bot对战 ===")
    wins = [0] * 4
    scores = [0.0] * 4
    for i in range(n_games):
        g = Game(seed=400000 + i, human_seat=-1)
        bots = {s: Bot(g, s) for s in range(4)}
        guard = 0
        while g.phase != "game_over" and guard < 500:
            guard += 1
            if g.phase == "discard_wait":
                seat = g.turn
                g.action_discard(seat, bots[seat].choose_discard())
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
        if g.winner is not None:
            wins[g.winner] += 1
        for s in range(4):
            scores[s] += g.players[s].score_delta
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{n_games}: 各座位胜率 {[f'{w/(i+1):.1%}' for w in wins]}")
    print(f"  最终: 胜率 {[f'{w/n_games:.1%}' for w in wins]}")
    print(f"  场均: {[f'{s/n_games:+.2f}' for s in scores]}")
    return wins, scores


def test_model_vs_rulebot_decision_match(model_path, n_games=20):
    """在真实对局中, 逐手比较模型和规则Bot的选择"""
    print("\n=== 测试2: 模型 vs 规则Bot 逐手对比 ===")
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    model = build_model(ckpt["size"])
    model.load_state_dict(ckpt["model"])
    model.eval()

    match, total = 0, 0
    diff_examples = []

    for i in range(n_games):
        g = Game(seed=500000 + i, human_seat=-1)
        bots = {s: Bot(g, s) for s in range(4)}
        guard = 0
        while g.phase != "game_over" and guard < 500:
            guard += 1
            if g.phase == "discard_wait":
                seat = g.turn
                rule_choice = bots[seat].choose_discard()

                # 模型选择
                feat = encode_state(g, seat)
                x = torch.from_numpy(feat).unsqueeze(0)
                mask = legal_discard_mask(g.players[seat].hand_counts).unsqueeze(0)
                with torch.no_grad():
                    probs = model.policy(x, mask)[0]
                model_choice = int(probs.argmax().item())

                total += 1
                if model_choice == rule_choice:
                    match += 1
                else:
                    # 记录不同选择
                    from backend.rules.tiles import tile_short
                    from backend.rules.win import shanten
                    counts = g.players[seat].hand_counts
                    c_after_rule = list(counts); c_after_rule[rule_choice] -= 1
                    c_after_model = list(counts); c_after_model[model_choice] -= 1
                    diff_examples.append({
                        "rule": tile_short(rule_choice),
                        "model": tile_short(model_choice),
                        "rule_shanten": shanten(c_after_rule),
                        "model_shanten": shanten(c_after_model),
                        "rule_prob": float(probs[rule_choice]),
                        "model_prob": float(probs[model_choice]),
                    })

                # 用规则Bot的选择继续游戏(保持一致性)
                g.action_discard(seat, rule_choice)
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

    print(f"  一致率: {match}/{total} = {match/total:.1%}")
    if diff_examples:
        # 统计: 模型选错时, shanten 差多少
        worse_shanten = sum(1 for e in diff_examples if e["model_shanten"] > e["rule_shanten"])
        equal_shanten = sum(1 for e in diff_examples if e["model_shanten"] == e["rule_shanten"])
        print(f"  不同选择中: 模型shanten更高={worse_shanten}, 相同={equal_shanten}, 总不同={len(diff_examples)}")
        print(f"  前5个不同选择:")
        for e in diff_examples[:5]:
            print(f"    规则打{e['rule']}(向听{e['rule_shanten']},P={e['rule_prob']:.2f}) vs "
                  f"模型打{e['model']}(向听{e['model_shanten']},P={e['model_prob']:.2f})")


def test_netbot_seat_variations(model_path, n_games=200):
    """NetBot 在不同座位的胜率"""
    print("\n=== 测试3: NetBot 在不同座位的胜率 ===")
    for seat in range(4):
        wins, total = 0, 0.0
        for i in range(n_games):
            g = Game(seed=600000 + i, human_seat=-1)
            net = NetBot(g, seat, model_path)
            rule_bots = {s: Bot(g, s) for s in range(4) if s != seat}
            def get_bot(s):
                return net if s == seat else rule_bots[s]
            guard = 0
            while g.phase != "game_over" and guard < 500:
                guard += 1
                if g.phase == "discard_wait":
                    s2 = g.turn
                    g.action_discard(s2, get_bot(s2).choose_discard())
                elif g.phase == "react_wait":
                    s2 = list(g.pending_actions.keys())[0]
                    b = get_bot(s2)
                    if g.pending_actions[s2].get("gang") and \
                            b.decide_gang(g.last_discard, "ming"):
                        g.action_gang(s2)
                    elif g.pending_actions[s2].get("peng") and \
                            b.decide_peng(g.last_discard):
                        g.action_peng(s2)
                    else:
                        g.action_pass(s2)
            total += g.players[seat].score_delta
            if g.winner == seat:
                wins += 1
        print(f"  座位{seat}: 胜率 {wins/n_games:.1%}, 场均 {total/n_games:+.2f}")


if __name__ == "__main__":
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "..", "..", "models", "bc_only_small.pt")

    # 测试1: 4规则Bot
    test_pure_rule_bots(200)

    # 测试2: 逐手对比
    if os.path.exists(model_path):
        test_model_vs_rulebot_decision_match(model_path, 20)
        # 测试3: 不同座位
        test_netbot_seat_variations(model_path, 200)
