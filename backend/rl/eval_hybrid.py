"""测试混合决策不同 alpha 值的胜率"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backend.game.engine import Game
from backend.ai.bot import Bot
from backend.rl.hybrid_bot import HybridBot


def eval_hybrid(model_path, alpha, n_games=200):
    wins, total = 0, 0.0
    for i in range(n_games):
        g = Game(seed=900000 + i, human_seat=-1)
        net = HybridBot(g, 0, model_path, alpha=alpha)
        rule_bots = {s: Bot(g, s) for s in range(1, 4)}
        guard = 0
        while g.phase != "game_over" and guard < 500:
            guard += 1
            if g.phase == "discard_wait":
                seat = g.turn
                bot = net if seat == 0 else rule_bots[seat]
                g.action_discard(seat, bot.choose_discard())
            elif g.phase == "react_wait":
                s = list(g.pending_actions.keys())[0]
                b = net if s == 0 else rule_bots[s]
                if g.pending_actions[s].get("gang") and \
                        b.decide_gang(g.last_discard, "ming"):
                    g.action_gang(s)
                elif g.pending_actions[s].get("peng") and \
                        b.decide_peng(g.last_discard):
                    g.action_peng(s)
                else:
                    g.action_pass(s)
        total += g.players[0].score_delta
        if g.winner == 0:
            wins += 1
    return wins / n_games, total / n_games


if __name__ == "__main__":
    model_path = "models/bc_base_50k.pt"
    print(f"模型: {model_path}")
    for alpha in [0.0, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0]:
        wr, avg = eval_hybrid(model_path, alpha, 200)
        print(f"alpha={alpha:.2f}: 胜率 {wr:.1%}, 场均 {avg:+.2f}")
