import time
from backend.game.engine import Game
from backend.ai.bot_oracle import Bot as Oracle
from backend.ai.bot_v1 import Bot as V1

g = Game(seed=77, human_seat=-1)
oracle = Oracle(g, 0, beam=12)
bots = {i: V1(g, i) for i in range(1, 4)}
bots[0] = oracle

t_total, n = 0.0, 0
guard = 0
while g.phase != "game_over" and guard < 300:
    guard += 1
    if g.phase == "discard_wait":
        seat = g.turn
        if seat == 0:
            t0 = time.time()
            tile = oracle.choose_discard()
            t_total += time.time() - t0
            n += 1
        else:
            tile = bots[seat].choose_discard()
        g.action_discard(seat, tile)
    elif g.phase == "react_wait":
        s = list(g.pending_actions.keys())[0]
        b = bots[s]
        if g.pending_actions[s].get("gang") and \
                b.decide_gang(g.last_discard, "ming"):
            g.action_gang(s)
        elif g.pending_actions[s].get("peng") and b.decide_peng(g.last_discard):
            g.action_peng(s)
        else:
            g.action_pass(s)

print(f"oracle beam search: {t_total/max(n,1)*1000:.1f}ms/决策 ({n} 次)")
print(f"→ PIMC 用 K 个世界: K=8 约 {t_total/max(n,1)*8*1000:.0f}ms/决策")
