"""C 版牌型价值 E(libmjcore.so) 与 Python HandAnalyzer 的逐位对拍 + 性能验收。

用例: 截图手牌(用户实测用例) + 真实对局里采样的若干 14 张决策点。
口径: rho=1, kaizen 开, kai_max ∈ {1,2}, kai_topk=6。
判定: 每个可打牌的 E 两边差 < 1e-9 视为一致(浮点同序累加应逐位相等)。

用法: python -m tools.perf.test_hv_c_parity
"""

import random
import time

from backend.analysis import hv_native
from backend.analysis.hand_value import HandAnalyzer
from backend.game.engine import Game
from backend.rules.tiles import tile_name


def py_table(hand, visible, kai_max, kai_topk=6, rho=1.0):
    az = HandAnalyzer(hand, visible, rho=rho, kaizen=True,
                      kai_max=kai_max, kai_topk=kai_topk)
    out = {}
    for t in range(28):
        if hand[t] <= 0:
            continue
        h = list(hand)
        h[t] -= 1
        out[t] = az.E(tuple(h), az.u0)
    return out


def c_table(hand, visible, kai_max, kai_topk=6, rho=1.0):
    hv_native.set_hand(hand, visible, rho=rho, kaizen=True,
                       kai_max=kai_max, kai_topk=kai_topk)
    out = {}
    for t in range(28):
        if hand[t] <= 0:
            continue
        out[t] = hv_native.e_after_discard(t)
    return out


def sample_hands(n, seed=42):
    """从真实对局采样 14 张决策点的 (hand, visible)。"""
    rng = random.Random(seed)
    cases = []
    g_seeds = [rng.randrange(10 ** 6) for _ in range(60)]
    for gs in g_seeds:
        g = Game(seed=gs, human_seat=-1)
        # 直接读引擎状态: 各家轮流抽打(用 v31 走几步), 采集 discard_wait 时的状态
        from backend.ai.bot_native import NativeV31
        bots = {i: NativeV31(g, i) for i in range(4)}
        guard = 0
        while g.phase != "game_over" and guard < 500 and len(cases) < n * 4:
            guard += 1
            if g.phase == "discard_wait":
                s = g.turn
                hand = g.players[s].hand_counts
                visible = [0] * 28
                for q in g.players:
                    for t in q.discards:
                        visible[t] += 1
                    for m in q.melds:
                        visible[m["tile"]] += 3 if m["type"] == "peng" else 4
                for t, c in enumerate(hand):
                    visible[t] += c
                cases.append((hand, visible))
                g.action_discard(s, bots[s].choose_discard())
            else:
                ss = list(g.pending_actions.keys())[0]
                pend = g.pending_actions[ss]
                b = bots[ss]
                if pend.get("gang") and b.decide_gang(g.last_discard, "ming"):
                    g.action_gang(ss)
                elif pend.get("peng") and b.decide_peng(g.last_discard):
                    g.action_peng(ss)
                else:
                    g.action_pass(ss)
    rng.shuffle(cases)
    return cases[:n]


def main():
    # 截图手牌(用户用例)
    hand = [0] * 28
    for t in [0, 2, 4, 5, 5, 11, 13, 15, 16, 21, 22, 22, 22, 25]:
        hand[t] += 1
    visible = list(hand)
    cases = [(list(hand), list(visible))]
    cases += sample_hands(10)

    bad = 0
    for kai_max, ncase in ((1, len(cases)), (2, 5)):
        for i, (h, v) in enumerate(cases[:ncase]):
            t0 = time.time()
            tp = py_table(h, v, kai_max)
            tc = c_table(h, v, kai_max)
            for t in tp:
                d = abs(tp[t] - tc[t])
                if d > 1e-9:
                    bad += 1
                    print(f"不一致 kai_max={kai_max} case{i} tile{tile_name(t)}: "
                          f"py={tp[t]!r} c={tc[t]!r}")
            print(f"  kai_max={kai_max} case{i} ok ({time.time()-t0:.1f}s)",
                  flush=True)
        print(f"kai_max={kai_max}: {min(ncase, len(cases))} 手牌对拍完毕, "
              f"累计不一致 {bad}", flush=True)

    # 性能验收: 截图手牌整表, kai_max=2
    for kai_max in (1, 2):
        t0 = time.perf_counter()
        tc = c_table(hand, visible, kai_max)
        dt = time.perf_counter() - t0
        top = sorted(tc.items(), key=lambda kv: kv[1])[:3]
        print(f"C 整表 kai_max={kai_max}: {dt * 1000:.0f}ms  "
              f"最优: " + " ".join(f"{tile_name(t)}={e:.2f}" for t, e in top))
    print("OK" if bad == 0 else f"FAIL: {bad} 处不一致")


if __name__ == "__main__":
    main()
