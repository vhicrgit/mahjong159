"""可行性验证: 给牌型价值递推加"换型层"(同向听但进张变宽的摸牌也算进展)。

回答三个问题:
1. 状态数/耗时会不会爆炸
2. 能不能消除"改良盲"导致的系统性 +0.6 巡偏差(与 MC 对照)
3. 不同 margin(进张放宽幅度阈值)的敏感性

不合并进 tools/hand_value.py, 先看数字。
"""

import argparse
import sys
import time

sys.path.insert(0, ".")

from backend.native import native
from tools.perf.arbitrate_961000 import replay_to

RED = 27
_useful_set_cache = {}


def useful_set(hand):
    """摸到能降向听(含直接胡)的牌集合 —— 只与手牌有关, 与池子无关, 可缓存。"""
    key = bytes(hand)
    s = _useful_set_cache.get(key)
    if s is not None:
        return s
    base = native.shanten(list(hand))
    out = []
    for t in range(28):
        if hand[t] >= 4:
            continue
        h = list(hand)
        h[t] += 1
        if native.shanten(h) < base:
            out.append(t)
    _useful_set_cache[key] = out
    return out


class KaizenAnalyzer:
    """等待时间递推 + 换型层: 摸到不淘汰向听但让有效张变宽 >= margin 的牌也转移。"""

    def __init__(self, u0, margin=2, max_kai=2):
        self.u0 = u0
        self.margin = margin
        self.max_kai = max_kai
        self.memo = {}
        self.states = 0

    def ukeire(self, hand, u):
        return sum(u[t] for t in useful_set(hand) if u[t] > 0)

    def best_play(self, h14, u):
        """v10-lite: 最小向听, 同向听取最大进张。返回 (打后手牌, 向听, 进张)。"""
        best = None
        for d in range(28):
            if h14[d] <= 0:
                continue
            h = list(h14)
            h[d] -= 1
            s = native.shanten(h)
            uk = self.ukeire(h, u)
            key = (s, -uk)
            if best is None or key < best[0]:
                best = (key, tuple(h), s, uk)
        return best[1], best[2], best[3]

    def E(self, hand, u, kai=0):
        key = (hand, u, kai)
        v = self.memo.get(key)
        if v is not None:
            return v
        self.states += 1
        s = native.shanten(list(hand))
        N = sum(u)
        # 主通道: 降向听/胡
        useful = [(t, u[t]) for t in useful_set(hand) if u[t] > 0]
        # 换型通道: 不降向听但进张变宽
        kai_tiles = []
        if kai < self.max_kai:
            uk0 = self.ukeire(hand, u)
            for t in range(28):
                if u[t] <= 0 or any(t == x for x, _ in useful):
                    continue
                h14 = list(hand)
                h14[t] += 1
                h2, s2, uk2 = self.best_play(h14, u)
                if s2 == s and uk2 >= uk0 + self.margin:
                    kai_tiles.append((t, u[t], h2))
        U = sum(w for _, w in useful) + sum(w for _, w, _ in kai_tiles)
        if U <= 0:
            self.memo[key] = float(N) + 2.0 * s
            return self.memo[key]
        wait = (N + 1.0) / (U + 1.0)
        val = wait
        for t, w in useful:
            p = w / U
            h14 = list(hand)
            h14[t] += 1
            if native.is_win(h14):
                continue
            u2 = list(u)
            u2[t] -= 1
            h2, _, _ = self.best_play(h14, u2)
            val += p * self.E(h2, tuple(u2), 0)
        for t, w, h2 in kai_tiles:
            p = w / U
            u2 = list(u)
            u2[t] -= 1
            val += p * self.E(h2, tuple(u2), kai + 1)
        self.memo[key] = val
        return val


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--margin", type=int, default=2)
    ap.add_argument("--max-kai", type=int, default=2)
    args = ap.parse_args()

    g = replay_to(961000, 0, 4)
    p = g.players[0]
    hand = list(p.hand_counts)
    visible = [0] * 28
    for q in g.players:
        for t in q.discards:
            visible[t] += 1
        for m in q.melds:
            visible[m["tile"]] += 3 if m["type"] == "peng" else 4
    for t, n in enumerate(hand):
        visible[t] += n
    u0 = tuple(max(0, 4 - v) for v in visible)

    print("目标(MC, 无碰): 打5饼后 7.80 巡, 打7条后 7.66 巡")
    print(f"参数: margin={args.margin} max_kai={args.max_kai}")
    for d, name in ((13, "打5饼"), (6, "打7条")):
        h = list(hand)
        h[d] -= 1
        for margin, mk in ((99, 0), (args.margin, args.max_kai)):
            az = KaizenAnalyzer(u0, margin=margin, max_kai=mk)
            t0 = time.process_time()
            e = az.E(tuple(h), u0, 0)
            dt = time.process_time() - t0
            tag = "无换型层" if margin == 99 else \
                f"换型层(margin={margin},max_kai={mk})"
            print(f"  {name} {tag}: E={e:.2f} 巡, "
                  f"状态 {az.states}, 耗时 {dt:.1f}s")


if __name__ == "__main__":
    main()
