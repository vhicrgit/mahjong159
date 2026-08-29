"""牌型价值 Bot(HV): 用 tools/hand_value.py 的期望巡数做全部决策。

- 出牌: 打出后期望巡数 E 最小的牌(分析器口径: 自摸+碰通道, ρ=1, 无换型层)
- 碰:   碰后最优牌型的 E 严格小于当前 E 才碰
- 杠:   与 v1/v10/v31 同口径(不破坏已成听口才杠)

注意性能取舍: 换型层(kaizen)把单手分析从毫秒级拉到秒级, 只改善绝对值不改善排序,
bot 对战按 kaizen=False 跑。要做精细分析请用 tools/hand_value.py 交互式跑。
"""

from ..analysis.hand_value import HandAnalyzer
from ..native import native

RED = 27


class Bot:
    def __init__(self, game, seat: int, rho: float = 1.0):
        self.game = game
        self.seat = seat
        self.rho = rho

    def _analyzer(self):
        visible = [0] * 28
        for q in self.game.players:
            for t in q.discards:
                visible[t] += 1
            for m in q.melds:
                visible[m["tile"]] += 3 if m["type"] == "peng" else 4
        for t, n in enumerate(self.game.players[self.seat].hand_counts):
            visible[t] += n
        return HandAnalyzer(self.game.players[self.seat].hand_counts,
                            visible, rho=self.rho, kaizen=False)

    def choose_discard(self) -> int:
        p = self.game.players[self.seat]
        hand = list(p.hand_counts)
        az = self._analyzer()
        best_t, best_e = None, 1e18
        for t in range(28):
            if hand[t] <= 0:
                continue
            h = list(hand)
            h[t] -= 1
            e = az.E(tuple(h), az.u0)
            if e < best_e:
                best_e, best_t = e, t
        return best_t if best_t is not None else p.hand[-1]

    def decide_peng(self, tile: int) -> bool:
        p = self.game.players[self.seat]
        hand = list(p.hand_counts)
        az = self._analyzer()
        e_before = az.E(tuple(hand), az.u0)
        h2 = list(hand)
        h2[tile] -= 2
        best_after = 1e18
        for d in range(28):
            if h2[d] <= 0:
                continue
            h3 = list(h2)
            h3[d] -= 1
            best_after = min(best_after, az.E(tuple(h3), az.u0))
        return best_after < e_before

    def decide_gang(self, tile: int, kind: str) -> bool:
        p = self.game.players[self.seat]
        before = native.shanten(p.hand_counts)
        c = list(p.hand_counts)
        if kind == "ming":
            c[tile] -= 3
        elif kind == "an":
            c[tile] -= 4
        else:
            c[tile] -= 1
        after = native.shanten(c)
        return not (before == 0 and after > 0)
