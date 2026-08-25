"""安康159 - 规则AI对手

策略:
- 出牌: 牌效优先(向听数最小化 + 进张最大化) + 放杠风险惩罚
- 碰: 碰后向听数降低才碰
- 杠: 能杠则杠(杠分净收益为正), 但手握听牌时谨慎补杠破坏听口
- 目标: 快速自摸, 不防守过度(159玩法无点炮, 防守压力小)
"""

from ..rules.tiles import counts_from_tiles, tile_short
from ..rules.win import shanten
from ..rules.ting import discard_options, waiting_tiles

RED = 27


class Bot:
    def __init__(self, game, seat: int):
        self.game = game
        self.seat = seat

    def choose_discard(self) -> int:
        """14张状态选择打出哪张"""
        p = self.game.players[self.seat]
        counts = p.hand_counts
        opts = discard_options(counts)
        if not opts:
            return p.hand[-1]
        # 可见计数(用于估计进张剩余)
        visible = [0] * 28
        for q in self.game.players:
            for t in q.discards:
                visible[t] += 1
            for m in q.melds:
                visible[m["tile"]] += 3 if m["type"] == "peng" else 4
        if q.seat == self.seat:
            for t in p.hand:
                visible[t] += 1

        best_tile, best_score = None, -1e9
        for o in opts:
            t = o["tile"]
            # 剩余进张
            wr = sum(max(0, 4 - visible[w] - counts[w]) for w in o["waits"])
            # 放杠风险粗估
            risk = 0.0
            if t != RED:
                remain = 4 - visible[t] - counts[t]
                for q in self.game.players:
                    if q.seat != self.seat and any(
                            m["tile"] == t and m["type"] == "peng"
                            for m in q.melds):
                        risk = 1.0
                        break
                else:
                    risk = {3: 0.4, 2: 0.2, 1: 0.05, 0: 0.0}.get(
                        max(0, remain), 0.4)
            score = -100 * o["shanten"] + 3 * wr - 25 * risk
            if score > best_score:
                best_score, best_tile = score, t
        return best_tile

    def decide_peng(self, tile: int) -> bool:
        """是否碰: 碰后向听数降低才碰"""
        p = self.game.players[self.seat]
        counts = p.hand_counts
        before = shanten(counts)
        # 模拟碰: 手牌减2张(碰出后手牌-2, 副露+1)
        c2 = list(counts)
        c2[tile] -= 2
        after = shanten(c2)
        return after < before

    def decide_gang(self, tile: int, kind: str) -> bool:
        """是否杠: 杠分净收益为正, 一般杠; 但若杠会破坏听牌且进张很宽, 谨慎"""
        p = self.game.players[self.seat]
        counts = p.hand_counts
        s_before = shanten(counts)
        if kind == "ming":
            c2 = list(counts)
            c2[tile] -= 3
        elif kind == "an":
            c2 = list(counts)
            c2[tile] -= 4
        else:  # bu
            c2 = list(counts)
            c2[tile] -= 1
        s_after = shanten(c2)
        # 杠后向听数不变差才杠(听牌状态下不破坏听口)
        if s_before == 0 and s_after > 0:
            return False
        return True
