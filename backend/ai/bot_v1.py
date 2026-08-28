"""安康159 - 规则AI对手

策略:
- 出牌: 牌效优先(向听数最小化 + 进张最大化) + 放杠风险惩罚
- 碰: 碰后向听数降低才碰
- 杠: 能杠则杠(杠分净收益为正), 但手握听牌时谨慎补杠破坏听口
- 目标: 快速自摸, 不防守过度(159玩法无点炮, 防守压力小)

[口径变更] 原本 decide_peng/decide_gang 直接对副露后的 11/10 张暗牌调
shanten(), 而 shanten 的公式 8-2m-t-p 硬编码了 13 张手牌凑 4 面子, 导致
向听被高估约 2*副露数, "碰后向听降低"几乎永远不成立 —— 实测表现为
规则Bot 从不鸣牌。现已改用 shanten_with_melds 修正。
注意: 本文件曾作为全项目评测基线("1个测试Bot@座位0 + 3×v1"),
修复后基线强度已变强, docs/ceiling_and_bots.md 中旧数值不再与新测量同口径。
"""

from ..rules.tiles import counts_from_tiles, tile_short
from ..rules.win import shanten, shanten_with_melds
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
            wr = int(sum(max(0, 4 - visible[w] - counts[w]) for w in o["waits"]))
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
        """是否碰: 碰后向听数降低才碰(副露感知)"""
        p = self.game.players[self.seat]
        n_melds = len(p.melds)
        counts = list(p.hand_counts)
        before = shanten_with_melds(counts, n_melds)
        # 碰后: 暗牌-2, 副露+1, 且必须再打出一张
        c11 = list(counts)
        c11[tile] -= 2
        after = 99
        for d, cnt in enumerate(c11):
            if cnt <= 0:
                continue
            c10 = list(c11)
            c10[d] -= 1
            after = min(after, shanten_with_melds(c10, n_melds + 1))
        return after < before

    def decide_gang(self, tile: int, kind: str) -> bool:
        """是否杠: 杠分净收益为正, 一般杠; 但若杠会破坏听牌则不杠(副露感知)"""
        p = self.game.players[self.seat]
        n_melds = len(p.melds)
        counts = list(p.hand_counts)
        s_before = shanten_with_melds(counts, n_melds)
        c2 = list(counts)
        if kind == "ming":
            c2[tile] -= 3
            n_after = n_melds + 1
        elif kind == "an":
            c2[tile] -= 4
            n_after = n_melds + 1
        else:  # bu: 碰转杠, 副露数不变
            c2[tile] -= 1
            n_after = n_melds
        s_after = shanten_with_melds(c2, n_after)
        # 杠后向听数不变差才杠(听牌状态下不破坏听口)
        if s_before == 0 and s_after > 0:
            return False
        return True
