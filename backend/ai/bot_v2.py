"""安康159 - 规则AI对手 v2

相对 v1 (bot_v1.py) 的强化（经规则复核后修正版）:
  159 翻牌是胡牌后从牌堆顶翻6张数出来的, 与手牌/听牌无关。
  因此期望得分只由三件事决定: 胡牌概率、杠分收益、放杠损失。

1. 阶段切换: 牌墙减少时向听数追赶不动 → 降低牌效权重,
   提高防放杠权重 (终局被杠一记 = -(n+1) 直接扣分)
2. 对手威胁感知: 对手已碰的牌, 打第4张 = 必被杠, 重罚
3. 碰决策: 听牌时只在碰后仍听牌才碰 (保持进攻连续性, v1 会碰崩听口)
4. 杠决策: 不破坏听口的前提下能杠则杠 (杠分是确定收益)
5. 出牌候选用"进张+后续摸牌潜力"排序, 同向听时选牌面效率高的
"""

from ..rules.win import shanten
from ..rules.ting import discard_options

RED = 27


class Bot:
    """规则Bot v2: 牌效 + 阶段切换 + 放杠防守"""

    def __init__(self, game, seat: int):
        self.game = game
        self.seat = seat

    def _visible_counts(self) -> list[int]:
        """可见牌计数(含自己手牌)"""
        visible = [0] * 28
        for q in self.game.players:
            for t in q.discards:
                visible[t] += 1
            for m in q.melds:
                visible[m["tile"]] += 3 if m["type"] == "peng" else 4
        p = self.game.players[self.seat]
        for t in range(28):
            visible[t] += p.hand_counts[t]
        return visible

    def _penged_tiles_by_others(self) -> set[int]:
        """对手碰过的牌: 打出第4张必被杠"""
        tiles = set()
        for q in self.game.players:
            if q.seat == self.seat:
                continue
            for m in q.melds:
                if m["type"] == "peng":
                    tiles.add(m["tile"])
        return tiles

    def _endgame_factor(self) -> float:
        """终局因子 0→1: 牌墙剩60→0 张线性上升"""
        wall = self.game.wall_remaining()
        return max(0.0, min(1.0, (60 - wall) / 60.0))

    def choose_discard(self) -> int:
        p = self.game.players[self.seat]
        counts = p.hand_counts
        opts = discard_options(counts)
        if not opts:
            return p.hand[-1]

        visible = self._visible_counts()
        penged = self._penged_tiles_by_others()
        eg = self._endgame_factor()

        best_tile, best_score = None, -1e9
        for o in opts:
            t = o["tile"]
            # 剩余进张(排除所有可见)
            wr = sum(max(0, 4 - visible[w]) for w in o["waits"])

            # 放杠风险: 对手碰了这张 → 打第4张必被杠(明杠-3)
            risk = 0.0
            if t != RED:
                if t in penged:
                    risk = 1.0
                else:
                    remain = max(0, 4 - visible[t])
                    risk = {3: 0.4, 2: 0.2, 1: 0.05, 0: 0.0}.get(
                        remain, 0.4)

            # 终局: 向听追赶不动 → 牌效权重降, 防守权重升
            sh_w = 100.0 * (1.0 - 0.5 * eg)
            risk_w = 25.0 * (1.0 + 1.5 * eg)

            score = -sh_w * o["shanten"] + 3.0 * wr - risk_w * risk
            if score > best_score:
                best_score, best_tile = score, t
        return best_tile

    def decide_peng(self, tile: int) -> bool:
        """碰: 向听数降低才碰; 听牌时只在碰后仍听牌才碰"""
        p = self.game.players[self.seat]
        counts = p.hand_counts
        before = shanten(counts)
        c2 = list(counts)
        c2[tile] -= 2
        after = shanten(c2)
        if after < before:
            if before == 0:
                return after == 0  # 听牌时碰崩听口 = 亏
            return True
        return False

    def decide_gang(self, tile: int, kind: str) -> bool:
        """杠: 杠分确定收益(+3); 唯一例外是不破坏听口的前提下才杠"""
        p = self.game.players[self.seat]
        counts = p.hand_counts
        s_before = shanten(counts)
        if kind == "ming":
            c2 = list(counts)
            c2[tile] -= 3
        elif kind == "an":
            c2 = list(counts)
            c2[tile] -= 4
        else:  # bu 补杠
            c2 = list(counts)
            c2[tile] -= 1
        s_after = shanten(c2)
        if s_before == 0 and s_after > 0:
            return False
        return True
