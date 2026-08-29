"""原生(C)加速版规则 Bot: 与 bot_v1 / bot_v10 / bot_v31 逐位同口径。

只把"决策"搬进 C —— 引擎仍是 Python。理由: 一局里引擎只走 ~60 次状态转移,
而 Bot 决策要算 ~70 万次 shanten, 成本全在决策侧, 换掉决策就够了, 而且
不动引擎就没有规则走样的风险。

对应关系:
  NativeV1  <-> backend.ai.bot_v1.Bot
  NativeV10 <-> backend.ai.bot_v10.Bot
  NativeV31 <-> backend.ai.bot_v31.Bot   (与 v10 只差 decide_peng)
"""

from ..native import native
from .bot_v10 import Bot as V10Bot

RED = 27


class _V10Base(V10Bot):
    BOT_ID = 10

    def choose_discard(self) -> int:
        p = self.game.players[self.seat]
        hand = p.hand_counts
        unseen = self._unseen_counts()
        penged = [0] * 28
        for t in self._penged_by_others():
            penged[t] = 1
        t = native.choose_discard_v10(
            hand, unseen, penged, self._endgame_factor(),
            self.shanten_weight, self.ukeire_weight, self.cont_weight,
            self.risk_weight, self.cont_max_shanten)
        return t if t >= 0 else p.hand[-1]

    def decide_peng(self, tile: int) -> bool:
        return native.decide_peng(
            self.BOT_ID, self.game.players[self.seat].hand_counts, tile)

    def decide_gang(self, tile: int, kind: str) -> bool:
        return native.decide_gang(
            self.BOT_ID, self.game.players[self.seat].hand_counts, tile, kind)


class NativeV10(_V10Base):
    BOT_ID = 10


class NativeV31(_V10Base):
    """v31 = v10 + 副露感知的 decide_peng(碰后还要打一张才比较向听)。"""
    BOT_ID = 31


class NativeV1:
    BOT_ID = 1

    def __init__(self, game, seat: int):
        self.game = game
        self.seat = seat

    def _visible_and_penged(self):
        """复刻 bot_v1 的 visible 计算, 包括其 `if q.seat == self.seat` 写在
        循环外的既有行为(只有座位3 才把自己手牌计入 visible)。改动会改变
        v1 的策略, 因此这里原样保留。"""
        visible = [0] * 28
        q = None
        for q in self.game.players:
            for t in q.discards:
                visible[t] += 1
            for m in q.melds:
                visible[m["tile"]] += 3 if m["type"] == "peng" else 4
        if q is not None and q.seat == self.seat:
            for t in self.game.players[self.seat].hand:
                visible[t] += 1
        penged = [0] * 28
        for qq in self.game.players:
            if qq.seat == self.seat:
                continue
            for m in qq.melds:
                if m["type"] == "peng":
                    penged[m["tile"]] = 1
        return visible, penged

    def choose_discard(self) -> int:
        p = self.game.players[self.seat]
        visible, penged = self._visible_and_penged()
        t = native.choose_discard_v1(p.hand_counts, visible, penged)
        return t if t >= 0 else p.hand[-1]

    def decide_peng(self, tile: int) -> bool:
        return native.decide_peng(1, self.game.players[self.seat].hand_counts,
                                  tile)

    def decide_gang(self, tile: int, kind: str) -> bool:
        return native.decide_gang(1, self.game.players[self.seat].hand_counts,
                                  tile, kind)
