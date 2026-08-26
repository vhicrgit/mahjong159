"""Rule bot v30: v29 discard logic + win-prob based peng decision.

v10/v29 的碰牌规则是"向听下降就碰(除非碰崩听牌)"。这忽略了:
- 碰后手牌变短、失去红中配合/改良空间, 向听虽降但胡牌概率未必升
- 向听不降但胡牌概率上升的碰(听口变宽)会被错过

v30 用同一 winp 递归直接比较: P(碰后最优弃牌形态, k) vs P(不碰现手, k),
差值超过 V30_MARGIN 才碰。杠决策沿用 v10 规则。
"""

import os

from ..rules.win import shanten
from .bot_v10 import _add
from .bot_v29 import Bot as V29Bot, _win_prob


class Bot(V29Bot):
    def __init__(self, game, seat: int):
        super().__init__(game, seat)
        self.peng_margin = float(os.environ.get("V30_MARGIN", 0.0))
        self.peng_max_sh = int(os.environ.get("V30_PENG_MAX_SH", 2))

    def decide_peng(self, tile: int) -> bool:
        p = self.game.players[self.seat]
        counts13 = tuple(p.hand_counts)
        s_now = shanten(list(counts13))
        c11 = _add(_add(counts13, tile, -1), tile, -1)
        s_after = min(shanten(list(_add(c11, d, -1)))
                      for d, n in enumerate(c11) if n > 0)
        if s_now > self.peng_max_sh or s_after > self.peng_max_sh:
            return super().decide_peng(tile)
        unseen = self._unseen_counts()
        k = min(self.k_cap, max(1, self.game.wall_remaining() // 4))
        winp_no = _win_prob(counts13, unseen, k)
        winp_peng = max(_win_prob(h, unseen, k)
                        for d, n in enumerate(c11) if n > 0
                        for h in [_add(c11, d, -1)]
                        if shanten(list(h)) == s_after)
        return winp_peng > winp_no + self.peng_margin
