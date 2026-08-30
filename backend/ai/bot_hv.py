"""牌型价值 Bot(HV): 用 tools/hand_value.py 的期望巡数做全部决策。

- 出牌: 打出后期望巡数 E 最小的牌(分析器口径: 自摸+碰通道, ρ=1, 无换型层)
- 碰:   碰后最优牌型的 E 严格小于当前 E 才碰
- 杠:   与 v1/v10/v31 同口径(不破坏已成听口才杠)

模块级函数 choose_discard/decide_peng/decide_gang 与 Bot 的方法一一对应、
逻辑完全一致 —— 对手牌型分析器(backend/analysis/opp_model.py)用它们反演
学者 bot 的行为; 改决策逻辑时两边始终是同一份代码。

注意性能取舍: 换型层(kaizen)把单手分析从毫秒级拉到秒级, 只改善绝对值不改善排序,
bot 对战按 kaizen=False 跑。要做精细分析请用 tools/hand_value.py 交互式跑。
"""

from ..analysis.hand_value import HandAnalyzer
from ..native import native

RED = 27


def _mk_analyzer(hand_counts, visible_counts, rho, memo,
                 u_eff=None, held_exp=None):
    return HandAnalyzer(hand_counts, visible_counts, rho=rho,
                        kaizen=False, memo=memo,
                        u_eff=u_eff, held_exp=held_exp)


def choose_discard(hand_counts, visible_counts, rho: float = 1.0,
                   memo=None, u_eff=None, held_exp=None) -> int | None:
    """学者出牌: argmin E(打后手牌)。hand_counts 为 3k+2 张。平局取小 index。
    u_eff/held_exp: 对手牌型分析器给的先验(不给=均匀假设)。"""
    hand = list(hand_counts)
    az = _mk_analyzer(hand, visible_counts, rho, memo, u_eff, held_exp)
    best_t, best_e = None, 1e18
    for t in range(28):
        if hand[t] <= 0:
            continue
        h = list(hand)
        h[t] -= 1
        e = az.E(tuple(h), az.u_eff)
        if e < best_e:
            best_e, best_t = e, t
    return best_t


def decide_peng(hand_counts, visible_counts, tile: int,
                rho: float = 1.0, memo=None,
                u_eff=None, held_exp=None) -> bool:
    """学者碰判定: 碰后最优 E 严格下降才碰。hand_counts 为 3k+1 张。"""
    hand = list(hand_counts)
    if hand[tile] < 2:
        return False
    az = _mk_analyzer(hand, visible_counts, rho, memo, u_eff, held_exp)
    e_before = az.E(tuple(hand), az.u_eff)
    h2 = list(hand)
    h2[tile] -= 2
    best_after = 1e18
    for d in range(28):
        if h2[d] <= 0:
            continue
        h3 = list(h2)
        h3[d] -= 1
        best_after = min(best_after, az.E(tuple(h3), az.u_eff))
    return best_after < e_before


def decide_gang(hand_counts, tile: int, kind: str) -> bool:
    """学者杠判定: 不破坏已成听口才杠(与 v1/v10/v31 同口径)。"""
    c = list(hand_counts)
    before = native.shanten(c)
    if kind == "ming":
        c[tile] -= 3
    elif kind == "an":
        c[tile] -= 4
    else:
        c[tile] -= 1
    after = native.shanten(c)
    return not (before == 0 and after > 0)


class Bot:
    def __init__(self, game, seat: int, rho: float = 1.0, memo=None):
        self.game = game
        self.seat = seat
        self.rho = rho
        self.memo = memo if memo is not None else {}

    def _visible(self):
        visible = [0] * 28
        for q in self.game.players:
            for t in q.discards:
                visible[t] += 1
            for m in q.melds:
                visible[m["tile"]] += 3 if m["type"] == "peng" else 4
        for t, n in enumerate(self.game.players[self.seat].hand_counts):
            visible[t] += n
        return visible

    def choose_discard(self) -> int:
        p = self.game.players[self.seat]
        t = choose_discard(p.hand_counts, self._visible(), self.rho, self.memo)
        return t if t is not None else p.hand[-1]

    def decide_peng(self, tile: int) -> bool:
        return decide_peng(self.game.players[self.seat].hand_counts,
                           self._visible(), tile, self.rho, self.memo)

    def decide_gang(self, tile: int, kind: str) -> bool:
        return decide_gang(self.game.players[self.seat].hand_counts,
                           tile, kind)
