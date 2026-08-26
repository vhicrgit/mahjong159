"""Rule bot v31: v10 + 副露感知向听 —— 修复规则Bot"从不碰杠"的隐性bug。

背景: shanten() 公式硬编码 13 张手牌(8=2*4面子), 副露后暗牌变短(11/10张)
时向听被高估约 2*副露数。v1-v30 所有规则Bot 的 decide_peng 用
shanten(11张) < shanten(13张) 判断, 因高估几乎永远不成立 —— 实测 400 局
中 v10 有 169 次碰牌机会, 0 次判定碰(v21"不碰不杠"与 v10 无差异的真相)。

v31 修复:
- decide_peng/decide_gang 用 shanten_with_melds(暗牌, 副露数) 正确比较
- 无副露时 choose_discard 完全委托 v10(逐bit一致, 共享缓存)
- 有副露后用副露感知的向听/进张/两步推演重建 v10 同款出牌评估
"""

import os
from functools import lru_cache

from ..rules.win import is_win, shanten_with_melds
from .bot_v10 import Bot as V10Bot, _add

RED = 27


@lru_cache(maxsize=500000)
def _sh(hand: tuple[int, ...], n_melds: int) -> int:
    return shanten_with_melds(list(hand), n_melds)


@lru_cache(maxsize=1000000)
def _is_win_c(hand: tuple[int, ...]) -> bool:
    return is_win(list(hand))


@lru_cache(maxsize=500000)
def _tiles_m(hand: tuple[int, ...], n_melds: int) -> tuple[int, tuple[int, ...]]:
    """(向听, 有效张): 听牌时为听口(is_win 对任意 3n+2 暗牌正确), 否则为降向听进张。"""
    s = _sh(hand, n_melds)
    tiles = []
    for t in range(28):
        if hand[t] >= 4:
            continue
        h1 = _add(hand, t, 1)
        if s == 0:
            if _is_win_c(h1):
                tiles.append(t)
        elif _sh(h1, n_melds) < s:
            tiles.append(t)
    return s, tuple(tiles)


def _ukeire_m(hand: tuple[int, ...], n_melds: int, unseen: tuple[int, ...]) -> int:
    _, tiles = _tiles_m(hand, n_melds)
    return sum(unseen[t] for t in tiles)


@lru_cache(maxsize=300000)
def _second_step_m(hand: tuple[int, ...], n_melds: int,
                   unseen: tuple[int, ...]) -> float:
    """v10 _second_step_value 的副露感知版。"""
    total = sum(unseen)
    if total <= 0:
        return 0.0
    base_s = _sh(hand, n_melds)
    v = 0.0
    for draw, n in enumerate(unseen):
        if n <= 0:
            continue
        h1 = _add(hand, draw, 1)
        if base_s == 0 and _is_win_c(h1):
            v += n / total * 50.0
            continue
        best_s, best_u = 99, 0
        for disc, cnt in enumerate(h1):
            if cnt <= 0:
                continue
            h = _add(h1, disc, -1)
            s = _sh(h, n_melds)
            u = _ukeire_m(h, n_melds, unseen)
            if s < best_s or (s == best_s and u > best_u):
                best_s, best_u = s, u
        v += n / total * (20.0 * max(0, base_s - best_s) + 0.15 * best_u)
    return v


class Bot(V10Bot):
    def choose_discard(self) -> int:
        p = self.game.players[self.seat]
        n_melds = len(p.melds)
        if n_melds == 0:
            return super().choose_discard()
        counts = tuple(p.hand_counts)
        unseen = self._unseen_counts()
        penged = self._penged_by_others()
        eg = self._endgame_factor()
        opts = [(t, _sh(_add(counts, t, -1), n_melds))
                for t, c in enumerate(counts) if c > 0]
        if not opts:
            return p.hand[-1]
        min_sh = min(s for _, s in opts)
        best_t, best_score = None, -1e18
        for t, s in opts:
            h = _add(counts, t, -1)
            if s > min_sh:
                score = -10.0 * self.shanten_weight - self.shanten_weight * s
            else:
                u = _ukeire_m(h, n_melds, unseen)
                cont = _second_step_m(h, n_melds, unseen) \
                    if s <= self.cont_max_shanten else 0.0
                score = self.ukeire_weight * u + self.cont_weight * cont
            if t != RED:
                if t in penged:
                    risk = 1.0
                else:
                    risk = {3: 0.4, 2: 0.2, 1: 0.05, 0: 0.0}.get(unseen[t], 0.4)
                score -= self.risk_weight * (1.0 + 1.5 * eg) * risk
            if score > best_score:
                best_score, best_t = score, t
        return best_t

    def decide_peng(self, tile: int) -> bool:
        p = self.game.players[self.seat]
        n = len(p.melds)
        counts = tuple(p.hand_counts)
        before = _sh(counts, n)
        c11 = _add(_add(counts, tile, -1), tile, -1)
        after = min(_sh(_add(c11, d, -1), n + 1)
                    for d, cnt in enumerate(c11) if cnt > 0)
        if after < before:
            return before != 0 or after == 0
        return False

    def decide_gang(self, tile: int, kind: str) -> bool:
        p = self.game.players[self.seat]
        n = len(p.melds)
        counts = tuple(p.hand_counts)
        before = _sh(counts, n)
        c = list(counts)
        if kind == "ming":
            c[tile] -= 3
            n2 = n + 1
        elif kind == "an":
            c[tile] -= 4
            n2 = n + 1
        else:  # bu: 碰转杠, 副露数不变
            c[tile] -= 1
            n2 = n
        after = _sh(tuple(c), n2)
        return not (before == 0 and after > 0)
