"""Rule bot v29: v10 skeleton + near-tenpai analytic win-prob recursion.

- shanten>=2: 与 v10 完全一致(最小向听优先, ukeire + second-step continuation)。
- min_sh<=1 的最小向听候选间: 用"贪心策略下 P(未来 k 次自己摸牌内自摸)"排序,
  k 自适应剩余牌墙(k=min(KCAP, wall//4)), v10 启发分做同分 tie-break。

winp 递归(确定性, 非采样):
  - 听牌: P = 1-(1-W/T)^k, W=活听张数(冻结换口)
  - 向听 s>=1: 有效进张 d 以 n_d/T 概率转移到贪心打牌后的更低向听手;
    无效摸牌视为状态不变(不建模换口/重塑); k < s+1 时 P=0 剪枝。
该式同时编码"进张数"与"到达的听牌质量"(用户提的启发式偏置), 每步打牌
决策由 v10 贪心(向听↓, 同则 ukeire↑)引导。
"""

import os
from functools import lru_cache

from ..rules.win import is_win, shanten_cached
from ..rules.ting import discard_options
from .bot_v10 import Bot as V10Bot, _add, _ukeire, _second_step_value

RED = 27


@lru_cache(maxsize=1000000)
def _is_win14(hand14: tuple[int, ...]) -> bool:
    return is_win(list(hand14))


@lru_cache(maxsize=1000000)
def _tiles13(hand13: tuple[int, ...]) -> tuple[int, tuple[int, ...]]:
    """(向听, 有效张列表): 听牌时为听口, 否则为降向听的进张。只按手牌缓存。"""
    s = shanten_cached(hand13)
    tiles = []
    for t in range(28):
        if hand13[t] >= 4:
            continue
        h14 = _add(hand13, t, 1)
        if s == 0:
            if _is_win14(h14):
                tiles.append(t)
        elif shanten_cached(h14) < s:
            tiles.append(t)
    return s, tuple(tiles)


@lru_cache(maxsize=1000000)
def _disc_shantens(hand14: tuple[int, ...]) -> tuple:
    """每个可行弃牌的 (弃牌, 打后13张, 向听)。只按手牌缓存, 不算有效张。"""
    out = []
    for disc, cnt in enumerate(hand14):
        if cnt <= 0:
            continue
        h = _add(hand14, disc, -1)
        out.append((disc, h, shanten_cached(h)))
    return tuple(out)


def _greedy_next(hand14: tuple[int, ...], unseen: tuple[int, ...]) -> tuple[int, ...]:
    entries = _disc_shantens(hand14)
    min_s = min(e[2] for e in entries)
    best_key, best_h = None, None
    for disc, h, s in entries:
        if s > min_s:
            continue
        _, tiles = _tiles13(h)
        u = sum(unseen[t] for t in tiles)
        key = (-u, disc)
        if best_key is None or key < best_key:
            best_key, best_h = key, h
    return best_h


@lru_cache(maxsize=300000)
def _succ(hand13: tuple[int, ...], unseen: tuple[int, ...]) -> tuple:
    """有效进张的 (张, 未见数, 贪心打牌后的后继手牌), 跨 k 复用。"""
    _, tiles = _tiles13(hand13)
    return tuple((d, unseen[d], _greedy_next(_add(hand13, d, 1), unseen))
                 for d in tiles if unseen[d] > 0)


@lru_cache(maxsize=300000)
def _win_prob(hand13: tuple[int, ...], unseen: tuple[int, ...], k: int) -> float:
    total = sum(unseen)
    if total <= 0 or k <= 0:
        return 0.0
    s, tiles = _tiles13(hand13)
    if k < s + 1:
        return 0.0
    if s == 0:
        w = sum(unseen[t] for t in tiles)
        if w <= 0:
            return 0.0
        return 1.0 - (1.0 - w / total) ** k
    p_step = 0.0
    p_move = 0.0
    for d, n, nxt in _succ(hand13, unseen):
        pd = n / total
        p_move += pd
        p_step += pd * _win_prob(nxt, unseen, k - 1)
    return p_step + (1.0 - p_move) * _win_prob(hand13, unseen, k - 1)


class Bot(V10Bot):
    def __init__(self, game, seat: int):
        super().__init__(game, seat)
        self.k_cap = int(os.environ.get("V29_KCAP", 6))
        self.top_m = int(os.environ.get("V29_TOPM", 6))
        self.dp_max_sh = int(os.environ.get("V29_DP_MAX_SH", 1))
        self.round_digits = int(os.environ.get("V29_ROUND", 3))

    def choose_discard(self) -> int:
        p = self.game.players[self.seat]
        counts14 = tuple(p.hand_counts)
        opts = discard_options(list(counts14))
        if not opts:
            return p.hand[-1]
        unseen = self._unseen_counts()
        penged = self._penged_by_others()
        eg = self._endgame_factor()
        min_sh = min(o["shanten"] for o in opts)
        scores: dict[int, float] = {}
        for o in opts:
            t = o["tile"]
            h = _add(counts14, t, -1)
            s = o["shanten"]
            if s > min_sh:
                score = -10.0 * self.shanten_weight - self.shanten_weight * s
            else:
                u = _ukeire(h, unseen)
                cont = _second_step_value(h, unseen) if s <= self.cont_max_shanten else 0.0
                score = self.ukeire_weight * u + self.cont_weight * cont
            if t != RED:
                if t in penged:
                    risk = 1.0
                else:
                    risk = {3: 0.4, 2: 0.2, 1: 0.05, 0: 0.0}.get(unseen[t], 0.4)
                score -= self.risk_weight * (1.0 + 1.5 * eg) * risk
            scores[t] = score
        v10_pick = max(scores.items(), key=lambda kv: kv[1])[0]
        if min_sh > self.dp_max_sh:
            return v10_pick
        cands = sorted((o["tile"] for o in opts if o["shanten"] == min_sh),
                       key=lambda t: -scores[t])[:self.top_m]
        if len(cands) <= 1:
            return v10_pick
        k = min(self.k_cap, max(1, self.game.wall_remaining() // 4))
        best_t, best_key = None, None
        for t in cands:
            h = _add(counts14, t, -1)
            winp = _win_prob(h, unseen, k)
            key = (round(winp, self.round_digits), scores[t])
            if best_key is None or key > best_key:
                best_key, best_t = key, t
        return best_t
