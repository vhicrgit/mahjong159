"""安康159 - Oracle Bot: 完美信息上界测量

作弊 Bot: 直接读 game.wall, 知道自己未来会摸到哪些牌,
用 beam search 找"最快胡牌"的出牌路径。

用途: 测量本游戏的技术天花板。
  若 Oracle(知道自己所有未来摸牌) 对 3 个规则Bot 的胜率只有 X%,
  则任何不作弊的 AI 胜率 <= X%（因为 Oracle 拥有严格更多信息）。
  这决定了 ">50% 胜率" 目标是否在数学上可达。

注意: 摸牌位置按"无人碰杠"推算(碰杠会改变顺序), 每次决策重新
从当前 wall 推算, 因此中断可自我校正。
"""

from ..rules.win import is_win, shanten
from ..rules.ting import discard_options

RED = 27
BEAM_WIDTH = 12
WIN_DISCOUNT = 0.85    # 效用 = DISCOUNT^(到胡牌所需摸牌数)


def _no_win_utility(horizon: int, shanten_left: int) -> float:
    """未在视野内胡牌: 按"再过 2×剩余向听 步才胡"悲观外推。
    与胡牌效用同处 (0,1] 指数尺度 —— 这对 PIMC 跨世界平均至关重要,
    否则罕见幸运世界会盖过所有世界的向听进展(彩票效应, 实测崩到 19%)。"""
    return WIN_DISCOUNT ** (horizon + 2 * max(0, shanten_left) + 1)


def search_first_discard_detail(counts14: list[int],
                                future_draws: list[int],
                                beam: int = BEAM_WIDTH,
                                candidates=None) -> dict:
    """beam search 每个首出牌的结果明细。

    返回 {首出牌 tile: (win_depth 或 None, 视野内可达的最小向听)}。
    win_depth = 需要几次自己的摸牌才能自摸胡(0 = 下一次摸牌就胡)。
    PIMC 用胜率占比聚合(低方差), Oracle 用折扣效用 argmax。
    """
    horizon = len(future_draws)
    detail = {}
    won = set()
    # 初始层: 每个合法出牌各作为一个分支根
    beam_nodes = []
    for t in range(28):
        if counts14[t] <= 0:
            continue
        if candidates is not None and t not in candidates:
            continue
        c = list(counts14)
        c[t] -= 1
        s = shanten(c)
        beam_nodes.append((tuple(c), t, s))
        detail[t] = (None, s)
    if not beam_nodes:
        return detail

    for depth, draw in enumerate(future_draws):
        next_nodes = []
        seen = set()
        for hand13, first_t, _ in beam_nodes:
            if first_t in won:
                continue  # 该首出牌已找到更早的胡牌路径
            c = list(hand13)
            c[draw] += 1
            if is_win(c):
                won.add(first_t)
                detail[first_t] = (depth, -1)
                continue
            for t in range(28):
                if c[t] <= 0:
                    continue
                c[t] -= 1
                key = (tuple(c), first_t)
                if key not in seen:
                    seen.add(key)
                    next_nodes.append((tuple(c), first_t, shanten(c)))
                c[t] += 1
        if not next_nodes:
            break
        # 分组保留: 每个首出牌各留 beam 个最优后继, 防止单分支垄断
        by_first = {}
        for item in next_nodes:
            by_first.setdefault(item[1], []).append(item)
        beam_nodes = []
        for ft, items in by_first.items():
            items.sort(key=lambda x: x[2])
            beam_nodes.extend(items[:beam])
            if ft not in won:
                best_s = items[0][2]
                prev = detail.get(ft, (None, 99))
                if prev[0] is None and best_s < prev[1]:
                    detail[ft] = (None, best_s)
    return detail


def search_first_discard_scores(counts14: list[int],
                                future_draws: list[int],
                                beam: int = BEAM_WIDTH) -> dict:
    """折扣效用版(Oracle 用): {tile: utility ∈ (0,1]}"""
    horizon = len(future_draws)
    detail = search_first_discard_detail(counts14, future_draws, beam)
    out = {}
    for t, (wd, sh) in detail.items():
        out[t] = (WIN_DISCOUNT ** wd) if wd is not None \
            else _no_win_utility(horizon, sh)
    return out


class Bot:
    """Oracle: 已知自己未来摸牌序列, beam search 最快胡牌"""

    def __init__(self, game, seat: int, beam: int = BEAM_WIDTH):
        self.game = game
        self.seat = seat
        self.beam = beam

    def _my_future_draws(self, max_draws: int = 14) -> list[int]:
        """推算自己接下来会摸到的牌(假设无碰杠打断)。

        出牌后 (last_discarder+1)%4 摸 wall[0]。当前是我出牌 →
        下一个摸牌者是 (seat+1)%4, 摸 wall[0];
        我下次摸牌在 wall[3], 之后每隔 4 张。
        """
        wall = self.game.wall
        draws = []
        idx = 3  # 我的下一次摸牌位置
        while idx < len(wall) - 6 and len(draws) < max_draws:
            draws.append(wall[idx])
            idx += 4
        return draws

    def choose_discard(self) -> int:
        p = self.game.players[self.seat]
        counts14 = list(p.hand_counts)
        opts = discard_options(counts14)
        if not opts:
            return p.hand[-1]

        future = self._my_future_draws()
        scores = search_first_discard_scores(counts14, future, self.beam)
        if not scores:
            return min(opts, key=lambda o: o["shanten"])["tile"]
        return max(scores.items(), key=lambda kv: kv[1])[0]

    # ---------- 碰杠: 用 v2 同款逻辑 ----------

    def decide_peng(self, tile: int) -> bool:
        p = self.game.players[self.seat]
        counts = p.hand_counts
        before = shanten(counts)
        c2 = list(counts)
        c2[tile] -= 2
        after = shanten(c2)
        if after < before:
            if before == 0:
                return after == 0
            return True
        return False

    def decide_gang(self, tile: int, kind: str) -> bool:
        p = self.game.players[self.seat]
        counts = p.hand_counts
        s_before = shanten(counts)
        c2 = list(counts)
        if kind == "ming":
            c2[tile] -= 3
        elif kind == "an":
            c2[tile] -= 4
        else:
            c2[tile] -= 1
        s_after = shanten(c2)
        if s_before == 0 and s_after > 0:
            return False
        return True
