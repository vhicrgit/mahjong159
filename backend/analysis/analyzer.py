"""安康159 - 实时牌局分析器(从玩家视角, 仅公开信息)

功能:
1. 当前手牌向听数、听口
2. 打出每张牌后的: 向听数变化、听口宽度、放杠风险、综合建议
3. 当前胡牌的预期159收益(基于剩余1/5/9估计)
4. 对手危险度粗估(是否接近听牌)

所有计算为确定性算法 + 简单统计, CPU 毫秒级。
"""

from ..rules.tiles import TILE_COUNT, tile_suit, tile_rank, is_159, tile_short
from ..rules.win import shanten_with_melds
from ..rules.ting import waiting_tiles

RED = 27
TOTAL_PER_TILE = 4
N_159_TOTAL = 36  # 1/5/9 x 3花色 x 4张


class Analyzer:
    """基于公开信息的实时分析器"""

    def __init__(self, game, seat: int):
        self.game = game
        self.seat = seat

    # ---------- 信息汇总 ----------
    def _visible_counts(self) -> list[int]:
        """我可见的所有牌计数: 自己手牌 + 所有家弃牌 + 所有家副露"""
        c = [0] * TILE_COUNT
        for p in self.game.players:
            if p.seat == self.seat:
                for t in p.hand:
                    c[t] += 1
            for t in p.discards:
                c[t] += 1
            for m in p.melds:
                n = 3 if m["type"] == "peng" else 4
                c[m["tile"]] += n
        return c

    def _remaining_counts(self) -> list[int]:
        """未出现的牌(在别人手牌或牌堆里)"""
        vis = self._visible_counts()
        return [TOTAL_PER_TILE - vis[t] for t in range(TILE_COUNT)]

    # ---------- 放杠风险 ----------
    def gang_risk(self, tile: int) -> float:
        """打出 tile 被杠的风险 0~1(启发式)

        依据:
        - 若某家已碰此牌(有刻子), 打第4张必被明杠 -> 1.0
        - 否则按"未出现的该牌数量"估计: 未出现3张则可能有家捏着刻子, 风险高
        - 红中不能杠, 风险0
        """
        if tile == RED:
            return 0.0
        # 有人已碰此牌?
        for p in self.game.players:
            if p.seat == self.seat:
                continue
            for m in p.melds:
                if m["tile"] == tile and m["type"] == "peng":
                    return 1.0
        remain = self._remaining_counts()[tile]
        # 剩余3张(可能成刻): 风险高; 2张: 中; <=1张: 低
        if remain >= 3:
            base = 0.55
        elif remain == 2:
            base = 0.25
        elif remain == 1:
            base = 0.06
        else:
            base = 0.0
        # 牌局越后期, 别人成刻概率越高, 风险上调
        progress = 1 - self.game.wall_remaining() / 60.0
        progress = max(0.0, min(1.0, progress))
        return round(base * (0.7 + 0.3 * progress), 3)

    # ---------- 159 收益预估 ----------
    def expected_fan159(self) -> float:
        """当前胡牌的预期159数 E[n] = 6 * K/N
        K = 牌堆中估计的159数, N = 牌堆剩余
        未知159总数(在别人手牌+牌堆)按比例摊到牌堆。
        """
        vis = self._visible_counts()
        seen_159 = sum(vis[t] for t in range(27) if is_159(t))
        unseen_159 = N_159_TOTAL - seen_159
        wall_n = self.game.wall_remaining()
        if wall_n <= 0:
            return 0.0
        # 未知牌总数 = 别人手牌(39左右) + 牌堆
        others_hand = sum(len(p.hand) for p in self.game.players
                          if p.seat != self.seat)
        unknown_total = others_hand + wall_n
        if unknown_total <= 0:
            return 0.0
        k_est = unseen_159 * wall_n / unknown_total
        return round(6.0 * k_est / wall_n, 2)

    def expected_score_if_win(self) -> float:
        """当前胡牌的期望收益 = (E[n]+1) * 3 家"""
        return round((self.expected_fan159() + 1) * 3, 2)

    # ---------- 对手状态粗估 ----------
    def opponent_threat(self, opp_seat: int) -> dict:
        """对手危险度启发式评估(0~1), 越高越可能接近听牌"""
        p = self.game.players[opp_seat]
        discards = p.discards
        score = 0.0
        # 副露越多越接近听牌
        score += 0.15 * len(p.melds)
        # 后期出牌: 连续打出中张(4-6)说明牌型整齐
        mid_discards = sum(1 for t in discards[-6:]
                           if t < 27 and 3 <= tile_rank(t) <= 7)
        score += 0.08 * mid_discards
        # 牌局进度基础分
        progress = 1 - self.game.wall_remaining() / 60.0
        score += 0.3 * max(0.0, progress)
        return {"seat": opp_seat, "threat": round(min(score, 1.0), 2)}

    # ---------- 主分析 ----------
    def analyze_hand(self) -> dict:
        """分析我当前手牌(13张等价状态, 未摸牌)。副露感知。"""
        p = self.game.players[self.seat]
        counts = p.hand_counts
        n_melds = len(p.melds)
        s = shanten_with_melds(counts, n_melds)
        result = {
            "shanten": s,
            "is_ting": s == 0,
            "waits": [],
            "wait_count": 0,
            "expected_fan159": self.expected_fan159(),
            "expected_score_if_win": self.expected_score_if_win(),
            "opponents": [self.opponent_threat(o.seat)
                          for o in self.game.players if o.seat != self.seat],
        }
        if s == 0:
            waits = waiting_tiles(counts)  # is_win 对副露手(3n+1暗牌)同样正确
            rem = self._remaining_counts()
            result["waits"] = [
                {"tile": w, "name": tile_short(w), "remain": rem[w]}
                for w in waits
            ]
            result["wait_count"] = sum(rem[w] for w in waits)
        return result

    def analyze_discards(self) -> list[dict]:
        """14张等价状态: 分析打出每张牌的后果, 给出综合建议排序。副露感知。"""
        p = self.game.players[self.seat]
        counts = p.hand_counts
        n_melds = len(p.melds)
        rem = self._remaining_counts()
        out = []
        for t in range(TILE_COUNT):
            if counts[t] <= 0:
                continue
            counts[t] -= 1
            s = shanten_with_melds(counts, n_melds)
            waits = waiting_tiles(counts) if s == 0 else []
            counts[t] += 1
            risk = self.gang_risk(t)
            # 剩余进张(用未出现数估计, 更准确)
            wait_remains = sum(rem[w] for w in waits)
            # 综合分: 向听数小优先, 进张多优先, 风险低优先
            score = -100 * s + 3 * wait_remains - 30 * risk
            out.append({
                "tile": t,
                "name": tile_short(t),
                "shanten": s,
                "waits": [{"tile": w, "name": tile_short(w),
                           "remain": rem[w]} for w in waits],
                "wait_remain": wait_remains,
                "gang_risk": risk,
                "score": round(score, 1),
            })
        out.sort(key=lambda x: -x["score"])
        return out
