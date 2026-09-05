"""安康159 - 规则Bot v32: v31 + 听牌后的选择性碰牌。

来源: 2026-09-05 外部研究(交接包 `bot_calls/call_candidates.py` 的
`ReadyPengBot`)。他们的证据: 512 种子开发批 +0.0425 分/局, 4096 种子独立批
+0.0168 分/局(种子聚类 95%CI [+0.0023, +0.0314]), 胡牌率 +0.128 个百分点
(CI [+0.018, +0.238]); 16384 局里只有 160 局的结果被改变, 所以这是"小幅、
可叠加"的增强, 不是档次跃升。本仓库用全新种子段在 CRN 配对台架上复验:
4096 种子 **+0.0215 ± 0.0077 分/局(t=+2.80)**, 512 种子那批测不出来(SE 0.0146)。
详见 docs/rule_bot_and_analyzer_review.md 第二节。

v31 的 `decide_peng` 只接受"碰后向听严格下降"。已经听牌时碰完再弃一张通常
仍是 0 向听, 于是即使听口大幅变宽也会被拒 —— 例如 seed 766107 座位2, 上家打出
6条: 碰6条弃5条后暗手是 678条/789饼/567万 + 单红中, 配已碰的一副刻子, 红中能与
任意牌成将, 活听从 21 张扩到 94 张(7 种 -> 28 种)。

三个保守条件(全部满足才碰):

1. 当前已听牌(向听 0), 且原本的判据拒碰;
2. 存在某个碰后弃牌, 使新的**活听口**(仍有余张的听张)严格包含旧活听口 ——
   原听口一张都不许丢;
3. 名义剩余摸牌次数不减少。碰完必须立刻弃牌, 不会白摸一张, 所以"碰一定抢到
   自己下一摸"是错的: 不碰时下一摸距今 dist 张, 碰后通常变成 4 张一轮, 可能
   推迟。按剩余牌墙(扣掉黄庄保留的 6 张)与座位距离估算, 只接受不减少的。

条件 3 假设此后没有别的碰杠、也不提前终局, 因此是保守启发式, 不是严格支配
关系。全程只用自己手牌、公开弃牌/副露、最近出牌者与剩余牌墙长度, 无私有信息。

`ready_peng_plan` 抽成模块级函数是为了可复用、可单独测量。NN 侧
(`backend/rl/net_bot.py`)实测**不需要**它: 碰杠走分析器 E, 而 E 没有 v31 这个
缺口 —— 1500 局 1047 次碰机会里该判据命中 17 次, E 全部已经接受碰, 净增量 0
(1500 种子配对 A/B 两臂 6000 局逐局相同)。所以只用在规则 Bot 上。

同目录的 `DeadWaitGangBot`(听口变死就拒杠)在原研究里没有增益证据, 不接入:
杠分为正时牺牲部分听口是有价值的(补6饼把活听 24 压到 11, 但配对推演里杠的
场均分 +8.313 高于不杠的 +7.357)。
"""

from ..native import native
from .bot_native import NativeV31

HUANGZHUANG_RESERVE = 6      # 牌堆剩余<=6 黄庄, 见 engine.check_huangzhuang_before_draw


def unseen_counts(game, seat: int) -> list[int]:
    """未现张数: 4 - (自己手牌 + 所有弃牌 + 所有副露)。与 v10 同口径。"""
    vis = [0] * 28
    for p in game.players:
        for t in p.discards:
            vis[t] += 1
        for m in p.melds:
            vis[m["tile"]] += 3 if m["type"] == "peng" else 4
    for t, n in enumerate(game.players[seat].hand_counts):
        vis[t] += n
    return [max(0, 4 - v) for v in vis]


def live_waits(hand, unseen) -> set[int]:
    """活听口: 还摸得到的听张。逐张试判胡, 不用"向听=0 就一定有听口"的推断
    (四副露单钓那类边界上, 向听推断会把死听当活听)。"""
    out = set()
    h = list(hand)
    for t, u in enumerate(unseen):
        if u <= 0 or h[t] >= 4:
            continue
        h[t] += 1
        if native.is_win(h):
            out.add(t)
        h[t] -= 1
    return out


def ready_peng_plan(game, seat: int, tile: int):
    """听牌后的选择性碰牌判据。返回 (碰并弃牌后的暗牌计数, 计划弃牌) 或 None。

    返回 None 表示不该碰。调用方在碰成功后应把弃牌锁定为计划的那张 ——
    听口变宽的结论只对这张弃牌成立, 换一张就未必。
    """
    hand = game.players[seat].hand_counts
    if hand[tile] < 2 or native.shanten(hand) != 0:
        return None
    left = max(0, game.wall_remaining() - HUANGZHUANG_RESERVE)
    dist = (seat - game.last_discarder) % 4
    draws_no_call = max(0, (left - dist) // 4 + 1)
    draws_after_call = left // 4
    if draws_after_call <= 0 or draws_after_call < draws_no_call:
        return None
    unseen = unseen_counts(game, seat)
    old = live_waits(hand, unseen)
    if not old:
        return None                        # 死听不值得为它碰
    after = list(hand)
    after[tile] -= 2
    best = None                            # ((活听总张数, -弃牌), 弃牌)
    for d, n in enumerate(after):
        if n <= 0:
            continue
        after[d] -= 1
        new = live_waits(after, unseen)
        if old < new:                      # 严格包含: 原听口一张不丢
            key = (sum(unseen[t] for t in new), -d)
            if best is None or key > best[0]:
                best = (key, d)
        after[d] += 1
    if best is None:
        return None
    after[best[1]] -= 1
    return tuple(after), best[1]


class Bot(NativeV31):
    """v31 的出牌(native C 路径) + 听牌后的选择性碰牌。"""

    BOT_ID = 32

    def __init__(self, game, seat: int):
        super().__init__(game, seat)
        self._planned = None     # (碰并弃牌后的暗牌计数, 计划弃牌)

    def decide_peng(self, tile: int) -> bool:
        if super().decide_peng(tile):
            return True
        plan = ready_peng_plan(self.game, self.seat, tile)
        if plan is None:
            return False
        self._planned = plan
        return True

    def choose_discard(self) -> int:
        if self._planned is not None:
            hand, tile = self._planned
            self._planned = None
            if tuple(self.game.players[self.seat].hand_counts) == hand:
                return tile
        return super().choose_discard()
