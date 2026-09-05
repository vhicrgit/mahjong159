"""安康159 - 对手阵容配置

三个 AI 对手(座位 1/2/3), 各自使用不同强度的 Bot:
  座位1 下家: 菜鸟  - bot_v1   (纯解析: 向听+进张+放杠惩罚)
  座位2 对家: 老鸟  - bot_v32  (v31 + 听牌后选择性碰牌)
  座位3 上家: 挂哥  - bot_cheat(作弊: 可见牌墙, 不看对手手牌)

注意 bot_v4 -> bot_v31 的升级原因: shanten() 硬编码 13 张手牌公式, 副露后
暗牌变短会把向听高估约 2*副露数, 导致 v1~v30 的 decide_peng/decide_gang
几乎永远判定"碰杠必然变差"而从不鸣牌。v31 用 shanten_with_melds 修正。

v31 -> v32 的升级原因: v31 的 decide_peng 只接受"碰后向听严格下降", 已听牌时
碰完再弃一张通常仍是 0 向听, 于是即使活听口大幅变宽也拒碰。v32 只在"新活听口
严格包含旧活听口 且 名义剩余摸牌次数不减少"时才多碰这一类。4096 个全新种子的
CRN 配对(四座位轮转, 对手 3×v31n, 完整暗杠/补杠协议, 主指标真实净分):
**+0.0215 ± 0.0077 分/局, t=+2.80, 95%CI [+0.0065, +0.0366]**
(详见 docs/rule_bot_and_analyzer_review.md 第二节)。

挂哥使用 cheat_wall 档(wall_lookahead=32, see_opponents=False), 保持与原
bot_oracle 相同的"只能偷看牌墙"人设; 若要更强可改用全信息档
(wall_lookahead=-1, see_opponents=True, rollout=True)。
"""

from . import bot_v1, bot_v32, bot_cheat

# seat -> (显示名, 模块, 说明, 构造参数)
ROSTER = {
    1: ("菜鸟", bot_v1, "规则Bot v1: 牌效优先", {}),
    2: ("老鸟", bot_v32, "规则Bot v32: v31+听牌后选择性碰牌", {}),
    3: ("挂哥", bot_cheat, "作弊Bot: 可见牌墙", {
        "wall_lookahead": 32,
        "see_opponents": False,
        "beam": 12,
        "rollout": False,
    }),
}

HUMAN_NAME = "我"


def seat_name(seat: int) -> str:
    if seat in ROSTER:
        return ROSTER[seat][0]
    return HUMAN_NAME


def seat_desc(seat: int) -> str:
    if seat in ROSTER:
        return ROSTER[seat][2]
    return "玩家"


def build_bots(game, human_seat: int) -> dict:
    """为每个 AI 座位创建对应的 Bot 实例"""
    bots = {}
    for seat, (_, mod, _, kwargs) in ROSTER.items():
        if seat == human_seat:
            continue
        bots[seat] = mod.Bot(game, seat, **kwargs)
    return bots


# ---------- 可选 AI 档位(开局界面"对手选择"用) ----------
# kind -> (显示名, 说明)
KIND_INFO = {
    "v1":         ("菜鸟", "规则Bot v1: 牌效优先"),
    "v10":        ("中鸟", "规则Bot v10: 广义进张+两步推演"),
    "v31":        ("老鸟", "规则Bot v31: v10+副露感知碰牌"),
    "v32":        ("老鸟+", "规则Bot v32: v31+听牌后选择性碰牌(听口严格变宽才碰)"),
    "scholar":    ("学者", "牌型价值分析器: 每手算期望胡牌巡数"),
    "acnn":       ("AC学者", "神经网络: 分析器E值 DAgger 模仿(软标签) + E判据碰杠"),
    "target":     ("目标", "目标路线概率 Bot"),
    "cheat_wall": ("挂哥", "作弊: 可见牌墙"),
    "cheat_opp":  ("挂王", "作弊: 牌墙+对手手牌"),
    "cheat_full": ("神挂", "作弊: 全信息+rollout搜索"),
}


def kind_name(kind: str) -> str:
    return KIND_INFO.get(kind, (kind,))[0]


def make_bot(kind: str | None, game, seat: int, param: int = 0):
    """按档位名构造 Bot。kind=None/"default" 时用该座位的阵容默认。"""
    if kind in (None, "", "default"):
        info = ROSTER.get(seat)
        if info is None:
            kind = "v31"
        else:
            return info[1].Bot(game, seat, **info[3])
    if kind in ("v31", "normal"):
        from .bot_v31 import Bot as B
        return B(game, seat)
    if kind == "v32":
        from .bot_v32 import Bot as B
        return B(game, seat)
    if kind == "v10":
        from .bot_v10 import Bot as B
        return B(game, seat)
    if kind == "v1":
        from .bot_v1 import Bot as B
        return B(game, seat)
    if kind == "scholar":
        from .bot_hv import Bot as B
        return B(game, seat)
    if kind == "acnn":
        import os
        from ..rl.net_bot import NetBot
        path = os.environ.get("ACNN_MODEL",
                              "models/acnn_latest_best.pt")
        if not os.path.exists(path):
            path = "models/acnn_latest.pt"
        if not os.path.exists(path):
            # 模型文件不入库(~5MB); 没训练产物时退回老鸟, 别让游戏崩
            from .bot_v31 import Bot as B
            return B(game, seat)
        return NetBot(game, seat, path)
    if kind == "target":
        from .bot_target import Bot as B
        return B(game, seat)
    if kind == "cheat_wall":
        from .bot_cheat import Bot as B
        return B(game, seat, wall_lookahead=param or 32,
                 see_opponents=False, beam=12, rollout=False)
    if kind == "cheat_opp":
        from .bot_cheat import Bot as B
        return B(game, seat, wall_lookahead=param or 32,
                 see_opponents=True, beam=12, rollout=False)
    if kind == "cheat_full":
        from .bot_cheat import Bot as B
        return B(game, seat, wall_lookahead=-1, see_opponents=True,
                 beam=param or 4, rollout=True)
    from .bot_v31 import Bot as B
    return B(game, seat)
