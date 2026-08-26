"""安康159 - 对手阵容配置

三个 AI 对手(座位 1/2/3), 各自使用不同强度的 Bot:
  座位1 下家: 菜鸟  - bot_v1  (纯解析: 向听+进张+放杠惩罚)
  座位2 对家: 老鸟  - bot_v4  (解析骨架 + 同向听内 PIMC 搜索精修)
  座位3 上家: 挂哥  - oracle  (作弊: 直接读牌堆, beam search 最快胡牌)
"""

from . import bot_v1, bot_v4, bot_oracle

# seat -> (显示名, 模块, 说明)
ROSTER = {
    1: ("菜鸟", bot_v1, "规则Bot v1: 牌效优先"),
    2: ("老鸟", bot_v4, "规则Bot v4: 解析+搜索精修"),
    3: ("挂哥", bot_oracle, "作弊Bot: 可见牌堆"),
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
    for seat, (_, mod, _) in ROSTER.items():
        if seat == human_seat:
            continue
        bots[seat] = mod.Bot(game, seat)
    return bots
