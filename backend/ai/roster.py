"""安康159 - 对手阵容配置

三个 AI 对手(座位 1/2/3), 各自使用不同强度的 Bot:
  座位1 下家: 菜鸟  - bot_v1   (纯解析: 向听+进张+放杠惩罚)
  座位2 对家: 老鸟  - bot_v31  (v10 两步推演 + 副露感知向听, 会正常碰杠)
  座位3 上家: 挂哥  - bot_cheat(作弊: 可见牌墙, 不看对手手牌)

注意 bot_v4 -> bot_v31 的升级原因: shanten() 硬编码 13 张手牌公式, 副露后
暗牌变短会把向听高估约 2*副露数, 导致 v1~v30 的 decide_peng/decide_gang
几乎永远判定"碰杠必然变差"而从不鸣牌。v31 用 shanten_with_melds 修正。

挂哥使用 cheat_wall 档(wall_lookahead=32, see_opponents=False), 保持与原
bot_oracle 相同的"只能偷看牌墙"人设; 若要更强可改用全信息档
(wall_lookahead=-1, see_opponents=True, rollout=True)。
"""

from . import bot_v1, bot_v31, bot_cheat

# seat -> (显示名, 模块, 说明, 构造参数)
ROSTER = {
    1: ("菜鸟", bot_v1, "规则Bot v1: 牌效优先", {}),
    2: ("老鸟", bot_v31, "规则Bot v31: 两步推演+副露感知向听", {}),
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
