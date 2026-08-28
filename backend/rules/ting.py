"""安康159 - 听牌/进张分析

给定手牌计数(13张,未摸牌状态),枚举所有可能的进张,
判断摸到每张牌后是否能胡(听牌判断),以及打出每张牌后的听口。
"""

from .tiles import TILE_COUNT, tile_rank, is_159
from .win import is_win, shanten

RED = 27


def waiting_tiles(counts13: list[int]) -> list[int]:
    """13张手牌,返回所有能胡的进张列表(听口)"""
    waits = []
    for t in range(TILE_COUNT):
        if counts13[t] >= 4:
            continue
        counts13[t] += 1
        if is_win(counts13):
            waits.append(t)
        counts13[t] -= 1
    return waits


def is_ting(counts13: list[int]) -> bool:
    return len(waiting_tiles(counts13)) > 0


def useful_draws(counts13: list[int]) -> dict[int, int]:
    """返回每张进张能将向听数降到多少(只列能降低向听数的进张)"""
    base = shanten(counts13)
    result = {}
    for t in range(TILE_COUNT):
        if counts13[t] >= 4:
            continue
        counts13[t] += 1
        s = shanten(counts13)
        counts13[t] -= 1
        if s < base:
            result[t] = s
    return result


def discard_options(counts14: list[int]) -> list[dict]:
    """14张(摸牌后),枚举打出每张牌后的听牌情况。
    返回 [{'tile': t, 'shanten': s, 'waits': [...], 'wait_count': n}]"""
    options = []
    for t in range(TILE_COUNT):
        if counts14[t] <= 0:
            continue
        counts14[t] -= 1
        s = shanten(counts14)
        waits = waiting_tiles(counts14) if s == 0 else []
        # 剩余可摸的进张数(粗略: 4 - 手里已有)
        wait_count = int(sum(4 - counts14[w] for w in waits))
        options.append({
            "tile": t,
            "shanten": s,
            "waits": waits,
            "wait_count": wait_count,
        })
        counts14[t] += 1
    return options
