"""安康159门清麻将 - 牌定义与基础数据结构

牌组:条(万用T表示)、饼(B)、万(W) 1~9 各4张,共108张
     红中(HZ) 4张,共112张
内部编码:tile id = 0..29
  0-8:  条 1-9
  9-17: 饼 1-9
  18-26: 万 1-9
  27: 红中
"""

TIAO, BING, WAN, HZ = range(4)

# tile id 约定
# 条1..条9 -> 0..8
# 饼1..饼9 -> 9..17
# 万1..万9 -> 18..26
# 红中 -> 27
TILE_COUNT = 28  # 不同牌种数量

SUIT_NAMES = ["条", "饼", "万"]
SUIT_SHORT = ["T", "B", "W"]


def tile_suit(t: int) -> int:
    """返回花色: 0=条 1=饼 2=万 3=红中"""
    if t < 9:
        return TIAO
    if t < 18:
        return BING
    if t < 27:
        return WAN
    return HZ


def tile_rank(t: int) -> int:
    """返回点数 1..9,红中返回 0"""
    if t >= 27:
        return 0
    return t % 9 + 1


def is_suit_tile(t: int) -> bool:
    return t < 27


def is_159(t: int) -> bool:
    """是否 1/5/9(条饼万都算)"""
    if t >= 27:
        return False
    r = t % 9
    return r in (0, 4, 8)


def tile_name(t: int) -> str:
    if t == 27:
        return "红中"
    return f"{tile_rank(t)}{SUIT_NAMES[tile_suit(t)]}"


def tile_short(t: int) -> str:
    if t == 27:
        return "HZ"
    return f"{tile_rank(t)}{SUIT_SHORT[tile_suit(t)]}"


def build_wall() -> list[int]:
    """构建一副112张牌"""
    wall = []
    for t in range(27):
        wall += [t] * 4
    wall += [27] * 4  # 红中
    return wall


def counts_from_tiles(tiles: list[int]) -> list[int]:
    """手牌列表 -> 计数向量(长度28)"""
    c = [0] * TILE_COUNT
    for t in tiles:
        c[t] += 1
    return c


def tiles_from_counts(counts: list[int]) -> list[int]:
    out = []
    for t, n in enumerate(counts):
        out += [t] * n
    return out
