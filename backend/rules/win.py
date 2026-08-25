"""安康159 - 胡牌判断与向听数(支持红中癞子)

标准胡牌型:4面子(顺子/刻子)+ 1对将,红中可当任意牌。
红中数量最多4,DFS 状态空间小,配合 memoization 性能足够。
"""

from functools import lru_cache
from .tiles import TILE_COUNT, tile_suit, tile_rank, HZ

RED = 27  # 红中 tile id


def _same_suit_seq(a, b, c):
    """三个 tile id 是否构成同花色顺子(a<b<c 已排序的 rank)"""
    sa, sb, sc = tile_suit(a), tile_suit(b), tile_suit(c)
    if sa != sb or sb != sc or sa == HZ:
        return False
    ra, rb, rc = tile_rank(a), tile_rank(b), tile_rank(c)
    return rb == ra + 1 and rc == ra + 2


def _all_melds(counts: tuple, red: int) -> bool:
    """检查 counts(张数为3的倍数)在红中辅助下能否全部组成面子(顺/刻)"""
    counts = list(counts)
    t = -1
    for i in range(27):
        if counts[i] > 0:
            t = i
            break
    if t == -1:
        return red % 3 == 0

    # 刻子
    if counts[t] >= 3:
        counts[t] -= 3
        if _all_melds_cached(tuple(counts), red):
            return True
        counts[t] += 3
    if counts[t] >= 2 and red >= 1:
        counts[t] -= 2
        if _all_melds_cached(tuple(counts), red - 1):
            return True
        counts[t] += 2
    if counts[t] >= 1 and red >= 2:
        counts[t] -= 1
        if _all_melds_cached(tuple(counts), red - 2):
            return True
        counts[t] += 1
    # 顺子
    s, r = tile_suit(t), tile_rank(t)
    if s != HZ and r <= 7:
        t1, t2 = t + 1, t + 2
        need = 0
        use = [0, 0, 0]
        if counts[t] >= 1:
            use[0] = 1
        else:
            need += 1
        if t1 < 27 and counts[t1] >= 1:
            use[1] = 1
        else:
            need += 1
        if t2 < 27 and counts[t2] >= 1:
            use[2] = 1
        else:
            need += 1
        if need <= red:
            counts[t] -= use[0]
            counts[t1] -= use[1]
            counts[t2] -= use[2]
            if _all_melds_cached(tuple(counts), red - need):
                return True
            counts[t] += use[0]
            counts[t1] += use[1]
            counts[t2] += use[2]
    return False


@lru_cache(maxsize=200000)
def _all_melds_cached(counts: tuple, red: int) -> bool:
    return _all_melds(counts, red)


@lru_cache(maxsize=500000)
def shanten_cached(tiles_counts: tuple) -> int:
    return shanten(list(tiles_counts))


def is_win(tiles_counts: list[int]) -> bool:
    """判断计数向量(张数 mod 3 == 2)是否可胡牌(含红中癞子)

    枚举将的位置(含红中凑将),剩余用 DFS 检查能否全部组成面子。
    """
    total = sum(tiles_counts)
    if total % 3 != 2:
        return False
    red = tiles_counts[RED]
    base = list(tiles_counts[:27])

    # 将 = 普通对子,可用 0/1/2 张红中凑
    for t in range(27):
        for need in (0, 1, 2):
            if need > red:
                continue
            if base[t] + need >= 2 and base[t] >= 2 - need:
                c = list(base)
                c[t] -= (2 - need)
                if _all_melds_cached(tuple(c), red - need):
                    return True
    # 将 = 两张红中
    if red >= 2:
        if _all_melds_cached(tuple(base), red - 2):
            return True
    return False


@lru_cache(maxsize=500000)
def _dfs_cached(counts: tuple, red_left: int) -> tuple:
    """模块级 DFS。原实现是 shanten 内的闭包 + lru_cache —— 每次调用
    重建闭包导致缓存永不命中, MC rollout 全冷查询。提升到模块级后
    缓存跨调用共享。"""
    counts = list(counts)
    best = (0, 0, 0)  # m, t, p
    t = -1
    for i in range(27):
        if counts[i] > 0:
            t = i
            break
    if t == -1:
        m, rem = divmod(red_left, 3)
        if rem == 2:
            return (m, 0, 1)
        return (m, 0, 0)

    # 选项1: 孤张跳过
    counts[t] -= 1
    sub = _dfs_cached(tuple(counts), red_left)
    best = _combine(best, sub)
    counts[t] += 1

    # 选项2: 对子(将)
    if counts[t] >= 2:
        counts[t] -= 2
        sub = _dfs_cached(tuple(counts), red_left)
        best = _combine(best, (sub[0], sub[1], sub[2] + 1))
        counts[t] += 2
    if counts[t] >= 1 and red_left >= 1:
        counts[t] -= 1
        sub = _dfs_cached(tuple(counts), red_left - 1)
        best = _combine(best, (sub[0], sub[1], sub[2] + 1))
        counts[t] += 1

    # 选项3: 刻子
    if counts[t] >= 3:
        counts[t] -= 3
        sub = _dfs_cached(tuple(counts), red_left)
        best = _combine(best, (sub[0] + 1, sub[1], sub[2]))
        counts[t] += 3
    if counts[t] >= 2 and red_left >= 1:
        counts[t] -= 2
        sub = _dfs_cached(tuple(counts), red_left - 1)
        best = _combine(best, (sub[0] + 1, sub[1], sub[2]))
        counts[t] += 2

    # 选项4/5: 顺子/搭子 (同原逻辑, 仅递归名替换)
    s = tile_suit(t)
    r = tile_rank(t)
    if s != HZ and r <= 7:
        t1, t2 = t + 1, t + 2
        need = 0
        use = [1, 1, 1]
        if counts[t] >= 1:
            use[0] = 1
        else:
            need += 1
        if t1 < 27 and counts[t1] >= 1:
            use[1] = 1
        else:
            need += 1
        if t2 < 27 and counts[t2] >= 1:
            use[2] = 1
        else:
            need += 1
        if need <= red_left:
            counts[t] -= use[0]
            counts[t1] -= use[1]
            counts[t2] -= use[2]
            sub = _dfs_cached(tuple(counts), red_left - need)
            best = _combine(best, (sub[0] + 1, sub[1], sub[2]))
            counts[t] += use[0]
            counts[t1] += use[1]
            counts[t2] += use[2]

        # 两面 t,t+1 (红中补)
        if t1 < 27:
            need = (0 if counts[t] >= 1 else 1) + (
                0 if counts[t1] >= 1 else 1)
            if 0 < need <= red_left:
                u0 = 1 if counts[t] >= 1 else 0
                u1 = 1 if counts[t1] >= 1 else 0
                counts[t] -= u0
                counts[t1] -= u1
                sub = _dfs_cached(tuple(counts), red_left - need)
                best = _combine(best, (sub[0], sub[1] + 1, sub[2]))
                counts[t] += u0
                counts[t1] += u1
            if counts[t] >= 1 and counts[t1] >= 1:
                counts[t] -= 1
                counts[t1] -= 1
                sub = _dfs_cached(tuple(counts), red_left)
                best = _combine(best, (sub[0], sub[1] + 1, sub[2]))
                counts[t] += 1
                counts[t1] += 1
        # 嵌张 t,t+2 (红中补)
        if t2 < 27:
            need = (0 if counts[t] >= 1 else 1) + (
                0 if counts[t2] >= 1 else 1)
            if 0 < need <= red_left:
                u0 = 1 if counts[t] >= 1 else 0
                u2 = 1 if counts[t2] >= 1 else 0
                counts[t] -= u0
                counts[t2] -= u2
                sub = _dfs_cached(tuple(counts), red_left - need)
                best = _combine(best, (sub[0], sub[1] + 1, sub[2]))
                counts[t] += u0
                counts[t2] += u2
            if counts[t] >= 1 and counts[t2] >= 1:
                counts[t] -= 1
                counts[t2] -= 1
                sub = _dfs_cached(tuple(counts), red_left)
                best = _combine(best, (sub[0], sub[1] + 1, sub[2]))
                counts[t] += 1
                counts[t2] += 1
    return best


def shanten(tiles_counts: list[int]) -> int:
    """最小向听数(0=听牌, -1=已胡)。支持红中。

    用 DFS 求最大 (2*面子 + 搭子 + 将) 收益。
    shanten = 8 - 2*m - t - p, m<=4, m+t<=4
    """
    red = tiles_counts[RED]
    counts = tuple(tiles_counts[:27])
    m, t, p = _dfs_cached(counts, red)
    m = min(m, 4)
    t = min(t, 4 - m)
    p = min(p, 1)
    s = 8 - 2 * m - t - p
    return s


def _combine(a, b):
    """取两组 (m,t,p) 中更优(向听数更小)者"""
    def score(x):
        m, t, p = x
        m = min(m, 4)
        t = min(t, 4 - m)
        p = min(p, 1)
        return 2 * m + t + p
    return a if score(a) >= score(b) else b
