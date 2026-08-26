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
    # 顺子: t 可为顺子的头/中/尾张。t 是最小现存牌, 起点低于 t 的位置
    # 必然缺牌、由红中补(历史bug: 只试以 t 为起点 → "89+红中"等永不识别)
    s = tile_suit(t)
    if s != HZ:
        for start in (t - 2, t - 1, t):
            if start < 0 or tile_suit(start) != s or tile_rank(start) > 7:
                continue
            trio = (start, start + 1, start + 2)
            use = [1 if counts[x] >= 1 else 0 for x in trio]
            need = 3 - sum(use)
            if need > red:
                continue
            for x, u in zip(trio, use):
                counts[x] -= u
            ok = _all_melds_cached(tuple(counts), red - need)
            for x, u in zip(trio, use):
                counts[x] += u
            if ok:
                return True
    return False


@lru_cache(maxsize=200000)
def _all_melds_cached(counts: tuple, red: int) -> bool:
    return _all_melds(counts, red)


@lru_cache(maxsize=500000)
def shanten_cached(tiles_counts: tuple) -> int:
    return shanten(list(tiles_counts))


def shanten_with_melds(concealed_counts, n_melds: int) -> int:
    """副露感知向听: 已有 n_melds 个副露时, 对剩余暗牌的向听数。

    shanten() 已从手牌张数推导面子需求(13张→4, 副露1副的10张→3),
    原生支持副露手, 直接调用即可。

    历史教训: 最初的实现用"虚拟刻子填充"把手补回13张再算 —— 填充牌
    在无孤立空位时会被 DFS 挪用进真实手的顺子/搭子, 向听被低估。
    """
    return shanten(list(concealed_counts))


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


@lru_cache(maxsize=800000)
def _dfs_cached(counts: tuple, red_left: int, need: int) -> tuple:
    """返回 (m面子, t搭子, p将) 的 Pareto 前沿集合(tuple of tuples)。

    need = 这副手还需要凑的面子总数（含红中凑的）: 普通13张=4,
    副露 n 副后暗牌=4-n。原生支持任意手牌尺寸, 无需虚拟牌填充。
    m/t 的公式上限就是 need, 在剪枝时直接按 need 截断。

    历史bug3: 旧版每个子状态只返回单一最优 (m,t,p), 但最终公式
    shanten = 2*need - 2m - min(t,need-m) - min(p,1) 有截断, 局部同分的
    (3,1,0)/(3,0,1) 全局价值不同 → 贪心收敛丢最优, 多红中漏胡/向听偏高。
    """
    counts = list(counts)
    t = -1
    for i in range(27):
        if counts[i] > 0:
            t = i
            break
    if t == -1:
        m, rem = divmod(red_left, 3)
        if rem == 2:
            return _prune({(m, 0, 1), (m, 1, 0)}, need)
        return _prune({(m, 0, 0)}, need)

    cands: set = set()

    def add(sub: tuple, dm: int, dt: int, dp: int):
        for m, tt, p in sub:
            cands.add((m + dm, tt + dt, p + dp))

    # 选项1: 孤张跳过
    counts[t] -= 1
    add(_dfs_cached(tuple(counts), red_left, need), 0, 0, 0)
    counts[t] += 1

    # 选项2: 对子(将 或 刻子搭子; 历史bug4: 旧版对子只当将,
    # 双对子手"22将+66搭子"的 66 永不计入搭子)
    if counts[t] >= 2:
        counts[t] -= 2
        sub = _dfs_cached(tuple(counts), red_left, need)
        add(sub, 0, 0, 1)
        add(sub, 0, 1, 0)
        counts[t] += 2
    if counts[t] >= 1 and red_left >= 1:
        counts[t] -= 1
        add(_dfs_cached(tuple(counts), red_left - 1, need), 0, 0, 1)
        counts[t] += 1

    # 选项3: 刻子
    if counts[t] >= 3:
        counts[t] -= 3
        add(_dfs_cached(tuple(counts), red_left, need), 1, 0, 0)
        counts[t] += 3
    if counts[t] >= 2 and red_left >= 1:
        counts[t] -= 2
        add(_dfs_cached(tuple(counts), red_left - 1, need), 1, 0, 0)
        counts[t] += 2

    # 选项4: 顺子面子 (t 可为头/中/尾张, 缺牌由红中补;
    # 历史bug1: 只试以 t 为起点 → 红中补低位的顺子永不识别)
    s = tile_suit(t)
    if s != HZ:
        for start in (t - 2, t - 1, t):
            if start < 0 or tile_suit(start) != s or tile_rank(start) > 7:
                continue
            trio = (start, start + 1, start + 2)
            use = [1 if counts[x] >= 1 else 0 for x in trio]
            need_red = 3 - sum(use)
            if need_red > red_left:
                continue
            for x, u in zip(trio, use):
                counts[x] -= u
            add(_dfs_cached(tuple(counts), red_left - need_red, need), 1, 0, 0)
            for x, u in zip(trio, use):
                counts[x] += u

        # 选项5: 搭子 (历史bug2: 旧代码 r<=7 门禁把 8/9 点的搭子分支
        # 全部跳过 → "89两面"永不计入向听)
        r = tile_rank(t)
        # 两面/边张 t,t+1 (同花色: r<=8)
        if r <= 8 and counts[t + 1] >= 1:
            counts[t] -= 1
            counts[t + 1] -= 1
            add(_dfs_cached(tuple(counts), red_left, need), 0, 1, 0)
            counts[t] += 1
            counts[t + 1] += 1
        # 嵌张 t,t+2 (同花色: r<=7)
        if r <= 7 and counts[t + 2] >= 1:
            counts[t] -= 1
            counts[t + 2] -= 1
            add(_dfs_cached(tuple(counts), red_left, need), 0, 1, 0)
            counts[t] += 1
            counts[t + 2] += 1
        # 红中搭子 t+红 (完成面最宽, 覆盖旧的红中补两面/嵌张)
        if red_left >= 1:
            counts[t] -= 1
            add(_dfs_cached(tuple(counts), red_left - 1, need), 0, 1, 0)
            counts[t] += 1
    return _prune(cands, need)


def _prune(cands: set, need: int = 4) -> tuple:
    """截断到公式上限后保留分量支配意义下的 Pareto 前沿。"""
    capped = {(min(m, need), min(t, need), min(p, 1)) for m, t, p in cands}
    return tuple(sorted(
        x for x in capped
        if not any(y != x and y[0] >= x[0] and y[1] >= x[1] and y[2] >= x[2]
                   for y in capped)))


def shanten(tiles_counts: list[int]) -> int:
    """最小向听数(0=听牌, -1=已胡)。支持红中与任意手牌尺寸。

    need(面子需求)由手牌张数推导: 13张→4, 副露1副的10张→3。
    常数固定 2*need: 13张经典 8 - 2m - t - p; 14张(3n+2)自动得 -1=胡。
    """
    red = tiles_counts[RED]
    counts = tuple(tiles_counts[:27])
    total = sum(counts) + red
    need = max(1, (total - 1) // 3)
    best = 99
    for m, t, p in _dfs_cached(counts, red, need):
        m = min(m, need)
        t = min(t, need - m)
        p = min(p, 1)
        best = min(best, 2 * need - 2 * m - t - p)
    return best
