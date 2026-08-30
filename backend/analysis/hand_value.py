"""牌型价值分析器核心(HandAnalyzer) —— 被 tools/hand_value.py(CLI)、
backend/ai/bot_hv.py(学者Bot)、backend/main.py(分析面板)共用。
模型与口径说明见 tools/hand_value.py 的文件头注释。"""

import math

import numpy as np

from ..native import native

RED = 27

_USEFUL_SET_CACHE = {}


def useful_set(hand) -> list[int]:
    """摸到能降向听(含直接胡)的牌集合。只与手牌有关, 可全局缓存。"""
    key = bytes(hand)
    out = _USEFUL_SET_CACHE.get(key)
    if out is not None:
        return out
    base = native.shanten(list(hand))
    out = []
    for t in range(28):
        if hand[t] >= 4:
            continue
        h = list(hand)
        h[t] += 1
        if native.shanten(h) < base:
            out.append(t)
    _USEFUL_SET_CACHE[key] = out
    return out


class HandAnalyzer:
    def __init__(self, hand, visible_counts, rho=1.0,
                 kaizen=True, kai_margin=2, kai_max=1, kai_topk=6,
                 memo=None, u_eff=None, held_exp=None):
        """hand: 28 计数(当前手牌, 3k+1 或 3k+2 张)。
        visible_counts: 28 计数, 所有可见牌(自己手牌+所有人弃牌+所有副露)。
        rho: 对手摸到你要碰的牌后实际打出来的概率。1.0 = 一定打, 0.0 = 纯自摸。
        kaizen: 换型层 —— 摸到不降向听但让有效张变宽 >= kai_margin 的牌也算进展,
        每条路径最多 kai_max 次(全局预算, 不因降向听而重置 —— 若按"连续次数"
        重置, 每个状态都能花一次预算, 高向听手牌的 DP 图会爆炸, 实测 139s/手)。
        kai_topk: 每个状态最多保留的换型分支数(按进张净增排序) —— 换型层
        会让 DP 状态数从 ~16 爆炸到 ~64 万(单手 0.04s -> 280s), 必须截断;
        0 或 None 表示不截断(仅调试用)。
        memo: 可选外部 dict, 作为 E 的跨实例共享缓存(key 含全部参数)。
        u_eff: 可选 28 向量(允许小数) —— 牌墙中各类牌的期望剩余张数,
        替代默认的均匀假设 u0 = 4 - visible。来自对手牌型分析器:
        u_eff[t] = 不可见[t] - 对手持有期望[t]。只影响自摸/换型通道。
        held_exp: 可选 28 向量 —— 三家对手合计持有各类牌的期望张数。
        给了它, 碰通道权重从 u[t]*3ρ(均匀假设)改为 held_exp[t]*3ρ(按牌细化)。
        两者都不给时行为与旧版完全一致。"""
        self.hand0 = tuple(hand)
        self.u0 = tuple(max(0, 4 - v) for v in visible_counts)
        self.u_eff = (tuple(float(x) for x in u_eff) if u_eff is not None
                      else tuple(float(x) for x in self.u0))
        self.held_exp = tuple(float(x) for x in held_exp) \
            if held_exp is not None else None
        self.rho = rho
        self.kaizen = kaizen
        self.kai_margin = kai_margin
        self.kai_max = kai_max
        self.kai_topk = kai_topk or 99
        self.memo = memo if memo is not None else {}

    # ---------- 有效张(自摸通道) ----------
    def _useful(self, hand, u):
        """摸到能降向听(或胡)的牌 -> {t: 剩余张数}。"""
        s = native.shanten(list(hand))
        return s, {t: u[t] for t in useful_set(hand) if u[t] > 0}

    def _ukeire(self, hand, u):
        return sum(u[t] for t in useful_set(hand) if u[t] > 0)

    # ---------- 碰通道 ----------
    def _peng_transitions(self, hand, u):
        """可碰且有价值的对子 -> {t: (权重, 碰后最优弃牌, 碰后向听)}。
        碰 = 暗手 h[t]-2 成副露, 再从缩短的手牌打出最优一张。
        只在碰后最优向听严格低于当前时才有价值(v31 判碰同口径)。"""
        if self.rho <= 0:
            return {}
        s = native.shanten(list(hand))
        out = {}
        for t in range(27):          # 红中不能被碰
            if hand[t] != 2:
                continue
            if self.held_exp is not None:
                if self.held_exp[t] <= 0:
                    continue
                wt = self.held_exp[t] * 3.0 * self.rho
            else:
                if u[t] <= 0:
                    continue
                wt = u[t] * 3.0 * self.rho
            h2 = list(hand)
            h2[t] -= 2
            # 碰后须打一张: 找最优弃牌(最小向听, 同向听取进张最宽)
            best_d, best_s = -1, 99
            for d in range(28):
                if h2[d] <= 0:
                    continue
                h2[d] -= 1
                sd = native.shanten(h2)
                h2[d] += 1
                if sd < best_s:
                    best_s, best_d = sd, d
            if best_s < s:
                out[t] = (wt, best_d, best_s)
        return out

    def _fast_discard(self, h14, u):
        """摸到有效张后打出哪张: v10 牌效(只保留向听+进张, 关掉两步推演)。"""
        zeros = [0] * 28
        return native.choose_discard_v10(h14, u, zeros, 0.0,
                                         100.0, 1.0, 0.0, 0.0, -1)

    def _kai_candidates(self, hand, u, useful, s):
        """换型候选: [(t, w, gain, h2)], 已按 (-gain,-w) 排序并截断 topk。
        E 与 decompose 共用, 保证口径一致。"""
        kai_tiles = []
        if self.kaizen:
            uk0 = self._ukeire(hand, u)
            u0s = set(t for t, _ in useful)
            cands = []
            for t in range(28):
                if u[t] <= 0 or t in u0s:
                    continue
                h14 = list(hand)
                h14[t] += 1
                d = self._fast_discard(h14, u)
                h14[d] -= 1
                if native.shanten(h14) != s:
                    continue
                gain = self._ukeire(h14, u) - uk0
                if gain >= self.kai_margin:
                    cands.append((gain, t, u[t], tuple(h14)))
            # 只保留进张净增最多的 kai_topk 个分支(状态爆炸的保险丝)
            cands.sort(key=lambda x: (-x[0], -x[2]))
            kai_tiles = cands[: self.kai_topk]
        return kai_tiles

    # ---------- 期望巡数(递推; 巡=轮到自己行动的周期, 碰不消耗摸牌但占一巡) ----------
    def E(self, hand, u, kai=0) -> float:
        key = (hand, u, kai, self.rho, self.kaizen,
               self.kai_margin, self.kai_max, self.kai_topk,
               self.held_exp)
        v = self.memo.get(key)
        if v is not None:
            return v
        s = native.shanten(list(hand))
        useful = [(t, u[t]) for t in useful_set(hand) if u[t] > 0]
        pengs = self._peng_transitions(hand, u)
        kai_tiles = []
        if self.kaizen and kai < self.kai_max:
            kai_tiles = [(t, w, h2) for _g, t, w, h2
                         in self._kai_candidates(hand, u, useful, s)]
        N = sum(u)
        U = (sum(w for _, w in useful) + sum(w for _, w, _ in kai_tiles)
             + sum(w for w, _, _ in pengs.values()))
        if U <= 0:
            val = float(N) + 2.0 * s      # 有效张耗尽的死手
            self.memo[key] = val
            return val
        wait = (N + 1.0) / (U + 1.0)      # 无放回首次命中的精确期望
        val = wait
        for t, w in useful:                # 自摸通道(降向听/胡)
            p = w / U
            h14 = list(hand)
            h14[t] += 1
            if native.is_win(h14):
                continue                  # 这一摸直接胡, 无后续
            u2 = list(u)
            u2[t] -= 1
            d = self._fast_discard(h14, u2)
            h14[d] -= 1
            # kai 透传: 换型预算按整条路径计(不重置), 否则每个状态都能
            # 花一次预算, 高向听手牌的 DP 图会爆炸(实测 139s/手)
            val += p * self.E(tuple(h14), tuple(u2), kai)
        for t, w, h2 in kai_tiles:         # 换型通道: 不降向听但进张变宽
            p = w / U
            u2 = list(u)
            u2[t] -= 1
            val += p * self.E(h2, tuple(u2), kai + 1)
        for t, (w, d, _bs) in pengs.items():   # 碰通道: 对子成副露再弃牌
            p = w / U
            h2 = list(hand)
            h2[t] -= 2
            u2 = list(u)
            u2[t] -= 1
            h2[d] -= 1
            val += p * self.E(tuple(h2), tuple(u2), kai)
        self.memo[key] = val
        return val

    # ---------- E 的通道分解(可解释输出, 与 C mj_hv_explain 同口径) ----------
    def decompose(self, hand) -> dict:
        """hand: 3k+1 计数(出牌后状态)。返回:
        E/wait/c_useful/c_kai/c_peng 五个分量(E = 四项之和, 与 E() 同序累加),
        useful=[(t,剩余)], kai=[(t,剩余,净增)], peng=[(t,权重)] 明细。"""
        hand = tuple(hand)
        u = self.u_eff
        s = native.shanten(list(hand))
        useful = [(t, u[t]) for t in useful_set(hand) if u[t] > 0]
        pengs = self._peng_transitions(hand, u)
        kai_tiles = self._kai_candidates(hand, u, useful, s) \
            if self.kai_max > 0 else []
        N = sum(u)
        U = (sum(w for _, w in useful)
             + sum(w for _g, _t, w, _h in kai_tiles)
             + sum(w for w, _, _ in pengs.values()))
        cU = cK = cP = 0.0
        if U <= 0:
            wait = float(N) + 2.0 * s
            total = wait
        else:
            wait = (N + 1.0) / (U + 1.0)
            total = wait
            for t, w in useful:
                p = w / U
                h14 = list(hand)
                h14[t] += 1
                if native.is_win(h14):
                    continue
                u2 = list(u)
                u2[t] -= 1
                d = self._fast_discard(h14, u2)
                h14[d] -= 1
                c = p * self.E(tuple(h14), tuple(u2), 0)
                total += c
                cU += c
            for _g, t, w, h2 in kai_tiles:
                p = w / U
                u2 = list(u)
                u2[t] -= 1
                c = p * self.E(h2, tuple(u2), 1)
                total += c
                cK += c
            for t, (w, d, _bs) in pengs.items():
                p = w / U
                h2 = list(hand)
                h2[t] -= 2
                u2 = list(u)
                u2[t] -= 1
                h2[d] -= 1
                c = p * self.E(tuple(h2), tuple(u2), 0)
                total += c
                cP += c
        return {"E": total, "wait": wait, "c_useful": cU, "c_kai": cK,
                "c_peng": cP, "useful": useful,
                "kai": [(t, w, g) for g, t, w, _h in kai_tiles],
                "peng": [(t, w) for t, (w, _d, _s) in pengs.items()],
                "shanten": s}

    # ---------- n 进张内可达和牌型 ----------
    def enum_patterns(self, start_hand, max_draws):
        """DFS: 只走有效摸牌 + 保持最小向听的弃牌。按所需摸牌组合聚合。"""
        h0 = list(start_hand)
        assert sum(h0) % 3 == 1, "枚举从 3k+1 张(出牌后)状态开始"
        agg = {}     # needed_tuple -> {"patterns": set, "tiles": [(t,c)]}
        nodes = [0]

        def rec(hand, u, depth):
            nodes[0] += 1
            if nodes[0] > 300000 or depth > max_draws:
                return
            s, useful = self._useful(hand, u)
            for t in useful:
                h14 = list(hand)
                h14[t] += 1
                if native.is_win(h14):
                    needed = tuple(max(0, h14[i] - h0[i]) for i in range(28))
                    ent = agg.setdefault(needed, {"n_pat": 0})
                    ent["n_pat"] += 1
                    continue
                if depth == max_draws:
                    continue
                # 保持最小向听的弃牌才值得继续
                opts = []
                best = 99
                for d in range(28):
                    if h14[d] <= 0:
                        continue
                    h14[d] -= 1
                    sd = native.shanten(h14)
                    h14[d] += 1
                    if sd < best:
                        best = sd
                        opts = [d]
                    elif sd == best:
                        opts.append(d)
                u2 = list(u)
                u2[t] -= 1
                for d in opts:
                    h2 = list(h14)
                    h2[d] -= 1
                    rec(tuple(h2), tuple(u2), depth + 1)

        rec(tuple(h0), tuple(self.u0), 1)
        out = []
        for needed, ent in agg.items():
            tiles = [(t, c) for t, c in enumerate(needed) if c > 0]
            out.append({"tiles": tiles, "n_pat": ent["n_pat"],
                        "draws": sum(c for _, c in tiles)})
        out.sort(key=lambda x: (x["draws"], -x["n_pat"]))
        return out, nodes[0]

    # ---------- 单组合独立期望(包含-排除) ----------
    def pattern_time(self, tiles):
        """集齐所需组合 {(t,c)} 的期望摸牌数(独立地看这一个组合)。
        对子成刻(hand[t]==2 且需要第3张)的牌可用量 ×(1+3ρ)(若开启补偿)。"""
        if len(tiles) > 4 or any(c > 2 for _, c in tiles):
            return None
        h0 = self.hand0
        mult = 1.0 + 3.0 * self.rho
        a = []
        for t, c in tiles:
            av = self.u0[t]
            if self.rho > 0 and h0[t] == 2 and c == 1 and t != RED:
                av = min(int(round(av * mult)), 16)   # 速率×(1+3ρ) ≈ 可用量放大(近似)
            a.append(av)
        N = sum(self.u0)
        m = len(tiles)
        E = 0.0
        # E[T] = sum_k P(T > k); P(T>k) = P(存在某类牌没集齐) = 包含-排除
        for k in range(0, N + 1):
            pin = 0.0
            for mask in range(1, 1 << m):
                bits = bin(mask).count("1")
                sign = 1.0 if bits % 2 == 1 else -1.0
                A = sum(a[i] for i in range(m) if mask >> i & 1)
                # 子集 S 全部未集齐: 每类抽到的数 x_i < c_i
                ranges = [range(0, tiles[i][1]) for i in range(m)
                          if mask >> i & 1]
                idx = [i for i in range(m) if mask >> i & 1]
                tot = 0.0

                def rec2(j, xsum, prod):
                    if j == len(idx):
                        if k - xsum < 0 or k - xsum > N - A:
                            return
                        nonlocal_tot[0] += prod * \
                            math.comb(N - A, k - xsum)
                        return
                    i = idx[j]
                    for x in ranges[j]:
                        rec2(j + 1, xsum + x, prod * math.comb(a[i], x))

                nonlocal_tot = [0.0]
                rec2(0, 0, 1.0)
                pin += sign * nonlocal_tot[0] / math.comb(N, k)
            E += pin
        return E

    # ---------- Monte Carlo 对照(单人摸打模拟) ----------
    def mc(self, start_hand, n_sims=5000, seed=1):
        pool0 = [t for t in range(28) for _ in range(self.u0[t])]
        rng = np.random.default_rng(seed)
        counts = []
        h0 = list(start_hand)
        assert sum(h0) % 3 == 1
        for _ in range(n_sims):
            pool = rng.permutation(pool0)
            h = list(h0)
            u = list(self.u0)
            done = False
            for i, t in enumerate(pool):
                h[t] += 1
                u[t] -= 1
                if native.is_win(h):
                    counts.append(i + 1)
                    done = True
                    break
                d = self._fast_discard(h, u)
                h[d] -= 1
            if not done:
                counts.append(len(pool) + 2)
        return float(np.mean(counts))


