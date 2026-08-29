/* 安康159 - 胡牌判断与向听数(JS版, 支持红中癞子)
 *
 * 与 backend/rules/win.py 逐行对齐, 任何一侧修改必须同步另一侧,
 * 并用 mobile/_gen_cases.js + 跨语言单测验证 0 不一致。
 *
 * 算法: 标准胡牌型 = 4面子(顺/刻) + 1对将, 红中可当任意牌。
 *   isWin  : 枚举将的位置(含红中凑将), 余牌用 _allMelds 检查能否全组面子
 *   shanten: DFS 求 (面子m, 搭子t, 将p) 的 Pareto 前沿,
 *            shanten = min over 前沿 of 2*need - 2m - min(t, need-m) - min(p, 1)
 *            need(面子需求) 由手牌张数推导: 13张->4, 副露一副的10张->3,
 *            因此原生支持副露手, 不需要任何虚拟牌填充
 *
 * 已修复的 5 个历史 bug(与 Python 同步):
 *   bug1 顺子只试以 t 为起点 -> t 是最小现存牌, "t 作中/尾张、低位由红中补"
 *        的组合永不识别(如 89+红中 凑 7-8-9), 直接导致漏胡
 *   bug2 搭子分支的 r<=7 门禁把 8/9 点搭子全部跳过 -> "89两面"不计入向听
 *   bug3 每个子状态只返回单一最优 (m,t,p) -> 最终公式带截断
 *        (min(t,need-m)/min(p,1)), 局部同分的 (3,1,0)/(3,0,1) 全局价值不同,
 *        贪心收敛丢最优 -> 多红中场景漏胡/向听偏高
 *   bug4 对子只当"将" -> 双对子手 "22做将 + 66做搭子" 的 66 不计入搭子
 *   bug5 副露手用"虚拟刻子填充"补回13张再算 -> 填充牌在无孤立空位时
 *        会被 DFS 借去凑真实手的顺子/搭子, 向听被低估(偏乐观)。
 *        现改为 need 参数化 DFS, 彻底废弃填充法。
 *        另: need==0(四副露后只差将) 公式表达不了, 必须特判;
 *        旧版用 max(1,need) 兜底会把 1 张暗牌算成向听 2, 与 isWin 矛盾。
 *
 * 实现层优化(不改变结果):
 *   1. 递归中不 slice 拷贝, 原地改 + 回滚, 共享 scratch
 *   2. 缓存键用 String.fromCharCode 代替 join(",")
 *   3. (m,t,p) 打包成整数, 前沿用 Int 数组表示
 */

const _meldsCache = new Map();
const MELDS_CACHE_LIMIT = 400000;
const _shantenCache = new Map();
const SHANTEN_CACHE_LIMIT = 400000;
// 注: 达到上限时整体 clear() 而不是"停止写入"。
// 旧写法(size < LIMIT 才 set)会在缓存满后永久退化为零缓存,
// 实测使 Bot 单次决策从 ~100ms 退化到 ~6000ms。

// ---- (m,t,p) 打包: 每项 5 bits ----
function _pack(m, t, p) { return (m << 10) | (t << 5) | p; }
function _pm(v) { return v >> 10; }
function _pt(v) { return (v >> 5) & 31; }
function _pp(v) { return v & 31; }

function _keyOf(counts, extra) {
  // counts 长度 >=27, 值 0..4
  return String.fromCharCode(
    counts[0], counts[1], counts[2], counts[3], counts[4], counts[5],
    counts[6], counts[7], counts[8], counts[9], counts[10], counts[11],
    counts[12], counts[13], counts[14], counts[15], counts[16], counts[17],
    counts[18], counts[19], counts[20], counts[21], counts[22], counts[23],
    counts[24], counts[25], counts[26], extra);
}

/** 两个额外维度的缓存键(用于 DFS: redLeft + need)。
 * need 必须进缓存键 —— 同一副暗牌在不同 need 下的 Pareto 前沿
 * 截断位置不同(_prune 按 need 封顶), 混用会读到错的前沿。 */
function _keyOf2(counts, e1, e2) {
  return String.fromCharCode(
    counts[0], counts[1], counts[2], counts[3], counts[4], counts[5],
    counts[6], counts[7], counts[8], counts[9], counts[10], counts[11],
    counts[12], counts[13], counts[14], counts[15], counts[16], counts[17],
    counts[18], counts[19], counts[20], counts[21], counts[22], counts[23],
    counts[24], counts[25], counts[26], e1, e2);
}

/* ================= 胡牌判断 ================= */

function _allMelds(c, red) {
  const key = _keyOf(c, red);
  const hit = _meldsCache.get(key);
  if (hit !== undefined) return hit;

  let t = -1;
  for (let i = 0; i < 27; i++) {
    if (c[i] > 0) { t = i; break; }
  }
  let result = false;
  if (t === -1) {
    result = red % 3 === 0;
  } else {
    // 刻子
    if (!result && c[t] >= 3) {
      c[t] -= 3;
      if (_allMelds(c, red)) result = true;
      c[t] += 3;
    }
    if (!result && c[t] >= 2 && red >= 1) {
      c[t] -= 2;
      if (_allMelds(c, red - 1)) result = true;
      c[t] += 2;
    }
    if (!result && c[t] >= 1 && red >= 2) {
      c[t] -= 1;
      if (_allMelds(c, red - 2)) result = true;
      c[t] += 1;
    }
    // 顺子: t 可为顺子的头/中/尾张。t 是最小现存牌, 起点低于 t 的位置
    // 必然缺牌、由红中补(bug1)
    if (!result) {
      const s = tileSuit(t);
      if (s !== HZ) {
        for (let d = -2; d <= 0 && !result; d++) {
          const start = t + d;
          if (start < 0 || tileSuit(start) !== s || tileRank(start) > 7) continue;
          const a = start, b = start + 1, e = start + 2;
          const ua = c[a] >= 1 ? 1 : 0;
          const ub = c[b] >= 1 ? 1 : 0;
          const ue = c[e] >= 1 ? 1 : 0;
          const need = 3 - ua - ub - ue;
          if (need > red) continue;
          c[a] -= ua; c[b] -= ub; c[e] -= ue;
          const ok = _allMelds(c, red - need);
          c[a] += ua; c[b] += ub; c[e] += ue;
          if (ok) result = true;
        }
      }
    }
  }
  if (_meldsCache.size >= MELDS_CACHE_LIMIT) _meldsCache.clear();
  _meldsCache.set(key, result);
  return result;
}

const _winScratch = new Array(27).fill(0);

function isWin(tilesCounts) {
  let total = 0;
  for (let i = 0; i < tilesCounts.length; i++) total += tilesCounts[i];
  if (total % 3 !== 2) return false;
  const red = tilesCounts[RED];
  const base = _winScratch;
  for (let i = 0; i < 27; i++) base[i] = tilesCounts[i];

  // 将 = 普通对子, 可用 0/1/2 张红中凑
  for (let t = 0; t < 27; t++) {
    for (let need = 0; need <= 2; need++) {
      if (need > red) continue;
      const take = 2 - need;
      if (base[t] + need >= 2 && base[t] >= take) {
        base[t] -= take;
        const ok = _allMelds(base, red - need);
        base[t] += take;
        if (ok) return true;
      }
    }
  }
  // 将 = 两张红中
  if (red >= 2 && _allMelds(base, red - 2)) return true;
  return false;
}

/* ================= 向听数 ================= */

/** 截断到公式上限(按 need)后保留分量支配意义下的 Pareto 前沿 */
function _prune(cands, need) {
  // cands: Set<number> of packed (m,t,p)
  const capped = new Set();
  for (const v of cands) {
    let m = _pm(v); if (m > need) m = need;
    let t = _pt(v); if (t > need) t = need;
    let p = _pp(v); if (p > 1) p = 1;
    capped.add(_pack(m, t, p));
  }
  const arr = Array.from(capped);
  const out = [];
  for (const x of arr) {
    const xm = _pm(x), xt = _pt(x), xp = _pp(x);
    let dominated = false;
    for (const y of arr) {
      if (y === x) continue;
      if (_pm(y) >= xm && _pt(y) >= xt && _pp(y) >= xp) { dominated = true; break; }
    }
    if (!dominated) out.push(x);
  }
  out.sort((a, b) => a - b);
  return out;
}

/**
 * 返回 (m面子, t搭子, p将) 的 Pareto 前沿(packed int 的升序数组)。
 * 返回值被缓存共享, 调用方不得修改。
 *
 * @param {number} need 这副手还需凑的面子总数(含红中凑的): 普通13张=4,
 *   副露 n 副后暗牌=4-n。参与缓存键与 _prune 截断。
 */
function _shantenDfs(c, redLeft, need) {
  const key = _keyOf2(c, redLeft, need);
  const hit = _shantenCache.get(key);
  if (hit !== undefined) return hit;

  let t = -1;
  for (let i = 0; i < 27; i++) {
    if (c[i] > 0) { t = i; break; }
  }

  let result;
  if (t === -1) {
    const m = Math.floor(redLeft / 3);
    const rem = redLeft % 3;
    result = (rem === 2)
      ? _prune(new Set([_pack(m, 0, 1), _pack(m, 1, 0)]), need)
      : _prune(new Set([_pack(m, 0, 0)]), need);
  } else {
    const cands = new Set();
    const add = (sub, dm, dt, dp) => {
      for (const v of sub) {
        cands.add(_pack(_pm(v) + dm, _pt(v) + dt, _pp(v) + dp));
      }
    };

    // 选项1: 孤张跳过
    c[t] -= 1;
    add(_shantenDfs(c, redLeft, need), 0, 0, 0);
    c[t] += 1;

    // 选项2: 对子(将 或 刻子搭子; bug4)
    if (c[t] >= 2) {
      c[t] -= 2;
      const sub = _shantenDfs(c, redLeft, need);
      add(sub, 0, 0, 1);
      add(sub, 0, 1, 0);
      c[t] += 2;
    }
    if (c[t] >= 1 && redLeft >= 1) {
      c[t] -= 1;
      add(_shantenDfs(c, redLeft - 1, need), 0, 0, 1);
      c[t] += 1;
    }

    // 选项3: 刻子
    if (c[t] >= 3) {
      c[t] -= 3;
      add(_shantenDfs(c, redLeft, need), 1, 0, 0);
      c[t] += 3;
    }
    if (c[t] >= 2 && redLeft >= 1) {
      c[t] -= 2;
      add(_shantenDfs(c, redLeft - 1, need), 1, 0, 0);
      c[t] += 2;
    }

    const s = tileSuit(t);
    if (s !== HZ) {
      // 选项4: 顺子面子 (t 可为头/中/尾张, 缺牌由红中补; bug1)
      for (let d = -2; d <= 0; d++) {
        const start = t + d;
        if (start < 0 || tileSuit(start) !== s || tileRank(start) > 7) continue;
        const a = start, b = start + 1, e = start + 2;
        const ua = c[a] >= 1 ? 1 : 0;
        const ub = c[b] >= 1 ? 1 : 0;
        const ue = c[e] >= 1 ? 1 : 0;
        const needRed = 3 - ua - ub - ue;
        if (needRed > redLeft) continue;
        c[a] -= ua; c[b] -= ub; c[e] -= ue;
        add(_shantenDfs(c, redLeft - needRed, need), 1, 0, 0);
        c[a] += ua; c[b] += ub; c[e] += ue;
      }

      // 选项5: 搭子 (bug2: 旧代码 r<=7 门禁把 8/9 点搭子全部跳过)
      const r = tileRank(t);
      // 两面/边张 t,t+1 (同花色: r<=8)
      if (r <= 8 && c[t + 1] >= 1) {
        c[t] -= 1; c[t + 1] -= 1;
        add(_shantenDfs(c, redLeft, need), 0, 1, 0);
        c[t] += 1; c[t + 1] += 1;
      }
      // 嵌张 t,t+2 (同花色: r<=7)
      if (r <= 7 && c[t + 2] >= 1) {
        c[t] -= 1; c[t + 2] -= 1;
        add(_shantenDfs(c, redLeft, need), 0, 1, 0);
        c[t] += 1; c[t + 2] += 1;
      }
      // 红中搭子 t+红 (完成面最宽, 覆盖旧的红中补两面/嵌张)
      if (redLeft >= 1) {
        c[t] -= 1;
        add(_shantenDfs(c, redLeft - 1, need), 0, 1, 0);
        c[t] += 1;
      }
    }
    result = _prune(cands, need);
  }

  if (_shantenCache.size >= SHANTEN_CACHE_LIMIT) _shantenCache.clear();
  _shantenCache.set(key, result);
  return result;
}

const _shScratch = new Array(27).fill(0);

/**
 * 最小向听数(0=听牌, -1=已胡)。输入为长度 28 的计数(索引 27 为红中)。
 * 支持任意扌牌尺寸: need 由张数推导, 副露手直接传暗牌即可。
 */
function shanten(tilesCounts) {
  const red = tilesCounts[RED];
  const c = _shScratch;
  let total = red;
  for (let i = 0; i < 27; i++) { c[i] = tilesCounts[i]; total += c[i]; }
  // 地板除, 与 Python 的 (total - 1) // 3 一致(total=0 时得 -1)
  const need = Math.floor((total - 1) / 3);
  // need<=0 特判(四副露后只差将): 公式 2*need-2m-t-p 表达不了"只差将"
  // (m/t 全 0 时, 不成对的两张会被错算成 0)。不能用 max(1,need) 兜底 ——
  // 那会把 1 张暗牌算成向听 2, 与 isWin 自相矛盾(补成对即判胡)。
  if (need <= 0) {
    if (total <= 1) return 0;            // 1 张暗牌: 听对子
    if (red >= 1) return -1;             // 红中+任意 = 成对
    for (let i = 0; i < 27; i++) if (c[i] >= 2) return -1;
    return 1;
  }
  const front = _shantenDfs(c, red, need);
  let best = 99;
  for (const v of front) {
    let m = _pm(v); if (m > need) m = need;
    let t = _pt(v); const tcap = need - m; if (t > tcap) t = tcap;
    let p = _pp(v); if (p > 1) p = 1;
    const s = 2 * need - 2 * m - t - p;
    if (s < best) best = s;
  }
  return best;
}

/**
 * 副露感知向听: 已有 nMelds 个副露(碰/杠各算1个完成面子)时, 对剩下暗牌的向听数。
 *
 * shanten() 已从手牌张数推导面子需求(13张->4, 副露一副的10张->3),
 * 原生支持副露手, 直接转调即可。nMelds 仅为保持调用端兼容而保留, 不再使用。
 *
 * 历史教训(bug5): 最初的实现用"虚拟刻子填充"把手补回13张再算 —— 填充牌
 * 在无孤立空位时会被 DFS 借去凑真实手的顺子/搭子, 向听被低估。
 *
 * 与 backend/rules/win.py 的 shanten_with_melds 逐行对齐。
 *
 * @param {number[]} concealedCounts 长度 28 的暗牌计数(索引 27 为红中)
 * @param {number} nMelds 副露数(已忽略, 仅保留签名)
 * @returns {number} 向听数(0=听牌, -1=已胡)
 */
function shantenWithMelds(concealedCounts, nMelds) {
  return shanten(concealedCounts);
}
