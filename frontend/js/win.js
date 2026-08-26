/* 安康159 - 胡牌判断与向听数(JS优化版, 支持红中癞子)
 * 与原版算法/遍历顺序完全一致, 仅做实现层优化:
 *   1. 递归中不再 slice 拷贝数组(原地改+回滚, 共享 scratch)
 *   2. 缓存键用 String.fromCharCode 代替 join(",")(实测约 5x)
 *   3. (m,t,p) 三元组打包成整数, 避免每层分配数组
 *   4. score 比较函数提到模块级, 避免每次创建闭包
 */

const _meldsCache = new Map();
const MELDS_CACHE_LIMIT = 400000;
const _shantenCache = new Map();
const SHANTEN_CACHE_LIMIT = 400000;

// ---- (m,t,p) 打包: 5 bits 每项 ----
function _pack(m, t, p) { return (m << 10) | (t << 5) | p; }
function _pm(v) { return v >> 10; }
function _pt(v) { return (v >> 5) & 31; }
function _pp(v) { return v & 31; }

/** 与原版 _combineShanten 的 score 完全一致 */
function _packScore(v) {
  let m = _pm(v); if (m > 4) m = 4;
  let t = _pt(v); const tcap = 4 - m; if (t > tcap) t = tcap;
  let p = _pp(v); if (p > 1) p = 1;
  return 2 * m + t + p;
}

function _keyOf(counts, extra) {
  // counts 长度 27, 值 0..4
  return String.fromCharCode(
    counts[0], counts[1], counts[2], counts[3], counts[4], counts[5],
    counts[6], counts[7], counts[8], counts[9], counts[10], counts[11],
    counts[12], counts[13], counts[14], counts[15], counts[16], counts[17],
    counts[18], counts[19], counts[20], counts[21], counts[22], counts[23],
    counts[24], counts[25], counts[26], extra);
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
    // 顺子
    if (!result) {
      const s = tileSuit(t), r = tileRank(t);
      if (s !== HZ && r <= 7) {
        const t1 = t + 1, t2 = t + 2;
        let need = 0;
        let u0 = 0, u1 = 0, u2 = 0;
        if (c[t] >= 1) u0 = 1; else need++;
        if (t1 < 27 && c[t1] >= 1) u1 = 1; else need++;
        if (t2 < 27 && c[t2] >= 1) u2 = 1; else need++;
        if (need <= red) {
          c[t] -= u0; c[t1] -= u1; c[t2] -= u2;
          if (_allMelds(c, red - need)) result = true;
          c[t] += u0; c[t1] += u1; c[t2] += u2;
        }
      }
    }
  }
  if (_meldsCache.size < MELDS_CACHE_LIMIT) _meldsCache.set(key, result);
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

function _shantenDfs(c, redLeft) {
  const key = _keyOf(c, redLeft);
  const hit = _shantenCache.get(key);
  if (hit !== undefined) return hit;

  let t = -1;
  for (let i = 0; i < 27; i++) {
    if (c[i] > 0) { t = i; break; }
  }
  let best;
  if (t === -1) {
    const m = (redLeft / 3) | 0, rem = redLeft % 3;
    best = rem === 2 ? _pack(m, 0, 1) : _pack(m, 0, 0);
  } else {
    best = 0;   // pack(0,0,0)
    let bestScore = 0;
    const relax = (v) => {
      const sc = _packScore(v);
      if (sc > bestScore) { bestScore = sc; best = v; }
    };

    // 孤张跳过
    c[t] -= 1;
    relax(_shantenDfs(c, redLeft));
    c[t] += 1;

    // 对子(将)
    if (c[t] >= 2) {
      c[t] -= 2;
      const s = _shantenDfs(c, redLeft);
      relax(_pack(_pm(s), _pt(s), _pp(s) + 1));
      c[t] += 2;
    }
    if (c[t] >= 1 && redLeft >= 1) {
      c[t] -= 1;
      const s = _shantenDfs(c, redLeft - 1);
      relax(_pack(_pm(s), _pt(s), _pp(s) + 1));
      c[t] += 1;
    }

    // 刻子
    if (c[t] >= 3) {
      c[t] -= 3;
      const s = _shantenDfs(c, redLeft);
      relax(_pack(_pm(s) + 1, _pt(s), _pp(s)));
      c[t] += 3;
    }
    if (c[t] >= 2 && redLeft >= 1) {
      c[t] -= 2;
      const s = _shantenDfs(c, redLeft - 1);
      relax(_pack(_pm(s) + 1, _pt(s), _pp(s)));
      c[t] += 2;
    }

    const su = tileSuit(t), r = tileRank(t);
    if (su !== HZ && r <= 7) {
      const t1 = t + 1, t2 = t + 2;
      // 顺子
      let need = 0, u0 = 0, u1 = 0, u2 = 0;
      if (c[t] >= 1) u0 = 1; else need++;
      if (t1 < 27 && c[t1] >= 1) u1 = 1; else need++;
      if (t2 < 27 && c[t2] >= 1) u2 = 1; else need++;
      if (need <= redLeft) {
        c[t] -= u0; c[t1] -= u1; c[t2] -= u2;
        const s = _shantenDfs(c, redLeft - need);
        relax(_pack(_pm(s) + 1, _pt(s), _pp(s)));
        c[t] += u0; c[t1] += u1; c[t2] += u2;
      }
      // 两面搭子 t,t+1
      if (t1 < 27) {
        let n2 = (c[t] >= 1 ? 0 : 1) + (c[t1] >= 1 ? 0 : 1);
        if (n2 > 0 && n2 <= redLeft) {
          const a0 = c[t] >= 1 ? 1 : 0, a1 = c[t1] >= 1 ? 1 : 0;
          c[t] -= a0; c[t1] -= a1;
          const s = _shantenDfs(c, redLeft - n2);
          relax(_pack(_pm(s), _pt(s) + 1, _pp(s)));
          c[t] += a0; c[t1] += a1;
        }
        if (c[t] >= 1 && c[t1] >= 1) {
          c[t] -= 1; c[t1] -= 1;
          const s = _shantenDfs(c, redLeft);
          relax(_pack(_pm(s), _pt(s) + 1, _pp(s)));
          c[t] += 1; c[t1] += 1;
        }
      }
      // 嵌张 t,t+2
      if (t2 < 27) {
        let n2 = (c[t] >= 1 ? 0 : 1) + (c[t2] >= 1 ? 0 : 1);
        if (n2 > 0 && n2 <= redLeft) {
          const a0 = c[t] >= 1 ? 1 : 0, a2 = c[t2] >= 1 ? 1 : 0;
          c[t] -= a0; c[t2] -= a2;
          const s = _shantenDfs(c, redLeft - n2);
          relax(_pack(_pm(s), _pt(s) + 1, _pp(s)));
          c[t] += a0; c[t2] += a2;
        }
        if (c[t] >= 1 && c[t2] >= 1) {
          c[t] -= 1; c[t2] -= 1;
          const s = _shantenDfs(c, redLeft);
          relax(_pack(_pm(s), _pt(s) + 1, _pp(s)));
          c[t] += 1; c[t2] += 1;
        }
      }
    }
  }
  if (_shantenCache.size < SHANTEN_CACHE_LIMIT) _shantenCache.set(key, best);
  return best;
}

const _shScratch = new Array(27).fill(0);

function shanten(tilesCounts) {
  const red = tilesCounts[RED];
  const c = _shScratch;
  for (let i = 0; i < 27; i++) c[i] = tilesCounts[i];
  const v = _shantenDfs(c, red);
  let m = _pm(v); if (m > 4) m = 4;
  let t = _pt(v); const tcap = 4 - m; if (t > tcap) t = tcap;
  let p = _pp(v); if (p > 1) p = 1;
  return 8 - 2 * m - t - p;
}
