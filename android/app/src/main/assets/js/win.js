/* 安康159 - 胡牌判断与向听数(JS版, 支持红中癞子)
 * 翻译自 backend/rules/win.py
 */

// 缓存: key = counts.join(",") + "|" + red
const _meldsCache = new Map();
const MELDS_CACHE_LIMIT = 400000;

function _allMelds(counts, red) {
  const key = counts.join(",") + "|" + red;
  const hit = _meldsCache.get(key);
  if (hit !== undefined) return hit;

  const c = counts.slice();
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
        const use = [0, 0, 0];
        if (c[t] >= 1) use[0] = 1; else need++;
        if (t1 < 27 && c[t1] >= 1) use[1] = 1; else need++;
        if (t2 < 27 && c[t2] >= 1) use[2] = 1; else need++;
        if (need <= red) {
          c[t] -= use[0]; c[t1] -= use[1]; c[t2] -= use[2];
          if (_allMelds(c, red - need)) result = true;
          c[t] += use[0]; c[t1] += use[1]; c[t2] += use[2];
        }
      }
    }
  }
  if (_meldsCache.size < MELDS_CACHE_LIMIT) _meldsCache.set(key, result);
  return result;
}

function isWin(tilesCounts) {
  let total = 0;
  for (const n of tilesCounts) total += n;
  if (total % 3 !== 2) return false;
  const red = tilesCounts[RED];
  const base = tilesCounts.slice(0, 27);

  // 将 = 普通对子, 可用 0/1/2 张红中凑
  for (let t = 0; t < 27; t++) {
    for (let need = 0; need <= 2; need++) {
      if (need > red) continue;
      if (base[t] + need >= 2 && base[t] >= 2 - need) {
        const c = base.slice();
        c[t] -= (2 - need);
        if (_allMelds(c, red - need)) return true;
      }
    }
  }
  // 将 = 两张红中
  if (red >= 2 && _allMelds(base, red - 2)) return true;
  return false;
}

// 向听数: shanten = 8 - 2*m - t - p
const _shantenCache = new Map();
const SHANTEN_CACHE_LIMIT = 400000;

function _combineShanten(a, b) {
  const score = (x) => {
    let m = Math.min(x[0], 4);
    let t = Math.min(x[1], 4 - m);
    let p = Math.min(x[2], 1);
    return 2 * m + t + p;
  };
  return score(a) >= score(b) ? a : b;
}

function _shantenDfs(counts, redLeft) {
  const key = counts.join(",") + "|" + redLeft;
  const hit = _shantenCache.get(key);
  if (hit !== undefined) return hit;

  const c = counts.slice();
  let t = -1;
  for (let i = 0; i < 27; i++) {
    if (c[i] > 0) { t = i; break; }
  }
  let best;
  if (t === -1) {
    const m = Math.floor(redLeft / 3), rem = redLeft % 3;
    best = rem === 2 ? [m, 0, 1] : [m, 0, 0];
  } else {
    best = [0, 0, 0];
    // 孤张跳过
    c[t] -= 1;
    best = _combineShanten(best, _shantenDfs(c, redLeft));
    c[t] += 1;

    // 对子(将)
    if (c[t] >= 2) {
      c[t] -= 2;
      const s = _shantenDfs(c, redLeft);
      best = _combineShanten(best, [s[0], s[1], s[2] + 1]);
      c[t] += 2;
    }
    if (c[t] >= 1 && redLeft >= 1) {
      c[t] -= 1;
      const s = _shantenDfs(c, redLeft - 1);
      best = _combineShanten(best, [s[0], s[1], s[2] + 1]);
      c[t] += 1;
    }

    // 刻子
    if (c[t] >= 3) {
      c[t] -= 3;
      const s = _shantenDfs(c, redLeft);
      best = _combineShanten(best, [s[0] + 1, s[1], s[2]]);
      c[t] += 3;
    }
    if (c[t] >= 2 && redLeft >= 1) {
      c[t] -= 2;
      const s = _shantenDfs(c, redLeft - 1);
      best = _combineShanten(best, [s[0] + 1, s[1], s[2]]);
      c[t] += 2;
    }

    const su = tileSuit(t), r = tileRank(t);
    if (su !== HZ && r <= 7) {
      const t1 = t + 1, t2 = t + 2;
      // 顺子
      let need = 0; const use = [0, 0, 0];
      if (c[t] >= 1) use[0] = 1; else need++;
      if (t1 < 27 && c[t1] >= 1) use[1] = 1; else need++;
      if (t2 < 27 && c[t2] >= 1) use[2] = 1; else need++;
      if (need <= redLeft) {
        c[t] -= use[0]; c[t1] -= use[1]; c[t2] -= use[2];
        const s = _shantenDfs(c, redLeft - need);
        best = _combineShanten(best, [s[0] + 1, s[1], s[2]]);
        c[t] += use[0]; c[t1] += use[1]; c[t2] += use[2];
      }
      // 两面搭子 t,t+1
      if (t1 < 27) {
        let n2 = (c[t] >= 1 ? 0 : 1) + (c[t1] >= 1 ? 0 : 1);
        if (n2 > 0 && n2 <= redLeft) {
          const u0 = c[t] >= 1 ? 1 : 0, u1 = c[t1] >= 1 ? 1 : 0;
          c[t] -= u0; c[t1] -= u1;
          const s = _shantenDfs(c, redLeft - n2);
          best = _combineShanten(best, [s[0], s[1] + 1, s[2]]);
          c[t] += u0; c[t1] += u1;
        }
        if (c[t] >= 1 && c[t1] >= 1) {
          c[t] -= 1; c[t1] -= 1;
          const s = _shantenDfs(c, redLeft);
          best = _combineShanten(best, [s[0], s[1] + 1, s[2]]);
          c[t] += 1; c[t1] += 1;
        }
      }
      // 嵌张 t,t+2
      if (t2 < 27) {
        let n2 = (c[t] >= 1 ? 0 : 1) + (c[t2] >= 1 ? 0 : 1);
        if (n2 > 0 && n2 <= redLeft) {
          const u0 = c[t] >= 1 ? 1 : 0, u2 = c[t2] >= 1 ? 1 : 0;
          c[t] -= u0; c[t2] -= u2;
          const s = _shantenDfs(c, redLeft - n2);
          best = _combineShanten(best, [s[0], s[1] + 1, s[2]]);
          c[t] += u0; c[t2] += u2;
        }
        if (c[t] >= 1 && c[t2] >= 1) {
          c[t] -= 1; c[t2] -= 1;
          const s = _shantenDfs(c, redLeft);
          best = _combineShanten(best, [s[0], s[1] + 1, s[2]]);
          c[t] += 1; c[t2] += 1;
        }
      }
    }
  }
  if (_shantenCache.size < SHANTEN_CACHE_LIMIT) _shantenCache.set(key, best);
  return best;
}

function shanten(tilesCounts) {
  const red = tilesCounts[RED];
  const counts = tilesCounts.slice(0, 27);
  const [m0, t0, p0] = _shantenDfs(counts, red);
  const m = Math.min(m0, 4);
  const t = Math.min(t0, 4 - m);
  const p = Math.min(p0, 1);
  return 8 - 2 * m - t - p;
}
