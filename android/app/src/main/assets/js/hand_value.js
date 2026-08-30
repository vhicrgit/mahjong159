/* 安康159 - 牌型价值分析器(HandAnalyzer, JS版)
 *
 * 与 backend/analysis/hand_value.py 逐行对齐, 任何一侧修改必须同步另一侧,
 * 并用跨语言单测验证 0 不一致。
 *
 * 模型: E(hand, u) = 期望巡数(巡 = 轮到自己行动的周期; 碰不消耗摸牌但占一巡)。
 *   递推 = 首次命中有效张的等待巡数 + 各通道命中后的后续期望之和。
 *   三条通道:
 *     自摸通道  摸到能降向听(或直接胡)的牌
 *     换型通道  摸到不降向听但让有效张变宽 >= kaiMargin 的牌(kaizen 层)
 *     碰通道    对子被打出后碰成副露, 再打一张
 *
 * 只移植了学者Bot 需要的部分。Python 侧的 enum_patterns / pattern_time / mc
 * 是 CLI 交互分析用的(依赖 numpy 与包含-排除组合数), 移动端不需要, 未移植。
 *
 * 性能取舍(与 Python 一致): 换型层(kaizen)把单手分析从毫秒级拉到秒级,
 * 只改善绝对值不改善排序, 所以 Bot 对战按 kaizen=false 跑。
 *
 * 【有效张口径】Python 侧其实存在两种 ukeire 定义:
 *   hand_value.useful_set  一律用 shanten(h+t) < base
 *   C 的 tiles_info      对 s==0 特判成 is_win(h+t), 否则 shanten 降低
 * _fast_discard 走 C 口径, _ukeire 走 useful_set 口径。两者在 3k+1 与 3k+2
 * 形态上实测等价(听牌时加牌能胡 <=> 向听降到 -1), 仅在 3k+0 形态上不等价。
 * 而 E() 只会从 3k+1 手牌出发(h14 = hand+1 为 3k+2), 永远不会碰到 3k+0,
 * 所以本文件统一用 usefulSet 一种定义就够了。
 * 注: 若向 E() 传入 3k+2 手牌(越调用约定), h14 会变成 3k+0, 两端会出现
 * 分歧 —— 写测试时必须用对的手牌张数(弃牌用 14-3n, 应对用 13-3n)。
 *
 * 依赖: tiles.js(TILE_COUNT/RED), win.js(shanten/isWin)
 */

/* ================= 有效张集合(全局缓存) ================= */

const _usefulSetCache = new Map();
const USEFUL_SET_CACHE_LIMIT = 200000;

// 注: 曾试过再加两层缓存(顶层 shanten 结果缓存 + _fastDiscard 结果缓存),
// 实测毫无改善(4582ms vs 4781ms, 在噪声内) —— E 递归访问的状态大多互不重复,
// 瓶颈是状态总量而非重复查询。已移除, 避免无收益的复杂度。

/** 28 计数的缓存键。值域 0..4, 用 charCode 拼串比 join(",") 快很多 */
function _hvKey(c) {
  return String.fromCharCode(
    c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7], c[8], c[9],
    c[10], c[11], c[12], c[13], c[14], c[15], c[16], c[17], c[18], c[19],
    c[20], c[21], c[22], c[23], c[24], c[25], c[26], c[27]);
}

/**
 * 摸到能降向听(含直接胡)的牌集合, 升序。只与手牌有关, 可全局缓存。
 * 对齐 hand_value.py 的 useful_set。
 * @param {number[]} hand 长度 28 的计数
 * @returns {number[]} 升序 tile 列表(返回值被缓存共享, 调用方不得修改)
 */
function usefulSet(hand) {
  const key = _hvKey(hand);
  const hit = _usefulSetCache.get(key);
  if (hit !== undefined) return hit;
  const base = shanten(hand);
  const out = [];
  for (let t = 0; t < 28; t++) {
    if (hand[t] >= 4) continue;
    hand[t]++;
    const s = shanten(hand);
    hand[t]--;
    if (s < base) out.push(t);
  }
  // 达到上限整体 clear(), 不能"满了就不写" —— 那会永久退化为零缓存
  if (_usefulSetCache.size >= USEFUL_SET_CACHE_LIMIT) _usefulSetCache.clear();
  _usefulSetCache.set(key, out);
  return out;
}

/* ================= 分析器 ================= */

class HandAnalyzer {
  /**
   * @param {number[]} hand 28 计数(当前手牌, 3k+1 或 3k+2 张)
   * @param {number[]} visibleCounts 28 计数, 所有可见牌(自己手牌+所有人弃牌+所有副露)
   * @param {object} [opts]
   *   rho     对手摸到你要碰的牌后实际打出来的概率。1=一定打, 0=纯自摸
   *   kaizen  换型层开关(Bot 对战用 false)
   *   kaiMargin / kaiMax  换型判定阈值与每条路径的最大换型次数(全局预算,
   *           不随降向听重置; 若按连续次数重置会让高向听手牌 DP 图爆炸)
   *   kaiTopk 每状态最多保留的换型分支数(按进张净增排序)。换型层会让
   *           DP 状态数从 ~16 爆炸到 ~64 万(单手 0.04s -> 280s), 必须截断;
   *           传 0 表示不截断(仅调试用)
   */
  constructor(hand, visibleCounts, opts) {
    const o = opts || {};
    this.hand0 = hand.slice();
    this.u0 = new Array(28);
    for (let i = 0; i < 28; i++) {
      const v = 4 - visibleCounts[i];
      this.u0[i] = v > 0 ? v : 0;
    }
    this.rho = o.rho !== undefined ? o.rho : 1.0;
    this.kaizen = o.kaizen !== undefined ? o.kaizen : true;
    this.kaiMargin = o.kaiMargin !== undefined ? o.kaiMargin : 2;
    this.kaiMax = o.kaiMax !== undefined ? o.kaiMax : 1;
    this.kaiTopk = o.kaiTopk !== undefined ? o.kaiTopk : 6;
    this.memo = new Map();
  }

  /** 有效张总张数(任意向听下) */
  _ukeire(hand, u) {
    let sum = 0;
    for (const t of usefulSet(hand)) if (u[t] > 0) sum += u[t];
    return sum;
  }

  /**
   * 可碰且有价值的对子 -> [[t, 权重, 碰后最优弃牌, 碰后向听], ...] (t 升序)
   *
   * 碰 = 暗手 h[t]-2 成副露, 再从缩短的手牌打出最优一张。
   * 只在碰后最优向听严格低于当前时才有价值(与 v31 判碰同口径)。
   *
   * 顺序必须保持 t 升序 —— E() 里按此顺序累加浮点, 换序会产生末位差异。
   */
  _pengTransitions(hand, u) {
    if (this.rho <= 0) return [];
    const s = shanten(hand);
    const out = [];
    for (let t = 0; t < 27; t++) {          // 红中不能被碰
      if (hand[t] !== 2 || u[t] <= 0) continue;
      const h2 = hand.slice();
      h2[t] -= 2;
      // 碰后须打一张: 找最优弃牌(最小向听)
      let bestD = -1, bestS = 99;
      for (let d = 0; d < 28; d++) {
        if (h2[d] <= 0) continue;
        h2[d]--;
        const sd = shanten(h2);
        h2[d]++;
        if (sd < bestS) { bestS = sd; bestD = d; }
      }
      if (bestS < s) out.push([t, u[t] * 3.0 * this.rho, bestD, bestS]);
    }
    return out;
  }

  /**
   * 摸到有效张后打出哪张: v10 牌效(只保留向听+进张, 关掉两步推演与放杠风险)。
   *
   * 等价于 Python 的
   *   native.choose_discard_v10(h14, u, zeros, 0.0, 100.0, 1.0, 0.0, 0.0, -1)
   * 即 C 里 score_discards_v10 在 SW=100 UW=1 CW=0 RW=0 cont_max=-1 下的行为:
   *   - 向听非最小的候选: score = -10*SW - SW*s (恒为负, 必然落选)
   *   - 向听最小的候选:   score = UW * ukeire
   *   - 取最大, 严格 > 比较且按 tile 升序扫描 -> 同分取最小 tile
   * (cont_max=-1 时 s<=cont_max 不成立, 因为 3k+1 手牌向听 >= 0, 故 cont 恒为 0)
   */
  _fastDiscard(h14, u) {
    const cand = [], cs = [];
    let minSh = 99;
    for (let t = 0; t < 28; t++) {
      if (h14[t] <= 0) continue;
      h14[t]--;
      const s = shanten(h14);
      h14[t]++;
      cand.push(t); cs.push(s);
      if (s < minSh) minSh = s;
    }
    if (!cand.length) return -1;
    let bestT = -1, bestScore = -1e18;
    for (let k = 0; k < cand.length; k++) {
      const t = cand[k], s = cs[k];
      let score;
      if (s > minSh) {
        score = -10.0 * 100.0 - 100.0 * s;
      } else {
        h14[t]--;
        score = 1.0 * this._ukeire(h14, u);
        h14[t]++;
      }
      if (score > bestScore) { bestScore = score; bestT = t; }
    }
    return bestT;
  }

  /**
   * 期望巡数。hand/u 均为长度 28 的计数数组, 不会被修改。
   *
   * 浮点累加顺序必须与 Python 完全一致(wait -> 自摸通道 -> 换型通道 -> 碰通道,
   * 每条通道内按 t 升序), 否则末位会出现差异。
   */
  E(hand, u, kai) {
    if (kai === undefined) kai = 0;
    const key = _hvKey(hand) + "|" + _hvKey(u) + "|" + kai;
    const hit = this.memo.get(key);
    if (hit !== undefined) return hit;

    const s = shanten(hand);
    const useful = [];
    for (const t of usefulSet(hand)) if (u[t] > 0) useful.push([t, u[t]]);
    const pengs = this._pengTransitions(hand, u);

    const kaiTiles = [];
    if (this.kaizen && kai < this.kaiMax) {
      const uk0 = this._ukeire(hand, u);
      const inUseful = new Set();
      for (const [t] of useful) inUseful.add(t);
      const cands = [];
      for (let t = 0; t < 28; t++) {
        if (u[t] <= 0 || inUseful.has(t)) continue;
        const h14 = hand.slice();
        h14[t]++;
        const d = this._fastDiscard(h14, u);
        h14[d]--;
        if (shanten(h14) !== s) continue;
        const gain = this._ukeire(h14, u) - uk0;
        if (gain >= this.kaiMargin) {
          cands.push([gain, t, u[t], h14]);
        }
      }
      // 只保留进张净增最多的 kaiTopk 个分支(状态爆炸的保险丝)
      // 排序与 Python 的 (-gain, -w) 稳定排序(t 升序插入)完全一致
      cands.sort((a, b) => (b[0] - a[0]) || (b[2] - a[2]) || (a[1] - b[1]));
      const cap = this.kaiTopk > 0 ? this.kaiTopk : cands.length;
      for (const [, t, w, h2] of cands.slice(0, cap)) kaiTiles.push([t, w, h2]);
    }

    let N = 0;
    for (let i = 0; i < 28; i++) N += u[i];
    // 与 Python 的 (sum(useful) + sum(kai)) + sum(peng) 同序: 前两项整数, 末项浮点
    let sU = 0;
    for (const [, w] of useful) sU += w;
    let sK = 0;
    for (const kt of kaiTiles) sK += kt[1];
    let sP = 0;
    for (const pg of pengs) sP += pg[1];
    const U = sU + sK + sP;

    if (U <= 0) {
      const val = N + 2.0 * s;            // 有效张耗尽的死手
      this.memo.set(key, val);
      return val;
    }

    let val = (N + 1.0) / (U + 1.0);      // 无放回首次命中的精确期望

    for (const [t, w] of useful) {        // 自摸通道(降向听/胡)
      const p = w / U;
      const h14 = hand.slice();
      h14[t]++;
      if (isWin(h14)) continue;           // 这一摸直接胡, 无后续
      const u2 = u.slice();
      u2[t]--;
      const d = this._fastDiscard(h14, u2);
      h14[d]--;
      // kai 透传: 换型预算按整条路径计(不重置), 否则每个状态都能
      // 花一次预算, 高向听手牌的 DP 图会爆炸(实测 139s/手)
      val += p * this.E(h14, u2, kai);
    }

    for (const [t, w, h2] of kaiTiles) {  // 换型通道: 不降向听但进张变宽
      const p = w / U;
      const u2 = u.slice();
      u2[t]--;
      val += p * this.E(h2, u2, kai + 1);
    }

    for (const [t, w, d] of pengs) {      // 碰通道: 对子成副露再弃牌
      const p = w / U;
      const h2 = hand.slice();
      h2[t] -= 2;
      const u2 = u.slice();
      u2[t]--;
      h2[d]--;
      val += p * this.E(h2, u2, kai);
    }

    this.memo.set(key, val);
    return val;
  }
}
