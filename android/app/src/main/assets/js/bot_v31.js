/* 安康159 - 老鸟 Bot (JS版): v31 = v10 两步推演 + 副露感知向听
 *
 * 逐行移植自 backend/ai/bot_v10.py 与 backend/ai/bot_v31.py,
 * 修改时必须同步两边, 并用跨语言单测验证。
 *
 * 依赖: tiles.js, win.js(shanten/shantenWithMelds/isWin),
 *       ting.js(discardOptions/usefulDraws/waitingTiles)
 *
 * v10 核心思想: 坚持最小向听优先, 在同向听候选间用
 *   "任意向听下的有效进张(ukeire)" + "一步摸牌后最优弃牌价值(两步推演)"
 * 做 tie-break, 替代 v1/v2 只在听牌时才有 wait_count 的缺口。
 *
 * v31 增量: 有副露时全部换用 shantenWithMelds, 并修正 decidePeng/decideGang,
 * 修复 v1~v30 "从不鸣牌" 的隐性 bug。
 */

// ---- v10 默认权重(对齐 bot_v10.py 的环境变量默认值) ----
const V10_SHANTEN_W = 100.0;
const V10_UKEIRE_W = 1.0;
const V10_CONT_W = 0.5;
const V10_RISK_W = 0.0;      // 注意: v10/v31 默认不加放杠风险惩罚
const V10_CONT_MAX_SH = 2;

/* ================= 带缓存的辅助量 ================= */

const _ukCache = new Map();
const _ssCache = new Map();
const UK_CACHE_LIMIT = 200000;
const SS_CACHE_LIMIT = 120000;
// 达到上限整体 clear(), 避免"满了就不再写入"导致的永久退化

function _handKey(counts) {
  return String.fromCharCode.apply(null, counts);
}

/** 有效进张张数: 听牌时用听口, 否则用能降向听的进张 */
function _ukeire(hand13, unseen, nMelds) {
  const key = _handKey(hand13) + "|" + nMelds + "|" + _handKey(unseen);
  const hit = _ukCache.get(key);
  if (hit !== undefined) return hit;
  const s = shantenWithMelds(hand13, nMelds);
  let tiles;
  if (s === 0) {
    tiles = nMelds === 0 ? waitingTiles(hand13) : _waitsWithMelds(hand13, nMelds);
  } else {
    tiles = [];
    for (let t = 0; t < TILE_COUNT; t++) {
      if (hand13[t] >= 4) continue;
      hand13[t]++;
      const s1 = shantenWithMelds(hand13, nMelds);
      hand13[t]--;
      if (s1 < s) tiles.push(t);
    }
  }
  let sum = 0;
  for (const t of tiles) sum += unseen[t];
  if (_ukCache.size >= UK_CACHE_LIMIT) _ukCache.clear();
  _ukCache.set(key, sum);
  return sum;
}

/** 副露状态下的听口: 摸一张后 shantenWithMelds 变 -1(已胡) */
function _waitsWithMelds(hand, nMelds) {
  const out = [];
  for (let t = 0; t < TILE_COUNT; t++) {
    if (hand[t] >= 4) continue;
    hand[t]++;
    const s = shantenWithMelds(hand, nMelds);
    hand[t]--;
    if (s < 0) out.push(t);
  }
  return out;
}

/** 两步推演: 摸一张后按最优弃牌能到什么局面, 期望加权 */
function _secondStepValue(hand13, unseen, nMelds) {
  const key = _handKey(hand13) + "|" + nMelds + "|" + _handKey(unseen);
  const hit = _ssCache.get(key);
  if (hit !== undefined) return hit;

  let total = 0;
  for (let i = 0; i < unseen.length; i++) total += unseen[i];
  if (total <= 0) return 0.0;

  const baseS = shantenWithMelds(hand13, nMelds);
  let v = 0.0;
  for (let draw = 0; draw < TILE_COUNT; draw++) {
    const n = unseen[draw];
    if (n <= 0) continue;
    hand13[draw]++;
    if (baseS === 0) {
      const won = nMelds === 0 ? isWin(hand13)
        : shantenWithMelds(hand13, nMelds) < 0;
      if (won) {
        v += n / total * 50.0;
        hand13[draw]--;
        continue;
      }
    }
    let bestS = 99, bestU = 0;
    for (let disc = 0; disc < TILE_COUNT; disc++) {
      if (hand13[disc] <= 0) continue;
      hand13[disc]--;
      const s = shantenWithMelds(hand13, nMelds);
      const u = _ukeire(hand13, unseen, nMelds);
      hand13[disc]++;
      if (s < bestS || (s === bestS && u > bestU)) { bestS = s; bestU = u; }
    }
    hand13[draw]--;
    v += n / total * (20.0 * Math.max(0, baseS - bestS) + 0.15 * bestU);
  }
  if (_ssCache.size >= SS_CACHE_LIMIT) _ssCache.clear();
  _ssCache.set(key, v);
  return v;
}

/* ================= 老鸟 Bot ================= */

class BotV31 extends Bot {
  constructor(game, seat, opts) {
    super(game, seat);
    const o = opts || {};
    this.shantenWeight = o.shantenWeight !== undefined ? o.shantenWeight : V10_SHANTEN_W;
    this.ukeireWeight = o.ukeireWeight !== undefined ? o.ukeireWeight : V10_UKEIRE_W;
    this.contWeight = o.contWeight !== undefined ? o.contWeight : V10_CONT_W;
    this.riskWeight = o.riskWeight !== undefined ? o.riskWeight : V10_RISK_W;
    this.contMaxShanten = o.contMaxShanten !== undefined ? o.contMaxShanten : V10_CONT_MAX_SH;
  }

  _visibleCounts() {
    const visible = new Array(TILE_COUNT).fill(0);
    for (const q of this.game.players) {
      for (const t of q.discards) visible[t]++;
      for (const m of q.melds) visible[m.tile] += (m.type === "peng" ? 3 : 4);
    }
    const mc = this.game.players[this.seat].handCounts();
    for (let t = 0; t < TILE_COUNT; t++) visible[t] += mc[t];
    return visible;
  }

  _unseenCounts() {
    const visible = this._visibleCounts();
    const out = new Array(TILE_COUNT);
    for (let t = 0; t < TILE_COUNT; t++) out[t] = Math.max(0, 4 - visible[t]);
    return out;
  }

  _pengedByOthers() {
    const out = new Set();
    for (const q of this.game.players) {
      if (q.seat === this.seat) continue;
      for (const m of q.melds) if (m.type === "peng") out.add(m.tile);
    }
    return out;
  }

  _endgameFactor() {
    return Math.max(0, Math.min(1, (60 - this.game.wallRemaining()) / 60));
  }

  chooseDiscard() {
    const p = this.game.players[this.seat];
    const nMelds = p.melds.length;
    const counts14 = p.handCounts();
    const unseen = this._unseenCounts();
    const penged = this._pengedByOthers();
    const eg = this._endgameFactor();

    // 候选 (打出的牌, 打出后的向听)
    const opts = [];
    for (let t = 0; t < TILE_COUNT; t++) {
      if (counts14[t] <= 0) continue;
      counts14[t]--;
      opts.push({ tile: t, shanten: shantenWithMelds(counts14, nMelds) });
      counts14[t]++;
    }
    if (!opts.length) return p.hand[p.hand.length - 1];

    let minSh = Infinity;
    for (const o of opts) if (o.shanten < minSh) minSh = o.shanten;

    let bestT = null, bestScore = -1e18;
    for (const o of opts) {
      const t = o.tile, s = o.shanten;
      let score;
      if (s > minSh) {
        // 非最小向听: 直接重罚, 保证"最小向听优先"
        score = -10.0 * this.shantenWeight - this.shantenWeight * s;
      } else {
        counts14[t]--;
        const u = _ukeire(counts14, unseen, nMelds);
        const cont = (s <= this.contMaxShanten)
          ? _secondStepValue(counts14, unseen, nMelds) : 0.0;
        counts14[t]++;
        score = this.ukeireWeight * u + this.contWeight * cont;
      }
      if (t !== RED && this.riskWeight > 0) {
        let risk;
        if (penged.has(t)) risk = 1.0;
        else {
          const m = { 3: 0.4, 2: 0.2, 1: 0.05, 0: 0.0 };
          risk = m[unseen[t]] !== undefined ? m[unseen[t]] : 0.4;
        }
        score -= this.riskWeight * (1.0 + 1.5 * eg) * risk;
      }
      if (score > bestScore) { bestScore = score; bestT = t; }
    }
    return bestT;
  }

  decidePeng(tile) {
    const p = this.game.players[this.seat];
    const nMelds = p.melds.length;
    const counts = p.handCounts();
    const before = shantenWithMelds(counts, nMelds);
    const c11 = counts.slice();
    c11[tile] -= 2;
    let after = 99;
    for (let d = 0; d < TILE_COUNT; d++) {
      if (c11[d] <= 0) continue;
      c11[d]--;
      const s = shantenWithMelds(c11, nMelds + 1);
      c11[d]++;
      if (s < after) after = s;
    }
    if (after < before) return before !== 0 || after === 0;
    return false;
  }

  decideGang(tile, kind) {
    const p = this.game.players[this.seat];
    const nMelds = p.melds.length;
    const counts = p.handCounts();
    const before = shantenWithMelds(counts, nMelds);
    const c = counts.slice();
    let n2;
    if (kind === "ming") { c[tile] -= 3; n2 = nMelds + 1; }
    else if (kind === "an") { c[tile] -= 4; n2 = nMelds + 1; }
    else { c[tile] -= 1; n2 = nMelds; }   // bu: 碰转杠, 副露数不变
    const after = shantenWithMelds(c, n2);
    return !(before === 0 && after > 0);
  }
}
