/* 安康159 - 进阶 Bot(JS版): 老鸟(v4) 与 挂哥(oracle)
 * 移植自 backend/ai/bot_v4.py 与 backend/ai/bot_oracle.py
 * 依赖: tiles.js, win.js(isWin/shanten), ting.js(discardOptions), bot.js(Bot=菜鸟v1)
 */

const WIN_DISCOUNT = 0.85;   // 效用 = DISCOUNT^(到胡牌所需摸牌数)

/**
 * beam search: 每个"首出牌"的结果明细
 * @param {number[]} counts14 手牌计数(14张)
 * @param {number[]} futureDraws 未来会摸到的牌序列
 * @param {number} beam 每个首出牌保留的后继数
 * @param {Set<number>|null} candidates 限定只搜这些首出牌
 * @returns {Map<number, {wd: number|null, sh: number}>} wd=胡牌深度(0=下次摸牌就胡)
 */
function searchFirstDiscardDetail(counts14, futureDraws, beam, candidates) {
  const horizon = futureDraws.length;
  const detail = new Map();
  const won = new Set();
  let beamNodes = [];   // [handCountsArray, firstTile, shanten]

  for (let t = 0; t < TILE_COUNT; t++) {
    if (counts14[t] <= 0) continue;
    if (candidates && !candidates.has(t)) continue;
    const c = counts14.slice();
    c[t] -= 1;
    const s = shanten(c);
    beamNodes.push([c, t, s]);
    detail.set(t, { wd: null, sh: s });
  }
  if (!beamNodes.length) return detail;

  for (let depth = 0; depth < futureDraws.length; depth++) {
    const draw = futureDraws[depth];
    const nextNodes = [];
    const seen = new Set();
    for (const [hand13, firstT] of beamNodes) {
      if (won.has(firstT)) continue;
      const c = hand13.slice();
      c[draw] += 1;
      if (isWin(c)) {
        won.add(firstT);
        detail.set(firstT, { wd: depth, sh: -1 });
        continue;
      }
      for (let t = 0; t < TILE_COUNT; t++) {
        if (c[t] <= 0) continue;
        c[t] -= 1;
        const key = c.join(",") + "|" + firstT;
        if (!seen.has(key)) {
          seen.add(key);
          nextNodes.push([c.slice(), firstT, shanten(c)]);
        }
        c[t] += 1;
      }
    }
    if (!nextNodes.length) break;
    // 分组保留: 每个首出牌各留 beam 个最优后继
    const byFirst = new Map();
    for (const item of nextNodes) {
      if (!byFirst.has(item[1])) byFirst.set(item[1], []);
      byFirst.get(item[1]).push(item);
    }
    beamNodes = [];
    for (const [ft, items] of byFirst) {
      items.sort((a, b) => a[2] - b[2]);
      for (let i = 0; i < Math.min(beam, items.length); i++) beamNodes.push(items[i]);
      if (!won.has(ft)) {
        const bestS = items[0][2];
        const prev = detail.get(ft) || { wd: null, sh: 99 };
        if (prev.wd === null && bestS < prev.sh) detail.set(ft, { wd: null, sh: bestS });
      }
    }
  }
  return detail;
}

function noWinUtility(horizon, shantenLeft) {
  return Math.pow(WIN_DISCOUNT, horizon + 2 * Math.max(0, shantenLeft) + 1);
}

/* ============ 老鸟: v4 解析骨架 + 同向听内采样搜索精修 ============ */
class BotV4 extends Bot {
  constructor(game, seat, opts) {
    super(game, seat);
    const o = opts || {};
    // 手机端性能考虑: worlds 默认降到 16(Python 版 48)
    this.worlds = o.worlds !== undefined ? o.worlds : 16;
    this.beam = o.beam !== undefined ? o.beam : 5;
    this.horizon = o.horizon !== undefined ? o.horizon : 5;
    this.refineScale = o.refineScale !== undefined ? o.refineScale : 30.0;
    this.riskScale = 25.0;
  }

  unseenCounts() {
    const unseen = new Array(TILE_COUNT).fill(4);
    const me = this.game.players[this.seat];
    const mc = me.handCounts();
    for (let t = 0; t < TILE_COUNT; t++) unseen[t] -= mc[t];
    for (const q of this.game.players) {
      for (const t of q.discards) unseen[t] -= 1;
      for (const m of q.melds) unseen[m.tile] -= (m.type === "peng" ? 3 : 4);
    }
    return unseen.map(u => Math.max(0, u));
  }

  pengedByOthers() {
    const s = new Set();
    for (const q of this.game.players) {
      if (q.seat === this.seat) continue;
      for (const m of q.melds) if (m.type === "peng") s.add(m.tile);
    }
    return s;
  }

  endgameFactor() {
    const wall = this.game.wallRemaining();
    return Math.max(0, Math.min(1, (60 - wall) / 60));
  }

  sampleFuture(pool, wallSize, oppTotal) {
    const p = pool.slice();
    // Fisher-Yates
    for (let i = p.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      const tmp = p[i]; p[i] = p[j]; p[j] = tmp;
    }
    const wall = p.slice(oppTotal, oppTotal + wallSize);
    const draws = [];
    let idx = 3;
    while (idx < wall.length && draws.length < this.horizon) {
      draws.push(wall[idx]);
      idx += 4;
    }
    return draws;
  }

  chooseDiscard() {
    const p = this.game.players[this.seat];
    const counts14 = p.handCounts();
    const opts = discardOptions(counts14);
    if (!opts.length) return p.hand[p.hand.length - 1];

    const unseen = this.unseenCounts();
    const penged = this.pengedByOthers();
    const eg = this.endgameFactor();
    const riskW = this.riskScale * (1.0 + 1.5 * eg);
    const riskOf = (t) => {
      if (t === RED) return 0.0;
      if (penged.has(t)) return 1.0;
      const m = { 3: 0.4, 2: 0.2, 1: 0.05, 0: 0.0 };
      return m[unseen[t]] !== undefined ? m[unseen[t]] : 0.4;
    };

    // 1) 解析层(主导项)
    let minSh = Infinity;
    for (const o of opts) minSh = Math.min(minSh, o.shanten);
    const base = new Map();
    for (const o of opts) {
      const wr = o.waits.reduce((s, w) => s + unseen[w], 0);
      const shW = 100.0 * (1.0 - 0.5 * eg);
      base.set(o.tile, -shW * o.shanten + 3.0 * wr - riskW * riskOf(o.tile));
    }
    const argmaxBase = () => {
      let bt = null, bv = -Infinity;
      for (const [t, v] of base) if (v > bv) { bv = v; bt = t; }
      return bt;
    };

    // 2) 精修层: 只在最小向听候选间搜索
    const cands = opts.filter(o => o.shanten === minSh).map(o => o.tile);
    if (cands.length <= 1 || this.worlds <= 0) return argmaxBase();

    const pool = [];
    for (let t = 0; t < TILE_COUNT; t++) for (let i = 0; i < unseen[t]; i++) pool.push(t);
    let oppTotal = 0;
    for (const r of [1, 2, 3]) oppTotal += this.game.players[(this.seat + r) % 4].hand.length;
    const wallSize = Math.max(0, pool.length - oppTotal);
    if (wallSize < 4) return argmaxBase();

    const candSet = new Set(cands);
    const utilSum = new Map(cands.map(t => [t, 0.0]));
    let nWorlds = 0;
    for (let w = 0; w < this.worlds; w++) {
      const future = this.sampleFuture(pool, wallSize, oppTotal);
      if (!future.length) continue;
      const det = searchFirstDiscardDetail(counts14, future, this.beam, candSet);
      nWorlds++;
      const h = future.length;
      for (const [t, r] of det) {
        const u = (r.wd !== null) ? Math.pow(WIN_DISCOUNT, r.wd)
          : Math.pow(WIN_DISCOUNT, h + 2 * Math.max(0, r.sh) + 1);
        utilSum.set(t, (utilSum.get(t) || 0) + u);
      }
    }
    if (nWorlds === 0) return argmaxBase();

    let bestT = null, bestV = -Infinity;
    for (const t of cands) {
      const v = base.get(t) + this.refineScale * (utilSum.get(t) / nWorlds);
      if (v > bestV) { bestV = v; bestT = t; }
    }
    return bestT;
  }

  // 碰杠: v2 逻辑(比 v1 更谨慎: 听牌后碰须保持听牌)
  decidePeng(tile) {
    const counts = this.game.players[this.seat].handCounts();
    const before = shanten(counts);
    const c2 = counts.slice();
    c2[tile] -= 2;
    const after = shanten(c2);
    if (after < before) {
      if (before === 0) return after === 0;
      return true;
    }
    return false;
  }
}

/* ============ 挂哥: Oracle 作弊(直接读牌堆) ============ */
class BotOracle extends Bot {
  constructor(game, seat, opts) {
    super(game, seat);
    const o = opts || {};
    this.beam = o.beam !== undefined ? o.beam : 10;
  }

  /** 推算自己接下来会摸到的牌(假设无碰杠打断): wall[3], wall[7], ... */
  myFutureDraws(maxDraws) {
    const md = maxDraws || 12;
    const wall = this.game.wall;
    const draws = [];
    let idx = 3;
    while (idx < wall.length - 6 && draws.length < md) {
      draws.push(wall[idx]);
      idx += 4;
    }
    return draws;
  }

  chooseDiscard() {
    const p = this.game.players[this.seat];
    const counts14 = p.handCounts();
    const future = this.myFutureDraws();
    if (!future.length) return super.chooseDiscard();  // 牌堆见底, 退回菜鸟逻辑

    const det = searchFirstDiscardDetail(counts14, future, this.beam, null);
    const horizon = future.length;
    let bestT = null, bestU = -Infinity;
    for (const [t, r] of det) {
      const u = (r.wd !== null) ? Math.pow(WIN_DISCOUNT, r.wd)
        : noWinUtility(horizon, r.sh);
      if (u > bestU) { bestU = u; bestT = t; }
    }
    return bestT !== null ? bestT : super.chooseDiscard();
  }

  decidePeng(tile) {
    // 作弊视角: 碰之后仍能按未来摸牌最快胡才碰
    const counts = this.game.players[this.seat].handCounts();
    const before = shanten(counts);
    const c2 = counts.slice();
    c2[tile] -= 2;
    const after = shanten(c2);
    if (before === 0) return after === 0;
    return after < before;
  }
}
