/* 安康159 - 实时分析器(JS版)
 * 翻译自 backend/analysis/analyzer.py
 */

const N_159_TOTAL = 36;

class Analyzer {
  constructor(game, seat) {
    this.game = game;
    this.seat = seat;
  }

  visibleCounts() {
    const c = new Array(TILE_COUNT).fill(0);
    for (const p of this.game.players) {
      if (p.seat === this.seat) for (const t of p.hand) c[t]++;
      for (const t of p.discards) c[t]++;
      for (const m of p.melds) c[m.tile] += m.type === "peng" ? 3 : 4;
    }
    return c;
  }

  remainingCounts() {
    const vis = this.visibleCounts();
    return vis.map(n => 4 - n);
  }

  gangRisk(tile) {
    if (tile === RED) return 0.0;
    for (const p of this.game.players) {
      if (p.seat === this.seat) continue;
      for (const m of p.melds) {
        if (m.tile === tile && m.type === "peng") return 1.0;
      }
    }
    const remain = this.remainingCounts()[tile];
    let base;
    if (remain >= 3) base = 0.55;
    else if (remain === 2) base = 0.25;
    else if (remain === 1) base = 0.06;
    else base = 0.0;
    const progress = Math.max(0, Math.min(1, 1 - this.game.wallRemaining() / 60));
    return Math.round(base * (0.7 + 0.3 * progress) * 1000) / 1000;
  }

  expectedFan159() {
    const vis = this.visibleCounts();
    let seen159 = 0;
    for (let t = 0; t < 27; t++) if (is159(t)) seen159 += vis[t];
    const unseen159 = N_159_TOTAL - seen159;
    const wallN = this.game.wallRemaining();
    if (wallN <= 0) return 0;
    const othersHand = this.game.players
      .filter(p => p.seat !== this.seat)
      .reduce((s, p) => s + p.hand.length, 0);
    const unknownTotal = othersHand + wallN;
    if (unknownTotal <= 0) return 0;
    const kEst = unseen159 * wallN / unknownTotal;
    return Math.round(6 * kEst / wallN * 100) / 100;
  }

  expectedScoreIfWin() {
    return Math.round((this.expectedFan159() + 1) * 3 * 100) / 100;
  }

  opponentThreat(oppSeat) {
    const p = this.game.players[oppSeat];
    let score = 0;
    score += 0.15 * p.melds.length;
    const recent = p.discards.slice(-6);
    const mid = recent.filter(t => t < 27 && tileRank(t) >= 3 && tileRank(t) <= 7).length;
    score += 0.08 * mid;
    const progress = 1 - this.game.wallRemaining() / 60;
    score += 0.3 * Math.max(0, progress);
    return { seat: oppSeat, threat: Math.round(Math.min(score, 1) * 100) / 100 };
  }

  analyzeHand() {
    const p = this.game.players[this.seat];
    const counts = countsFromTiles(p.hand);
    const s = shanten(counts);
    const result = {
      shanten: s,
      is_ting: s === 0,
      waits: [],
      wait_count: 0,
      expected_fan159: this.expectedFan159(),
      expected_score_if_win: this.expectedScoreIfWin(),
      opponents: this.game.players
        .filter(o => o.seat !== this.seat)
        .map(o => this.opponentThreat(o.seat)),
    };
    if (s === 0) {
      const waits = waitingTiles(counts);
      const rem = this.remainingCounts();
      result.waits = waits.map(w => ({ tile: w, name: tileName(w), remain: rem[w] }));
      result.wait_count = waits.reduce((sum, w) => sum + rem[w], 0);
    }
    return result;
  }

  analyzeDiscards() {
    const p = this.game.players[this.seat];
    const counts = countsFromTiles(p.hand);
    const opts = discardOptions(counts);
    const rem = this.remainingCounts();
    const out = [];
    for (const o of opts) {
      const t = o.tile;
      const risk = this.gangRisk(t);
      const waitRemains = o.waits.reduce((sum, w) => sum + rem[w], 0);
      const score = -100 * o.shanten + 3 * waitRemains - 30 * risk;
      out.push({
        tile: t,
        name: tileName(t),
        shanten: o.shanten,
        waits: o.waits.map(w => ({ tile: w, name: tileName(w), remain: rem[w] })),
        wait_remain: waitRemains,
        gang_risk: risk,
        score: Math.round(score * 10) / 10,
      });
    }
    out.sort((a, b) => b.score - a.score);
    return out;
  }
}
