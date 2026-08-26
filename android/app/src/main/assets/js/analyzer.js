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
    const nMelds = p.melds.length;
    // 副露感知: 直接对副露后的 11/10 张暗牌调 shanten() 会高估约 2*副露数,
    // 与 backend/analysis/analyzer.py 保持一致
    const s = shantenWithMelds(counts, nMelds);
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

  /**
   * 有效进张(广义进张): 听牌时=听口; 未听牌时=能降低向听的进张。
   *
   * 修复旧缺陷: 旧版进张取自 discardOptions 的 waits, 而 waits 仅在向听=0
   * 时非空, 导致非听牌局面下所有候选牌的进张项恒为 0, 打分退化成
   * "-100*向听 - 30*风险", 唯一区分依据变成放杠风险。而风险按"外面还剩几
   * 张"估算: 手里持一对 -> 外面剩 2 张 -> 风险 0.25; 单张 -> 外面剩 3 张 -> 风险 0.55,
   * 于是系统性地推荐"拆对子/打将"。实测 386 个非听牌局面中 72.5% 如此。
   *
   * 修法与后端 v10/v31 同口径: 任意向听下都计算有效进张。
   */
  effectiveDraws(counts, nMelds, s) {
    const out = [];
    for (let w = 0; w < TILE_COUNT; w++) {
      if (counts[w] >= 4) continue;
      counts[w]++;
      const ok = (s === 0) ? isWin(counts)
        : shantenWithMelds(counts, nMelds) < s;
      counts[w]--;
      if (ok) out.push(w);
    }
    return out;
  }

  analyzeDiscards() {
    const p = this.game.players[this.seat];
    const counts = p.handCounts();
    const nMelds = p.melds.length;
    const rem = this.remainingCounts();
    const out = [];
    for (let t = 0; t < TILE_COUNT; t++) {
      if (counts[t] <= 0) continue;
      counts[t]--;
      const s = shantenWithMelds(counts, nMelds);
      const waits = s === 0 ? waitingTiles(counts) : [];
      const draws = this.effectiveDraws(counts, nMelds, s);
      counts[t]++;
      const risk = this.gangRisk(t);
      const waitRemains = waits.reduce((sum, w) => sum + rem[w], 0);
      const ukeire = draws.reduce((sum, w) => sum + rem[w], 0);
      // 综合分: 向听小优先 >> 有效进张多优先 > 放杠风险低优先
      // 风险权重从 30 降到 10: 避免风险项反过来压过牌效
      const score = -100 * s + 3 * ukeire - 10 * risk;
      out.push({
        tile: t,
        name: tileName(t),
        shanten: s,
        waits: waits.map(w => ({ tile: w, name: tileName(w), remain: rem[w] })),
        wait_remain: waitRemains,
        ukeire,
        gang_risk: risk,
        score: Math.round(score * 10) / 10,
      });
    }
    out.sort((a, b) => b.score - a.score);
    return out;
  }
}
