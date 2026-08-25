/* 安康159 - 规则AI对手(JS版)
 * 翻译自 backend/ai/bot.py
 */

class Bot {
  constructor(game, seat) {
    this.game = game;
    this.seat = seat;
  }

  chooseDiscard() {
    const p = this.game.players[this.seat];
    const counts = p.handCounts();
    const opts = discardOptions(counts);
    if (!opts.length) return p.hand[p.hand.length - 1];

    // 可见计数
    const visible = new Array(28).fill(0);
    for (const q of this.game.players) {
      for (const t of q.discards) visible[t]++;
      for (const m of q.melds) visible[m.tile] += m.type === "peng" ? 3 : 4;
    }
    for (const t of p.hand) visible[t]++;

    let bestTile = null, bestScore = -1e9;
    for (const o of opts) {
      const t = o.tile;
      const wr = o.waits.reduce((sum, w) => sum + Math.max(0, 4 - visible[w]), 0);
      let risk = 0;
      if (t !== RED) {
        const remain = 4 - visible[t];
        let someonePeng = false;
        for (const q of this.game.players) {
          if (q.seat !== this.seat && q.melds.some(m => m.tile === t && m.type === "peng")) {
            someonePeng = true;
            break;
          }
        }
        if (someonePeng) risk = 1.0;
        else risk = { 3: 0.4, 2: 0.2, 1: 0.05, 0: 0.0 }[Math.max(0, remain)] ?? 0.4;
      }
      const score = -100 * o.shanten + 3 * wr - 25 * risk;
      if (score > bestScore) { bestScore = score; bestTile = t; }
    }
    return bestTile;
  }

  decidePeng(tile) {
    const p = this.game.players[this.seat];
    const counts = p.handCounts();
    const before = shanten(counts);
    const c2 = counts.slice();
    c2[tile] -= 2;
    return shanten(c2) < before;
  }

  decideGang(tile, kind) {
    const p = this.game.players[this.seat];
    const counts = p.handCounts();
    const sBefore = shanten(counts);
    const c2 = counts.slice();
    if (kind === "ming") c2[tile] -= 3;
    else if (kind === "an") c2[tile] -= 4;
    else c2[tile] -= 1;
    const sAfter = shanten(c2);
    if (sBefore === 0 && sAfter > 0) return false;
    return true;
  }
}
