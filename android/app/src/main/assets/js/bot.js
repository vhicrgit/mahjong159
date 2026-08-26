/* 安康159 - 规则AI对手(菜鸟 v1, JS版)
 * 翻译自 backend/ai/bot_v1.py, 修改时必须同步两边
 *
 * decidePeng/decideGang 使用副露感知向听(shantenWithMelds):
 * 直接对碰后的 11 张暗牌调 shanten() 会高估约 2*副露数,
 * 导致"碰后向听降低"几乎永远不成立, Bot 从不鸣牌。
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
    const nMelds = p.melds.length;
    const counts = p.handCounts();
    const before = shantenWithMelds(counts, nMelds);
    // 碰后: 暗牌-2, 副露+1, 且必须再打出一张
    const c11 = counts.slice();
    c11[tile] -= 2;
    let after = 99;
    for (let d = 0; d < TILE_COUNT; d++) {
      if (c11[d] <= 0) continue;
      c11[d] -= 1;
      const s = shantenWithMelds(c11, nMelds + 1);
      c11[d] += 1;
      if (s < after) after = s;
    }
    return after < before;
  }

  decideGang(tile, kind) {
    const p = this.game.players[this.seat];
    const nMelds = p.melds.length;
    const counts = p.handCounts();
    const sBefore = shantenWithMelds(counts, nMelds);
    const c2 = counts.slice();
    let nAfter;
    if (kind === "ming") { c2[tile] -= 3; nAfter = nMelds + 1; }
    else if (kind === "an") { c2[tile] -= 4; nAfter = nMelds + 1; }
    else { c2[tile] -= 1; nAfter = nMelds; }   // bu: 碰转杠, 副露数不变
    const sAfter = shantenWithMelds(c2, nAfter);
    if (sBefore === 0 && sAfter > 0) return false;
    return true;
  }
}
