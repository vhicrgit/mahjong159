/* 安康159 - 学者 Bot (HV, JS版): 用牌型价值分析器的期望巡数做全部决策
 *
 * 逐行移植自 backend/ai/bot_hv.py, 修改时必须同步两边。
 *
 * - 出牌: 打出后期望巡数 E 最小的牌(分析器口径: 自摸+碰通道, rho=1, 无换型层)
 * - 碰:   碰后最优牌型的 E 严格小于当前 E 才碰
 * - 杠:   与 v1/v10/v31 同口径(不破坏已成听口才杠)
 *
 * 性能取舍: 换型层(kaizen)把单手分析从毫秒级拉到秒级, 只改善绝对值不改善排序,
 * 所以对战按 kaizen=false 跑。
 *
 * 强度参考(后端实测, logs/eval_hv.log):
 *   HV  vs 3xv1   胜率 32.6% (+-2.7)  场均 +0.76
 *   v31 vs 3xv1   胜率 31.7% (+-2.6)  场均 +0.82
 *   配对检验 McNemar p=0.5247 -> 与老鸟 v31 强度无统计显著差异, 是另一种风格
 *
 * 依赖: tiles.js, win.js(shanten/isWin), hand_value.js(HandAnalyzer), bot.js(Bot)
 */

class BotHV extends Bot {
  constructor(game, seat, opts) {
    super(game, seat);
    const o = opts || {};
    this.rho = o.rho !== undefined ? o.rho : 1.0;
  }

  _analyzer() {
    const visible = new Array(28).fill(0);
    for (const q of this.game.players) {
      for (const t of q.discards) visible[t]++;
      for (const m of q.melds) visible[m.tile] += m.type === "peng" ? 3 : 4;
    }
    const hc = this.game.players[this.seat].handCounts();
    for (let t = 0; t < 28; t++) visible[t] += hc[t];
    return new HandAnalyzer(hc, visible, { rho: this.rho, kaizen: false });
  }

  chooseDiscard() {
    const p = this.game.players[this.seat];
    // wasm 快路径: 整个 E 递归在 C 里跑, 一次决策只跨界一次(实测快 7-66 倍)
    const w = MJWasm.hvChooseDiscard(this.game, this.seat, this.rho);
    if (w !== null) return w;
    const hand = p.handCounts();
    const az = this._analyzer();
    let bestT = null, bestE = 1e18;
    for (let t = 0; t < 28; t++) {
      if (hand[t] <= 0) continue;
      const h = hand.slice();
      h[t]--;
      const e = az.E(h, az.u0);
      if (e < bestE) { bestE = e; bestT = t; }
    }
    return bestT !== null ? bestT : p.hand[p.hand.length - 1];
  }

  decidePeng(tile) {
    const w = MJWasm.hvDecidePeng(this.game, this.seat, this.rho, tile);
    if (w !== null) return w;
    const p = this.game.players[this.seat];
    const hand = p.handCounts();
    const az = this._analyzer();
    const eBefore = az.E(hand, az.u0);
    const h2 = hand.slice();
    h2[tile] -= 2;
    let bestAfter = 1e18;
    for (let d = 0; d < 28; d++) {
      if (h2[d] <= 0) continue;
      const h3 = h2.slice();
      h3[d]--;
      const e = az.E(h3, az.u0);
      if (e < bestAfter) bestAfter = e;
    }
    return bestAfter < eBefore;
  }

  decideGang(tile, kind) {
    const w = MJWasm.hvDecideGang(this.game, this.seat, this.rho, tile, kind);
    if (w !== null) return w;
    const p = this.game.players[this.seat];
    const counts = p.handCounts();
    // shanten 已按手牌张数推导面子需求, 原生支持副露手, 直接调即可
    const before = shanten(counts);
    const c = counts.slice();
    if (kind === "ming") c[tile] -= 3;
    else if (kind === "an") c[tile] -= 4;
    else c[tile] -= 1;
    const after = shanten(c);
    return !(before === 0 && after > 0);
  }
}
