/* 安康159 - 听牌/进张分析(JS版)
 * 翻译自 backend/rules/ting.py
 */

function waitingTiles(counts13) {
  const waits = [];
  for (let t = 0; t < TILE_COUNT; t++) {
    if (counts13[t] >= 4) continue;
    counts13[t]++;
    if (isWin(counts13)) waits.push(t);
    counts13[t]--;
  }
  return waits;
}

function isTing(counts13) {
  return waitingTiles(counts13).length > 0;
}

function discardOptions(counts14) {
  const options = [];
  for (let t = 0; t < TILE_COUNT; t++) {
    if (counts14[t] <= 0) continue;
    counts14[t]--;
    const s = shanten(counts14);
    const waits = s === 0 ? waitingTiles(counts14) : [];
    const waitCount = waits.reduce((sum, w) => sum + (4 - counts14[w]), 0);
    options.push({ tile: t, shanten: s, waits, wait_count: waitCount });
    counts14[t]++;
  }
  return options;
}
