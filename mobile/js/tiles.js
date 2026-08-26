/* 安康159 - 牌定义(JS版)
 * 牌id: 0-8条, 9-17饼, 18-26万, 27红中
 */

const TIAO = 0, BING = 1, WAN = 2, HZ = 3;
const RED = 27;
const TILE_COUNT = 28;
const SUIT_NAMES = ["条", "饼", "万"];

function tileSuit(t) {
  if (t < 9) return TIAO;
  if (t < 18) return BING;
  if (t < 27) return WAN;
  return HZ;
}

function tileRank(t) {
  return t >= 27 ? 0 : (t % 9) + 1;
}

function isSuitTile(t) { return t < 27; }

function is159(t) {
  if (t >= 27) return false;
  const r = t % 9;
  return r === 0 || r === 4 || r === 8;
}

function tileName(t) {
  if (t === RED) return "红中";
  return tileRank(t) + SUIT_NAMES[tileSuit(t)];
}

function buildWall() {
  const wall = [];
  for (let t = 0; t < 27; t++) for (let i = 0; i < 4; i++) wall.push(t);
  for (let i = 0; i < 4; i++) wall.push(RED);
  return wall;
}

function countsFromTiles(tiles) {
  const c = new Array(TILE_COUNT).fill(0);
  for (const t of tiles) c[t]++;
  return c;
}
