/* 安康159麻将 - 手机版交互逻辑(增强版)
 * 音效 + SVG牌面 + 拖拽排序 + 出牌动画 + 累计积分
 */

let game = null;
let selectedUid = -1;         // 选中的手牌唯一id
let lastAnalysis = null;
let displayHand = [];         // [{uid, tile}] 显示手牌(可拖拽排序)
let uidCounter = 1;
let lastLogLen = 0;           // 用于音效: 跟踪新增日志
const AI_DELAY = 200;   // AI 回合间隔(搜索本身也需时间, 不必额外等待太久)
const SEAT_NAMES = ["我", "下家", "对家", "上家"];

// 可选 AI 档位: kind -> {昵称, 说明, 工厂函数}
// 档位名与 backend/ai/roster.py 的 KIND_INFO 对齐, 便于两端对照。
// 桌面版还有 v10(中鸟) / target(目标) / cheat_opp(挂王) / cheat_full(神挂),
// 对应的 bot_v10.py / bot_target.py / bot_cheat.py 尚未移植到 JS, 所以不列。
// v4 是手机版独有的(解析+PIMC), 后端无对应档位。
const BOT_KINDS = {
  v1:         { nick: "菜鸟", desc: "规则Bot v1: 牌效优先",
                make: (g, s) => new Bot(g, s) },
  v4:         { nick: "老手", desc: "规则Bot v4: 解析+PIMC精修",
                make: (g, s) => new BotV4(g, s, { worlds: 24, beam: 6, horizon: 6 }) },
  v31:        { nick: "老鸟", desc: "规则Bot v31: 两步推演+副露感知",
                make: (g, s) => new BotV31(g, s) },
  scholar:    { nick: "学者", desc: "牌型价值: 期望胡牌巡数",
                make: (g, s) => new BotHV(g, s) },
  cheat_wall: { nick: "挂哥", desc: "作弊Bot: 可见牌堆",
                make: (g, s) => new BotOracle(g, s, { beam: 12 }) },
};

// 默认阵容: 与 backend/ai/roster.py 的 ROSTER 保持一致
const DEFAULT_KINDS = { 1: "v1", 2: "v31", 3: "cheat_wall" };
const KINDS_STORE_KEY = "mj159_seat_kinds";

// 当前生效的各席档位(持久化到 localStorage)
let seatKinds = loadSeatKinds();

function loadSeatKinds() {
  const out = Object.assign({}, DEFAULT_KINDS);
  try {
    const raw = localStorage.getItem(KINDS_STORE_KEY);
    if (raw) {
      const saved = JSON.parse(raw);
      for (const s of [1, 2, 3]) {
        if (BOT_KINDS[saved[s]]) out[s] = saved[s];   // 档位名可能已下线, 校验后再用
      }
    }
  } catch (e) { /* localStorage 不可用时退回默认 */ }
  return out;
}

function saveSeatKinds() {
  try { localStorage.setItem(KINDS_STORE_KEY, JSON.stringify(seatKinds)); }
  catch (e) { /* 忽略 */ }
}

function seatKindInfo(seat) {
  return BOT_KINDS[seatKinds[seat]] || BOT_KINDS[DEFAULT_KINDS[seat]];
}

let aiBots = {};   // seat -> bot 实例(每局初始化)

function seatNick(seat) {
  return (seat >= 1 && seat <= 3) ? seatKindInfo(seat).nick : "我";
}

// 显示某家"思考中"(同步搜索前先给反馈, 避免看起来卡死)
function showThinking(seat) {
  const el = document.getElementById("status");
  if (el) el.innerHTML = `<span class='thinking'>${seatLabel(seat)} 思考中...</span>`;
  const box = document.getElementById("opp-" + seat);
  if (box) box.classList.add("active");
}

// 显示用: "下家 菜鸟" / "我"
function seatLabel(seat) {
  if (seat === 0) return "我";
  return SEAT_NAMES[seat] + " " + seatNick(seat);
}

// ---------- 累计积分 ----------
function getTotalScores() {
  try { return JSON.parse(localStorage.getItem("mj159_total") || "[0,0,0,0]"); }
  catch (e) { return [0, 0, 0, 0]; }
}
function addTotalScores(deltas) {
  const t = getTotalScores();
  for (let i = 0; i < 4; i++) t[i] += deltas[i];
  localStorage.setItem("mj159_total", JSON.stringify(t));
  return t;
}
function resetTotalScores() {
  localStorage.setItem("mj159_total", JSON.stringify([0, 0, 0, 0]));
}

// ---------- 工具 ----------
function tileInfo(t) {
  if (t === RED) return { suit: -1, name: "红中" };
  const suit = Math.floor(t / 9), num = (t % 9) + 1;
  return { suit, name: num + SUIT_NAMES[suit] };
}

function makeTileEl(t, size, clickable, uid) {
  const el = document.createElement("div");
  el.className = "tile face" + (size ? " " + size : "");
  el.innerHTML = TileArt.svg(t, undefined, undefined);
  if (clickable) {
    el.dataset.uid = uid;
    el.dataset.tile = t;
    attachTileHandlers(el);
  }
  return el;
}

function makeBackEl(size) {
  const el = document.createElement("div");
  el.className = "tile back" + (size ? " " + size : "");
  el.innerHTML = TileArt.svg(-1);
  return el;
}

function makeMeldEl(m, size) {
  const wrap = document.createElement("div");
  wrap.className = "meld";
  const n = m.type === "peng" ? 3 : 4;
  for (let i = 0; i < n; i++) wrap.appendChild(makeTileEl(m.tile, size));
  return wrap;
}

// ---------- 手牌同步(逻辑手牌 -> 显示手牌) ----------
function syncDisplayHand() {
  const logical = game.players[0].hand;  // 引擎手牌(有序)
  // 多重集合计数
  const logicalCount = {};
  logical.forEach(t => logicalCount[t] = (logicalCount[t] || 0) + 1);
  // 移除 displayHand 中已不在逻辑手牌里的牌
  displayHand = displayHand.filter(c => {
    if (logicalCount[c.tile] > 0) { logicalCount[c.tile]--; return true; }
    return false;
  });
  // 添加新摸到的牌(logicalCount 里剩余的), 插入到适当位置(不整体重排, 保留手动排序)
  const newTiles = [];
  Object.keys(logicalCount).forEach(k => {
    const t = Number(k);
    for (let i = 0; i < logicalCount[k]; i++) newTiles.push(t);
  });
  for (const t of newTiles) {
    insertTileSorted(t);
  }
}

// 把新牌插入到 displayHand 中合适的位置(第一个比它大的牌前面), 不打乱已有顺序
function insertTileSorted(tile) {
  let idx = displayHand.length;
  for (let i = 0; i < displayHand.length; i++) {
    if (displayHand[i].tile > tile) { idx = i; break; }
  }
  displayHand.splice(idx, 0, { uid: uidCounter++, tile });
}

// 整理牌: 整个手牌按值排序
function sortHand() {
  displayHand.sort((a, b) => a.tile - b.tile);
  selectedUid = -1;
  SoundFX.click();
  renderMyHand();
}

// ---------- 音效(根据新增日志) ----------
function playSoundsFromLog() {
  if (!game) return;
  const logs = game.log.slice(lastLogLen);
  lastLogLen = game.log.length;
  for (const line of logs) {
    if (line.includes("胡牌")) {
      const m = line.match(/座位(\d)/);
      if (m && Number(m[1]) === 0) SoundFX.win(); else SoundFX.lose();
    } else if (line.includes("黄庄")) {
      SoundFX.huang();
    } else if (line.includes("杠")) {
      SoundFX.gang();
    } else if (line.includes("碰")) {
      SoundFX.peng();
    } else if (line.includes("打出")) {
      SoundFX.discard();
    } else if (line.includes("摸牌") && line.includes("座位0")) {
      SoundFX.draw();
    }
  }
}

// ---------- 游戏流程 ----------
function newGame() {
  selectedUid = -1;
  lastAnalysis = null;
  displayHand = [];
  uidCounter = 1;
  lastLogLen = 0;
  document.getElementById("result-overlay").classList.add("hidden");
  document.getElementById("review-panel").classList.add("hidden");
  document.getElementById("analysis-panel").classList.add("hidden");
  document.getElementById("setup-overlay").classList.add("hidden");
  game = new Game(0);
  // 为每个 AI 座位按选定档位创建 bot
  aiBots = {};
  for (const s of [1, 2, 3]) aiBots[s] = seatKindInfo(s).make(game, s);
  updateSeatLabels();
  render();
  refreshAnalysis();
  maybeRunAI();
}

// 弃牌区标题原本是 HTML 里写死的"上家 挂哥"等, 选档后会与实际对手不符
function updateSeatLabels() {
  for (const s of [1, 2, 3]) {
    const el = document.querySelector(`#discard-${s} .dlabel`);
    if (el) el.textContent = `${SEAT_NAMES[s]} ${seatNick(s)}`;
  }
}

/* ---------- 开局选档面板 ---------- */

function openSetup() {
  const rows = document.getElementById("setup-rows");
  rows.innerHTML = "";
  for (const s of [3, 2, 1]) {          // 上家/对家/下家, 与牌桌布局同序
    const row = document.createElement("div");
    row.className = "setup-row";
    let opts = "";
    for (const [kind, info] of Object.entries(BOT_KINDS)) {
      const sel = seatKinds[s] === kind ? " selected" : "";
      opts += `<option value="${kind}"${sel}>${info.nick} - ${info.desc}</option>`;
    }
    row.innerHTML = `<span class="setup-seat">${SEAT_NAMES[s]}</span>`
      + `<select class="setup-sel" data-seat="${s}">${opts}</select>`;
    rows.appendChild(row);
  }
  // 提示区: 现在没什么要提醒的就置空(CSS 会把空元素隐掉)。
  // 原本这里写的是"学者档单次决策需 2-5 秒", 接入 wasm 后降到百毫秒级, 已不适用。
  document.getElementById("setup-note").textContent = "";
  document.getElementById("setup-overlay").classList.remove("hidden");
}

function applySetupAndStart() {
  for (const sel of document.querySelectorAll(".setup-sel")) {
    const s = Number(sel.dataset.seat);
    if (BOT_KINDS[sel.value]) seatKinds[s] = sel.value;
  }
  saveSeatKinds();
  newGame();
}

function maybeRunAI() {
  if (!game || game.phase === "game_over") { render(); return; }
  if (game.phase === "discard_wait" && game.turn === 0) { render(); return; }
  if (game.phase === "react_wait" && game.pendingActions[0]) { render(); return; }

  // 先把"思考中"画出来(setTimeout 让浏览器先重绘), 再做同步搜索
  const actor = (game.phase === "discard_wait") ? game.turn
    : Object.keys(game.pendingActions).map(Number).filter(s => s !== 0)[0];
  if (actor !== undefined) showThinking(actor);

  setTimeout(() => {
    if (!game || game.phase === "game_over") { render(); return; }
    if (game.phase === "discard_wait" && game.turn !== 0) {
      const bot = aiBots[game.turn];
      try { game.discard(game.turn, bot.chooseDiscard()); } catch (e) { console.error(e); }
    } else if (game.phase === "react_wait") {
      const seats = Object.keys(game.pendingActions).map(Number).filter(s => s !== 0);
      if (seats.length) {
        const s = seats[0];
        const bot = aiBots[s];
        const tile = game.lastDiscard;
        try {
          if (game.pendingActions[s].gang && bot.decideGang(tile, "ming")) game.gang(s);
          else if (game.pendingActions[s].peng && bot.decidePeng(tile)) game.peng(s);
          else game.pass(s);
        } catch (e) { console.error(e); }
      }
    }
    render();
    refreshAnalysis();
    maybeRunAI();
  }, AI_DELAY);
}

// ---------- 渲染 ----------
function render() {
  if (!game) return;
  syncDisplayHand();
  document.getElementById("wall-info").textContent = "剩" + game.wallRemaining();

  // 对手(紧凑: 昵称 + 剩余张数 + 副露)
  for (let s = 1; s <= 3; s++) {
    const p = game.players[s];
    const box = document.getElementById("opp-" + s);
    box.querySelector(".opp-nick").textContent = seatNick(s);
    box.querySelector(".opp-count").textContent = p.hand.length + "张";
    const melds = box.querySelector(".opp-melds");
    melds.innerHTML = "";
    for (const m of p.melds) melds.appendChild(makeMeldEl(m, "sm"));
    // 当前行动者高亮
    box.classList.toggle("active",
      game.phase !== "game_over" && game.turn === s);
  }

  // 弃牌(最后一张高亮 + 飞入动画方向)
  const dirMap = { 0: "bottom", 1: "right", 2: "top", 3: "left" };
  for (let s = 0; s <= 3; s++) {
    const area = document.querySelector("#discard-" + s + " .discards");
    area.innerHTML = "";
    const ds = game.players[s].discards;
    ds.forEach((t, i) => {
      const el = makeTileEl(t, "sm");
      if (i === ds.length - 1 && game.lastDiscarder === s) {
        el.classList.add("latest", "fly-" + dirMap[s]);
      }
      area.appendChild(el);
    });
  }

  renderMyHand();

  // 副露
  const mm = document.getElementById("my-melds");
  mm.innerHTML = "";
  for (const m of game.players[0].melds) mm.appendChild(makeMeldEl(m, "md"));

  // 状态
  const statusEl = document.getElementById("status");
  if (game.phase === "game_over") {
    statusEl.innerHTML = "";
    showResult();
  } else if (game.phase === "discard_wait" && game.turn === 0) {
    statusEl.innerHTML = "轮到你出牌 <span class='hint-dim'>" + (game.lastAction || "") + "</span>";
  } else if (game.phase === "react_wait" && game.pendingActions[0]) {
    const who = seatLabel(game.lastDiscarder);
    statusEl.innerHTML = `${who} 打出 <b class='hl-tile'>${tileInfo(game.lastDiscard).name}</b>, 你可碰/杠/过`;
  } else {
    statusEl.innerHTML = "对手行动中... <span class='hint-dim'>" + (game.lastAction || "") + "</span>";
  }

  updateActionButtons();
  updateScoreBar();
  playSoundsFromLog();
}

function renderMyHand() {
  const myHand = document.getElementById("my-hand");
  myHand.innerHTML = "";
  // 「荐」的来源: 分析面板打开且 wasm 就绪时用 HandAnalyzer 的期望巡数口径,
  // 否则用 analyzer.js 的启发式排序首项。
  const recTile = (lastAnalysis && lastAnalysis.rec_hv !== undefined)
    ? lastAnalysis.rec_hv
    : ((lastAnalysis && lastAnalysis.discards && lastAnalysis.discards.length)
        ? lastAnalysis.discards[0].tile : null);
  const drawnTile = (game.lastDrawn && game.lastDrawn.seat === 0) ? game.lastDrawn.tile : null;
  // 「荐」只标一张: 推荐结果是牌值, 若按牌值标记, 手里一对会两张都亮,
  // 看上去像在建议"把这对打掉"。与选中逻辑保持一致: 按索引只标首张。
  let recMarked = false;
  for (const card of displayHand) {
    const el = makeTileEl(card.tile, "lg", true, card.uid);
    if (card.uid === selectedUid) el.classList.add("selected");
    if (!recMarked && recTile !== null && card.tile === recTile) {
      el.classList.add("recommended");
      recMarked = true;
    }
    if (drawnTile !== null && card.tile === drawnTile) el.classList.add("drawn");
    myHand.appendChild(el);
  }
  updateTingHint();
}

// 把听口渲染成小号牌面图(而不是"3饼 6饼"这种文字), 与牌桌视觉统一
function waitTilesHTML(tiles) {
  let h = "";
  for (const t of tiles) {
    h += `<span class="wait-tile" title="${tileName(t)}">${TileArt.svg(t)}</span>`;
  }
  return h;
}

// 听牌提示
function updateTingHint() {
  let hintEl = document.getElementById("ting-hint");
  if (!hintEl) {
    hintEl = document.createElement("div");
    hintEl.id = "ting-hint";
    document.getElementById("my-area").prepend(hintEl);
  }
  if (!game || game.phase === "game_over") { hintEl.innerHTML = ""; return; }
  const p = game.players[0];
  const counts = p.handCounts();
  const nMelds = p.melds.length;

  // 预览: 选中某张牌时, 先算"打出它之后"是否听牌 —— 不用真的打出去试。
  // 选中态下优先显示预览, 方便逐张比较哪张能听、听得宽。
  if (selectedUid >= 0 && game.phase === "discard_wait" && game.turn === 0) {
    const card = displayHand.find(c => c.uid === selectedUid);
    if (card && counts[card.tile] > 0) {
      const after = counts.slice();
      after[card.tile]--;
      const sAfter = shantenWithMelds(after, nMelds);
      if (sAfter === 0) {
        const waits = waitingTiles(after);
        hintEl.innerHTML = `<span class='ting-preview'>打 ${tileName(card.tile)} 则听:</span>`
          + waitTilesHTML(waits);
        return;
      }
      if (sAfter > 0) {
        hintEl.innerHTML = `<span class='ting-preview-no'>打 ${tileName(card.tile)} 后向听数: ${sAfter}</span>`;
        return;
      }
    }
  }

  // 必须用副露感知向听: 碰/杠后暗牌变短(11/10张), 直接调 shanten() 会把
  // 向听高估约 2*副露数, 导致"有副露时已经听牌了却不提示"
  // (实测: 1个碰 + 暗牌听牌, 旧逻辑算出 2, 正确值是 0)
  const s = shantenWithMelds(counts, nMelds);
  if (s === 0 && p.hand.length % 3 === 1) {
    hintEl.innerHTML = `<span class='ting-ok'>已听牌, 听:</span>`
      + waitTilesHTML(waitingTiles(counts));
  } else if (s >= 0) {
    hintEl.innerHTML = `<span class='ting-no'>向听数: ${s}</span>`;
  } else {
    hintEl.innerHTML = "";
  }
}

// 累计积分条
function updateScoreBar() {
  let bar = document.getElementById("score-bar");
  if (!bar) {
    bar = document.createElement("div");
    bar.id = "score-bar";
    document.getElementById("topbar").appendChild(bar);
  }
  const t = getTotalScores();
  bar.innerHTML = `累计积分 我:${t[0] > 0 ? "+" : ""}${t[0]} <span class='reset-score' id='reset-score'>清零</span>`;
  const rst = document.getElementById("reset-score");
  if (rst) rst.onclick = (e) => { e.stopPropagation(); resetTotalScores(); updateScoreBar(); };
}

function updateActionButtons() {
  const bp = document.getElementById("btn-peng");
  const bg = document.getElementById("btn-gang");
  const bs = document.getElementById("btn-pass");
  const actions = document.getElementById("actions");
  const gc = document.getElementById("gang-choices");
  bp.classList.add("hidden"); bg.classList.add("hidden"); bs.classList.add("hidden");
  actions.classList.add("hidden"); gc.classList.add("hidden"); gc.innerHTML = "";

  if (!game || game.phase === "game_over") return;

  if (game.phase === "react_wait" && game.pendingActions[0]) {
    const pa = game.pendingActions[0];
    actions.classList.remove("hidden");
    if (pa.peng) bp.classList.remove("hidden");
    if (pa.gang) bg.classList.remove("hidden");
    bs.classList.remove("hidden");
  }
  if (game.phase === "discard_wait" && game.turn === 0) {
    const gopts = game.gangOptions(0);
    if (gopts.length) {
      gc.classList.remove("hidden");
      // 加文字提示: 以前只摆一张光秃的浮空牌, 看不出那是可点的开杠按钮
      const tip = document.createElement("span");
      tip.className = "gang-tip";
      tip.textContent = "可杠(点牌开杠):";
      gc.appendChild(tip);
      for (const t of gopts) {
        const p0 = game.players[0];
        const kind = p0.hand.filter(x => x === t).length === 4 ? "暗杠" : "补杠";
        const wrap = document.createElement("div");
        wrap.className = "gang-choice";
        const el = makeTileEl(t, "md");
        wrap.appendChild(el);
        const lb = document.createElement("span");
        lb.className = "gang-kind";
        lb.textContent = kind;
        wrap.appendChild(lb);
        wrap.onclick = () => doGangTile(t);
        gc.appendChild(wrap);
      }
    }
  }
}

// ---------- 交互: 点击选中/出牌 ----------
function attachTileHandlers(el) {
  // 点击(点按)出牌
  el.addEventListener("click", (e) => {
    if (dragMoved) return;  // 刚拖完不触发点击
    const uid = Number(el.dataset.uid);
    const tile = Number(el.dataset.tile);
    onTileClick(tile, uid);
  });
  // 拖拽排序
  el.addEventListener("pointerdown", onDragStart);
}

function onTileClick(tile, uid) {
  if (!game) return;
  if (game.phase === "discard_wait" && game.turn === 0) {
    if (selectedUid === uid) doDiscard(tile, uid);
    else { selectedUid = uid; renderMyHand(); }
  }
}

function doDiscard(tile, uid) {
  selectedUid = -1;
  recordDecision(tile);
  try { game.discard(0, tile); } catch (e) { alert(e.message); }
  render();
  refreshAnalysis();
  maybeRunAI();
}

function doPeng() {
  try { game.peng(0); SoundFX.peng(); } catch (e) { alert(e.message); }
  render(); refreshAnalysis();
}
function doGang() {
  try { game.gang(0); SoundFX.gang(); } catch (e) { alert(e.message); }
  render(); refreshAnalysis(); maybeRunAI();
}
function doGangTile(t) {
  try { game.gang(0, t); SoundFX.gang(); } catch (e) { alert(e.message); }
  render(); refreshAnalysis(); maybeRunAI();
}
function doPass() {
  try { game.pass(0); } catch (e) { alert(e.message); }
  render(); refreshAnalysis(); maybeRunAI();
}

// ---------- 手牌拖拽排序 ----------
let dragState = null;
let dragMoved = false;

function onDragStart(e) {
  if (!game || game.phase === "game_over") return;
  const el = e.currentTarget;
  dragState = {
    el,
    uid: Number(el.dataset.uid),
    startX: e.clientX,
    startIndex: displayHand.findIndex(c => c.uid === Number(el.dataset.uid)),
  };
  dragMoved = false;
  el.setPointerCapture(e.pointerId);
  el.classList.add("dragging");
  el.addEventListener("pointermove", onDragMove);
  el.addEventListener("pointerup", onDragEnd, { once: true });
  el.addEventListener("pointercancel", onDragEnd, { once: true });
}

function onDragMove(e) {
  if (!dragState) return;
  const dx = e.clientX - dragState.startX;
  if (Math.abs(dx) > 6) dragMoved = true;
  dragState.el.style.transform = `translateX(${dx}px) translateY(-14px)`;
  dragState.el.style.zIndex = 50;
}

function onDragEnd(e) {
  if (!dragState) return;
  const el = dragState.el;
  el.removeEventListener("pointermove", onDragMove);
  el.classList.remove("dragging");
  el.style.transform = "";
  el.style.zIndex = "";

  if (dragMoved) {
    // 根据落点 x 计算目标位置
    const myHand = document.getElementById("my-hand");
    const tiles = [...myHand.querySelectorAll(".tile")];
    const dropX = e.clientX;
    let targetIndex = displayHand.length - 1;
    for (let i = 0; i < tiles.length; i++) {
      const r = tiles[i].getBoundingClientRect();
      if (dropX < r.left + r.width / 2) { targetIndex = i; break; }
    }
    // 重排 displayHand
    const card = displayHand.splice(dragState.startIndex, 1)[0];
    if (card) {
      let insertAt = targetIndex;
      if (targetIndex > dragState.startIndex) insertAt = targetIndex - 1;
      displayHand.splice(Math.max(0, Math.min(insertAt, displayHand.length)), 0, card);
    }
    SoundFX.click();
    selectedUid = -1;
  }
  dragState = null;
  // 延迟重置 dragMoved, 避免 click 误触
  setTimeout(() => { dragMoved = false; renderMyHand(); }, 50);
}

// ---------- 复盘记录 ----------
function recordDecision(actualTile) {
  try {
    const az = new Analyzer(game, 0);
    const hand = az.analyzeHand();
    const opts = az.analyzeDiscards();
    if (!opts.length) return;
    const best = opts[0];
    game.reviewLog.push({
      step: game.reviewLog.length + 1,
      hand: game.players[0].hand.slice().sort((a, b) => a - b),
      actual: actualTile,
      recommended: best.tile,
      match: best.tile === actualTile,
      shanten_before: hand.shanten,
      shanten_after: best.shanten,
      options: opts.slice(0, 4).map(o => ({
        tile: o.tile, name: o.name, gang_risk: o.gang_risk,
        wait_remain: o.wait_remain, shanten: o.shanten,
      })),
    });
  } catch (e) { console.error(e); }
}

// ---------- 结算 ----------
function showResult() {
  const overlay = document.getElementById("result-overlay");
  overlay.classList.remove("hidden");
  const title = document.getElementById("result-title");
  const detail = document.getElementById("result-detail");
  const fan = document.getElementById("fan-tiles");
  const scores = document.getElementById("result-scores");

  // 累计积分
  const deltas = game.players.map(p => p.score_delta);
  const totals = addTotalScores(deltas);

  if (game.huangzhuang) {
    title.textContent = "黄庄";
    detail.textContent = "牌堆剩余不足, 本局流局, 杠分不结算";
    fan.innerHTML = "";
  } else {
    title.textContent = game.winner === 0 ? "你胡了!" : seatLabel(game.winner) + " 胡了";
    const KIND_NAME = { gangshang: "杠上花", tianhu: "天胡", zimo: "自摸" };
    detail.textContent = (KIND_NAME[game.winKind] || "自摸") +
      " | 中出1/5/9共 " + game.n159 + " 张, 每家赔 " + (game.n159 + 1) + " 分";
    fan.innerHTML = "";
    for (const t of game.fan159) fan.appendChild(makeTileEl(t, "md"));
  }
  let s = "";
  game.players.forEach((p, i) => {
    const cls = p.score_delta > 0 ? "score-row win" : "score-row";
    const sd = p.score_delta > 0 ? "+" + p.score_delta : "" + p.score_delta;
    const tt = totals[i] > 0 ? "+" + totals[i] : "" + totals[i];
    s += `<div class="${cls}"><span>${seatLabel(i)}</span><span>本局 ${sd} | 累计 ${tt}</span></div>`;
  });
  scores.innerHTML = s;
}

// ---------- 分析面板 ----------
// 代数计数器: 局面一变就作废上一次未完成的异步 E 计算
let hvGen = 0;

function refreshAnalysis() {
  if (!game || game.phase === "game_over") { lastAnalysis = null; return; }
  const az = new Analyzer(game, 0);
  const data = { hand: az.analyzeHand() };
  if (game.phase === "discard_wait" && game.turn === 0) {
    data.discards = az.analyzeDiscards();
  }
  const panel = document.getElementById("analysis-panel");
  const panelOpen = !panel.classList.contains("hidden");
  hvGen++;                       // 作废旧的异步计算
  lastAnalysis = data;
  if (panelOpen && data.discards && data.discards.length && MJWasm.ok) {
    // 先把面板画出来(带加载态), 再分片算期望巡数 —— 否则开局局面要阻塞 ~300ms
    data.hv_loading = true;
    data.hv_done = 0;
    renderAnalysis(data);
    computeHandValueAsync(data, hvGen);
  } else if (panelOpen) {
    renderAnalysis(data);
  }
  renderMyHand();
}

/** 让出一帧, 给浏览器机会把加载动画和已算出的部分画上去 */
function nextFrame() {
  return new Promise((resolve) => {
    if (typeof requestAnimationFrame === "function") requestAnimationFrame(() => resolve());
    else setTimeout(resolve, 0);
  });
}

/**
 * 逐张算期望巡数 E, 每张之间让出一帧, 过程中界面保持可响应。
 *
 * 两套口径并存而非取代:
 *   analyzer.js  score = -100*向听 + 3*有效进张   启发式加权, 毫秒级
 *   HandAnalyzer E = 期望胡牌巡数              精确递推, 有物理含义
 * 面板关闭时用前者; 打开时用后者接管「荐」与排序。
 *
 * 仅在 wasm 就绪时做: 纯 JS 跑 E 要 2-5 秒。
 * 全部算完才重排 —— 边算边排会让行不停跳动。
 */
async function computeHandValueAsync(data, gen) {
  let best = null, bestE = Infinity;
  for (let i = 0; i < data.discards.length; i++) {
    await nextFrame();
    if (gen !== hvGen) return;             // 局面已变, 丢弃本次结果
    const d = data.discards[i];
    const e = MJWasm.hvEAfterDiscard(game, 0, 1.0, d.tile);
    if (e !== null && e >= 0) {
      d.hv_e = Math.round(e * 100) / 100;
      if (e < bestE) { bestE = e; best = d.tile; }
    }
    data.hv_done = i + 1;
    renderAnalysis(data);                  // 渐进展示, 能看到 E 逐行填上
  }
  if (gen !== hvGen) return;
  data.hv_loading = false;
  if (best !== null) {
    data.rec_hv = best;
    // 按期望巡数升序重排(越小越好); 没算出 E 的排最后
    data.discards.sort((a, b) => {
      const ea = a.hv_e === undefined ? Infinity : a.hv_e;
      const eb = b.hv_e === undefined ? Infinity : b.hv_e;
      return ea - eb;
    });
  }
  renderAnalysis(data);
  renderMyHand();
}

function renderAnalysis(data) {
  const box = document.getElementById("analysis-content");
  const h = data.hand;
  let html = '<div class="ana-section"><div class="ana-title">手牌状态</div>';
  html += `<div class="ana-row">向听数: ${h.shanten}${h.is_ting ? " (已听牌)" : ""}</div>`;
  if (h.is_ting && h.waits.length) {
    html += '<div class="ana-row">听口:</div><div class="ana-wait-tiles">';
    for (const w of h.waits) html += makeTileEl(w.tile, "sm").outerHTML;
    html += `</div><div class="ana-row">进张剩余: ${h.wait_count} 张</div>`;
  }
  html += `<div class="ana-row">当前胡牌预期159: ${h.expected_fan159} 张</div>`;
  html += `<div class="ana-row">胡牌预期收益: +${h.expected_score_if_win} 分</div></div>`;

  html += '<div class="ana-section"><div class="ana-title">对手威胁</div>';
  for (const o of h.opponents) {
    const pct = Math.round(o.threat * 100);
    const cls = o.threat > 0.6 ? "risk-high" : (o.threat > 0.35 ? "risk-mid" : "risk-low");
    html += `<div class="ana-row">${seatLabel(o.seat)}: ${pct}%</div>`;
    html += `<div class="risk-bar"><div class="risk-fill ${cls}" style="width:${pct}%"></div></div>`;
  }
  html += "</div>";

  if (data.discards && data.discards.length) {
    const hasE = data.discards.some(d => d.hv_e !== undefined);
    const title = data.hv_loading
      ? "出牌建议"
      : (hasE ? "出牌建议(期望巡数升序)" : "出牌建议(牌效排序)");
    html += `<div class="ana-section"><div class="ana-title">${title}</div>`;
    if (data.hv_loading) {
      const total = data.discards.length;
      const pct = Math.round((data.hv_done || 0) / total * 100);
      html += `<div class="ana-loading"><span class="spinner"></span>`
        + `期望巡数计算中 ${data.hv_done || 0}/${total}`
        + `<span class="load-bar"><span class="load-fill" style="width:${pct}%"></span></span></div>`;
    }
    // 期望巡数条的相对刻度: 只拿「展示出来的那几行」做区间。
    // 若用全部候选的区间, 前几名的 E 往往挤在一小段里(如 18.31~18.76 对
    // 全体 18.31~20.83), 条子全是 82-100% 看不出差别。绝对值就在旁边写着,
    // 条子只负责"这几个选项之间怎么比"。
    const SHOW_N = Math.min(data.discards.length, 8);
    const shownE = data.discards.slice(0, SHOW_N)
      .filter(d => d.hv_e !== undefined).map(d => d.hv_e);
    const minE = shownE.length ? Math.min(...shownE) : 0;
    const maxE = shownE.length ? Math.max(...shownE) : 0;
    const spanE = maxE - minE;
    for (let i = 0; i < SHOW_N; i++) {
      const d = data.discards[i];
      const rpct = Math.round(d.gang_risk * 100);
      const isHv = data.rec_hv !== undefined && d.tile === data.rec_hv;
      html += `<div class="discard-suggest${i === 0 ? " best" : ""}${isHv ? " hv-best" : ""}">`;
      html += makeTileEl(d.tile, "sm").outerHTML;
      html += '<div class="info">';
      if (d.shanten === 0 && d.waits.length) {
        const wn = d.waits.map(w => w.name + "(" + w.remain + ")").join(" ");
        html += `打后听: ${wn}<br>进张 ${d.wait_remain} 张 | 被杠风险 ${rpct}%`;
      } else {
        html += `打后向听数: ${d.shanten} | 被杠风险 ${rpct}%`;
      }
      // 期望巡数: 还需要摸几巡才能胡(越小越好), 比启发式 score 直观
      if (d.hv_e !== undefined) {
        html += `<br><span class="hv-e${isHv ? " hv-e-best" : ""}">期望巡数 ${d.hv_e}${isHv ? " ★" : ""}</span>`;
        // 区间退化(全相等)时给满格, 避免除零。下限 12% 是为了让最差的一行
        // 也看得见条子, 不致于看起来像渲染失败
        const ratio = spanE > 1e-9 ? (maxE - d.hv_e) / spanE : 1;
        const w = Math.round(12 + 88 * ratio);
        const ecls = ratio >= 0.66 ? "e-good" : (ratio >= 0.33 ? "e-mid" : "e-bad");
        html += `<div class="risk-bar"><div class="risk-fill ${ecls}" style="width:${w}%"></div></div>`;
      } else if (data.hv_loading) {
        html += `<br><span class="hv-e hv-e-wait">期望巡数 计算中...</span>`;
      } else {
        // wasm 未启用: 没有 E 可画, 退回风险条
        const rcls = d.gang_risk > 0.5 ? "risk-high" : (d.gang_risk > 0.2 ? "risk-mid" : "risk-low");
        html += `<div class="risk-bar"><div class="risk-fill ${rcls}" style="width:${rpct}%"></div></div>`;
      }
      html += "</div></div>";
    }
    html += "</div>";
  }
  box.innerHTML = html;
}

// ---------- 复盘 ----------
function loadReview() {
  const log = game ? game.reviewLog : [];
  const sum = document.getElementById("review-summary");
  const box = document.getElementById("review-content");
  if (!log.length) {
    sum.textContent = "本局还没有出牌记录";
    box.innerHTML = "";
    return;
  }
  const matched = log.filter(s => s.match).length;
  sum.innerHTML = `共 ${log.length} 手 | 与AI一致 ${matched} 手 | 一致率 ${Math.round(matched / log.length * 100)}%`;
  let html = "";
  for (const s of log) {
    const cls = s.match ? "match" : "mismatch";
    const verdict = s.match ? '<span class="verdict good">一致</span>' : '<span class="verdict bad">不符</span>';
    html += `<div class="review-step ${cls}">`;
    html += `<div class="step-head"><span class="step-no">第${s.step}手</span>${verdict}</div>`;
    html += '<div class="hand-row">';
    for (const t of s.hand) html += makeTileEl(t, "sm").outerHTML;
    html += "</div>";
    html += `<div class="opt-row">你打: ${makeTileEl(s.actual, "sm").outerHTML} &nbsp; AI推荐: ${makeTileEl(s.recommended, "sm").outerHTML}</div>`;
    if (s.options && s.options.length) {
      const optText = s.options.map(o => `${o.name}(风险${Math.round(o.gang_risk * 100)}%)`).join(" ");
      html += `<div class="opt-row">候选: ${optText}</div>`;
    }
    html += `<div class="opt-row">向听数 ${s.shanten_before} -> ${s.shanten_after}</div>`;
    html += "</div>";
  }
  box.innerHTML = html;
}

// ---------- 启动 ----------
// 「新局」先开选档面板; 结算后的「再来一局」沿用当前配置, 不再弹
document.getElementById("btn-new").onclick = openSetup;
document.getElementById("btn-again").onclick = newGame;
document.getElementById("btn-setup-start").onclick = applySetupAndStart;
document.getElementById("btn-setup-cancel").onclick = () => {
  document.getElementById("setup-overlay").classList.add("hidden");
};
document.getElementById("btn-setup-default").onclick = () => {
  seatKinds = Object.assign({}, DEFAULT_KINDS);
  saveSeatKinds();
  openSetup();          // 重建面板以刷新下拉选中项
};
document.getElementById("btn-peng").onclick = doPeng;
document.getElementById("btn-gang").onclick = doGang;
document.getElementById("btn-pass").onclick = doPass;
document.getElementById("btn-sort").onclick = sortHand;
document.getElementById("btn-analysis").onclick = () => {
  document.getElementById("analysis-panel").classList.toggle("hidden");
  refreshAnalysis();
};
document.getElementById("btn-review").onclick = () => {
  document.getElementById("review-panel").classList.toggle("hidden");
  loadReview();
};
// 关面板后要重算一次: 否则 lastAnalysis 里的 rec_hv 会滞留,
// 「荐」停在期望巡数口径上不退回牌效口径
document.getElementById("close-analysis").onclick = () => {
  document.getElementById("analysis-panel").classList.add("hidden");
  refreshAnalysis();
};
document.getElementById("close-review").onclick = () =>
  document.getElementById("review-panel").classList.add("hidden");

// 先等 wasm 规则核心就绪再开局 —— 必须异步: 主线程上同步编译 >4KB 的 wasm 会被浏览器拒绝。
// 初始化失败(旧 WebView / wasm 被禁)时自动降级为纯 JS, 功能不受影响。
MJWasm.init().then((ok) => {
  if (!ok) console.warn("wasm 规则核心未启用, 降级为纯 JS(学者档会很慢)");
  newGame();
});
