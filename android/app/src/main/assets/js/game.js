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

// 对手阵容: seat -> {昵称, 工厂函数, 说明}
// 与 backend/ai/roster.py 的 ROSTER 保持一致
// 老鸟已从 v4 升级为 v31(副露感知向听 + 两步推演), 会正常碰杠
const ROSTER = {
  1: { nick: "菜鸟", desc: "规则Bot v1: 牌效优先",
       make: (g, s) => new Bot(g, s) },
  2: { nick: "老鸟", desc: "规则Bot v31: 两步推演+副露感知向听",
       make: (g, s) => new BotV31(g, s) },
  3: { nick: "挂哥", desc: "作弊Bot: 可见牌堆",
       make: (g, s) => new BotOracle(g, s, { beam: 12 }) },
};
let aiBots = {};   // seat -> bot 实例(每局初始化)

function seatNick(seat) {
  return ROSTER[seat] ? ROSTER[seat].nick : "我";
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
  game = new Game(0);
  // 为每个 AI 座位创建对应的 bot
  aiBots = {};
  for (const s of [1, 2, 3]) aiBots[s] = ROSTER[s].make(game, s);
  render();
  refreshAnalysis();
  maybeRunAI();
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
  const recTile = (lastAnalysis && lastAnalysis.discards && lastAnalysis.discards.length)
    ? lastAnalysis.discards[0].tile : null;
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
  // 必须用副露感知向听: 碰/杠后暗牌变短(11/10张), 直接调 shanten() 会把
  // 向听高估约 2*副露数, 导致"有副露时已经听牌了却不提示"
  // (实测: 1个碰 + 暗牌听牌, 旧逻辑算出 2, 正确值是 0)
  const s = shantenWithMelds(counts, p.melds.length);
  if (s === 0 && p.hand.length % 3 === 1) {
    const names = waitingTiles(counts).map(tileName).join(" ");
    hintEl.innerHTML = `<span class='ting-ok'>已听牌, 听: ${names}</span>`;
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
function refreshAnalysis() {
  if (!game || game.phase === "game_over") { lastAnalysis = null; return; }
  const az = new Analyzer(game, 0);
  const data = { hand: az.analyzeHand() };
  if (game.phase === "discard_wait" && game.turn === 0) {
    data.discards = az.analyzeDiscards();
  }
  lastAnalysis = data;
  const panel = document.getElementById("analysis-panel");
  if (!panel.classList.contains("hidden")) renderAnalysis(data);
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
    html += '<div class="ana-section"><div class="ana-title">出牌建议(最优在前)</div>';
    for (let i = 0; i < Math.min(data.discards.length, 8); i++) {
      const d = data.discards[i];
      const rpct = Math.round(d.gang_risk * 100);
      const rcls = d.gang_risk > 0.5 ? "risk-high" : (d.gang_risk > 0.2 ? "risk-mid" : "risk-low");
      html += `<div class="discard-suggest${i === 0 ? " best" : ""}">`;
      html += makeTileEl(d.tile, "sm").outerHTML;
      html += '<div class="info">';
      if (d.shanten === 0 && d.waits.length) {
        const wn = d.waits.map(w => w.name + "(" + w.remain + ")").join(" ");
        html += `打后听: ${wn}<br>进张 ${d.wait_remain} 张 | 被杠风险 ${rpct}%`;
      } else {
        html += `打后向听数: ${d.shanten} | 被杠风险 ${rpct}%`;
      }
      html += `<div class="risk-bar"><div class="risk-fill ${rcls}" style="width:${rpct}%"></div></div>`;
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
document.getElementById("btn-new").onclick = newGame;
document.getElementById("btn-again").onclick = newGame;
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
document.getElementById("close-analysis").onclick = () =>
  document.getElementById("analysis-panel").classList.add("hidden");
document.getElementById("close-review").onclick = () =>
  document.getElementById("review-panel").classList.add("hidden");

newGame();
