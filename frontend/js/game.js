/* 安康159麻将 - web版交互逻辑(增强版)
 * 音效 + SVG牌面 + 拖拽排序 + 出牌动画 + 累计积分
 */

const SUITS = ["条", "饼", "万"];
let state = null;
let selectedUid = -1;          // 选中的手牌唯一id
let lastAnalysis = null;
let displayHand = [];          // [{uid, tile}] 显示手牌(可拖拽排序)
let uidCounter = 1;
let lastActionPlayed = "";     // 用于音效: 跟踪已播放的动作
const SEAT_NAMES = ["我", "下家", "对家", "上家"];

// ---------- 累计积分 ----------
function getTotalScores() {
  try { return JSON.parse(localStorage.getItem("mj159_total_web") || "[0,0,0,0]"); }
  catch (e) { return [0, 0, 0, 0]; }
}
function addTotalScores(deltas) {
  const t = getTotalScores();
  for (let i = 0; i < 4; i++) t[i] += deltas[i];
  localStorage.setItem("mj159_total_web", JSON.stringify(t));
  return t;
}
function resetTotalScores() {
  localStorage.setItem("mj159_total_web", JSON.stringify([0, 0, 0, 0]));
}

// ---------- 工具 ----------
function tileInfo(t) {
  if (t === 27) return { suit: -1, name: "红中" };
  const suit = Math.floor(t / 9), num = (t % 9) + 1;
  return { suit, name: num + SUITS[suit] };
}

function makeTileEl(t, size, clickable, uid) {
  const el = document.createElement("div");
  el.className = "tile face" + (size ? " " + size : "");
  el.innerHTML = TileArt.svg(t);
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

// ---------- API ----------
async function api(path, body) {
  const opt = body !== undefined
    ? { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }
    : {};
  const r = await fetch("/api/" + path, opt);
  if (!r.ok) { console.error("API error", path); return null; }
  return r.json();
}

async function newGame() {
  selectedUid = -1;
  lastAnalysis = null;
  displayHand = [];
  uidCounter = 1;
  document.getElementById("result-overlay").classList.add("hidden");
  document.getElementById("review-panel").classList.add("hidden");
  // 三个座位分别可选; 全部"默认"时不带 bot_kinds, 走后端阵容默认
  const botParam = Number(document.getElementById("bot-param")?.value || 0);
  const seatSel = [1, 2, 3].map(i => document.getElementById("bot-kind-" + i)?.value || "default");
  const body = { dealer: 0, bot_param: botParam };
  if (seatSel.some(k => k !== "default")) {
    body.bot_kinds = { 1: seatSel[0], 2: seatSel[1], 3: seatSel[2] };
  }
  state = await api("new_game", body);
  render();
  refreshAnalysis();
}

// ---------- 手牌同步(服务器手牌 -> 显示手牌) ----------
function syncDisplayHand() {
  const logical = state.players[0].hand || [];
  const logicalCount = {};
  logical.forEach(t => logicalCount[t] = (logicalCount[t] || 0) + 1);
  displayHand = displayHand.filter(c => {
    if (logicalCount[c.tile] > 0) { logicalCount[c.tile]--; return true; }
    return false;
  });
  const newTiles = [];
  Object.keys(logicalCount).forEach(k => {
    for (let i = 0; i < logicalCount[k]; i++) newTiles.push(Number(k));
  });
  // 新牌插入到适当位置(不整体重排, 保留手动排序)
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

// ---------- 音效(根据最近动作) ----------
function playSoundsFromState() {
  if (!state) return;
  const la = state.last_action || "";
  if (la === lastActionPlayed) return;
  lastActionPlayed = la;
  if (state.phase === "game_over") {
    if (state.huangzhuang) SoundFX.huang();
    else if (state.winner === 0) SoundFX.win();
    else SoundFX.lose();
    return;
  }
  if (la.includes("打出")) SoundFX.discard();
  else if (la.includes("杠")) SoundFX.gang();
  else if (la.includes("碰")) SoundFX.peng();
  else if (la.includes("摸牌") && la.includes("座位0")) SoundFX.draw();
}

// ---------- 渲染 ----------
function render() {
  if (!state) return;
  syncDisplayHand();
  document.getElementById("wall-info").textContent = "牌堆: " + state.wall_remaining;
  document.getElementById("dealer-info").textContent = "庄家: " + SEAT_NAMES[state.dealer];

  // 对手
  for (let s = 1; s <= 3; s++) {
    const p = state.players[s];
    const box = document.getElementById("opp-" + s);
    const hand = box.querySelector(".opp-hand");
    const melds = box.querySelector(".opp-melds");
    hand.innerHTML = "";
    melds.innerHTML = "";
    for (let i = 0; i < p.hand_count; i++) hand.appendChild(makeBackEl(s === 2 ? "small" : "mini"));
    for (const m of p.melds) melds.appendChild(makeMeldEl(m, "mini"));
  }

  // 弃牌(最后一张高亮 + 方向飞入)
  const dirMap = { 0: "bottom", 1: "right", 2: "top", 3: "left" };
  for (let s = 0; s <= 3; s++) {
    const area = document.querySelector("#discard-" + s + " .discards");
    area.innerHTML = "";
    const ds = state.players[s].discards;
    ds.forEach((t, i) => {
      const el = makeTileEl(t, "mini");
      if (i === ds.length - 1 && state.last_discarder === s) {
        el.classList.add("latest", "fly-" + dirMap[s]);
      }
      area.appendChild(el);
    });
  }

  // 我的手牌
  renderMyHand();

  // 副露
  const myMelds = document.getElementById("my-melds");
  myMelds.innerHTML = "";
  for (const m of state.players[0].melds) myMelds.appendChild(makeMeldEl(m, "small"));

  // 状态提示
  const statusEl = document.getElementById("my-status");
  if (state.phase === "game_over") {
    statusEl.innerHTML = "";
    showResult();
  } else if (state.phase === "discard_wait" && state.turn === 0) {
    statusEl.innerHTML = "轮到你出牌 - 点击手牌, 再点一次确认 <span class='hint-dim'>" + (state.last_action || "") + "</span>";
  } else if (state.phase === "react_wait" && state.pending_actions && state.pending_actions["0"]) {
    const who = SEAT_NAMES[state.last_discarder];
    statusEl.innerHTML = `${who} 打出 <b class='hl-tile'>${tileInfo(state.last_discard).name}</b>, 你可碰/杠/过`;
  } else {
    statusEl.innerHTML = "等待其他玩家... <span class='hint-dim'>" + (state.last_action || "") + "</span>";
  }

  updateTingHint();
  updateActionButtons();
  updateScoreBar();
  playSoundsFromState();
}

function renderMyHand() {
  const myHand = document.getElementById("my-hand");
  myHand.innerHTML = "";
  const recTile = (lastAnalysis && lastAnalysis.discards && lastAnalysis.discards.length)
    ? lastAnalysis.discards[0].tile : null;
  const drawnTile = (state.last_drawn && state.last_drawn.seat === 0) ? state.last_drawn.tile : null;
  // 打出即听的候选集合(供选中提示与标记)
  window._tingAfter = {};
  if (lastAnalysis && lastAnalysis.discards) {
    for (const d of lastAnalysis.discards) {
      if (d.waits && d.waits.length) window._tingAfter[d.tile] = d.waits;
    }
  }
  for (const card of displayHand) {
    const el = makeTileEl(card.tile, "", true, card.uid);
    if (card.uid === selectedUid) el.classList.add("selected");
    if (recTile !== null && card.tile === recTile) el.classList.add("recommended");
    if (drawnTile !== null && card.tile === drawnTile) el.classList.add("drawn");
    if (window._tingAfter[card.tile]) el.classList.add("ting-able");
    myHand.appendChild(el);
  }
  updateSelectionHint();
}

// 选中牌时立即显示"打出后听什么"(打出前就能看到, 不会被打出去后的快速回合冲掉)
function updateSelectionHint() {
  let el = document.getElementById("sel-hint");
  if (!el) {
    el = document.createElement("div");
    el.id = "sel-hint";
    el.className = "hidden";
    const anchor = document.getElementById("ting-hint");
    if (anchor) anchor.after(el);
    else document.getElementById("my-area").prepend(el);
  }
  const card = displayHand.find(c => c.uid === selectedUid);
  const waits = card ? window._tingAfter[card.tile] : null;
  if (card && waits && waits.length) {
    const total = waits.reduce((s, w) => s + w.remain, 0);
    el.innerHTML = `<span class="sel-hint-label">打 </span>`
      + makeTileEl(card.tile, "mini").outerHTML
      + `<span class="sel-hint-label"> → 听: </span>`
      + waits.map(w => makeTileEl(w.tile, "mini").outerHTML
                  + `<span class="wait-rem">×${w.remain}</span>`).join("")
      + `<span class="sel-hint-label"> 共剩 ${total} 张</span>`;
    el.classList.remove("hidden");
  } else {
    el.innerHTML = "";
    el.classList.add("hidden");
  }
}

// 听牌状态提示
function updateTingHint() {
  let hintEl = document.getElementById("ting-hint");
  if (!hintEl) {
    hintEl = document.createElement("div");
    hintEl.id = "ting-hint";
    document.getElementById("my-area").prepend(hintEl);
  }
  if (!state || state.phase === "game_over" || !lastAnalysis) { hintEl.innerHTML = ""; return; }
  const h = lastAnalysis.hand;
  if (h.is_ting && h.waits && h.waits.length) {
    hintEl.innerHTML = `<span class='ting-ok'>已听牌, 听: ${h.waits.map(w => w.name).join(" ")}</span>`;
  } else if (h.shanten !== undefined && h.shanten >= 0) {
    hintEl.innerHTML = `<span class='ting-no'>向听数: ${h.shanten}</span>`;
  }
}

// 累计积分条
function updateScoreBar() {
  let bar = document.getElementById("score-bar");
  if (!bar) {
    bar = document.createElement("span");
    bar.id = "score-bar";
    document.querySelector(".top-info").prepend(bar);
  }
  const t = getTotalScores();
  bar.innerHTML = `累计 我:${t[0] > 0 ? "+" : ""}${t[0]} <span class='reset-score' id='reset-score'>清零</span>`;
  const rst = document.getElementById("reset-score");
  if (rst) rst.onclick = (e) => { e.stopPropagation(); resetTotalScores(); updateScoreBar(); };
}

function updateActionButtons() {
  const bp = document.getElementById("btn-peng");
  const bg = document.getElementById("btn-gang");
  const bs = document.getElementById("btn-pass");
  const gc = document.getElementById("gang-choices");
  bp.classList.add("hidden"); bg.classList.add("hidden"); bs.classList.add("hidden");
  gc.classList.add("hidden"); gc.innerHTML = "";

  if (!state || state.phase === "game_over") return;

  if (state.phase === "react_wait" && state.pending_actions && state.pending_actions["0"]) {
    const pa = state.pending_actions["0"];
    if (pa.peng) bp.classList.remove("hidden");
    if (pa.gang) bg.classList.remove("hidden");
    bs.classList.remove("hidden");
  }
  if (state.phase === "discard_wait" && state.turn === 0) {
    const gopts = state.gang_options || [];
    if (gopts.length > 0) {
      bg.classList.remove("hidden");
      gc.classList.remove("hidden");
      for (const t of gopts) {
        const el = makeTileEl(t, "small");
        el.onclick = () => doGangTile(t);
        gc.appendChild(el);
      }
    }
  }
}

// ---------- 交互: 点击选中/出牌 ----------
function attachTileHandlers(el) {
  el.addEventListener("click", () => {
    if (dragMoved) return;
    onTileClick(Number(el.dataset.tile), Number(el.dataset.uid));
  });
  el.addEventListener("pointerdown", onDragStart);
}

function onTileClick(tile, uid) {
  if (!state) return;
  if (state.phase === "discard_wait" && state.turn === 0) {
    if (selectedUid === uid) doDiscard(tile);
    else { selectedUid = uid; renderMyHand(); }
  }
}

async function doDiscard(t) {
  selectedUid = -1;
  SoundFX.discard();
  state = await api("discard", { tile: t });
  render();
  refreshAnalysis();
}

async function doPeng() {
  state = await api("peng", {}); SoundFX.peng();
  render(); refreshAnalysis();
}
async function doGang() {
  state = await api("gang", {}); SoundFX.gang();
  render(); refreshAnalysis();
}
async function doGangTile(t) {
  state = await api("gang", { tile: t }); SoundFX.gang();
  render(); refreshAnalysis();
}
async function doPass() {
  state = await api("pass", {});
  render(); refreshAnalysis();
}

// ---------- 手牌拖拽排序 ----------
let dragState = null;
let dragMoved = false;

function onDragStart(e) {
  if (!state || state.phase === "game_over") return;
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
    const myHand = document.getElementById("my-hand");
    const tiles = [...myHand.querySelectorAll(".tile")];
    const dropX = e.clientX;
    let targetIndex = displayHand.length - 1;
    for (let i = 0; i < tiles.length; i++) {
      const r = tiles[i].getBoundingClientRect();
      if (dropX < r.left + r.width / 2) { targetIndex = i; break; }
    }
    const card = displayHand.splice(dragState.startIndex, 1)[0];
    if (card) {
      let insertAt = targetIndex > dragState.startIndex ? targetIndex - 1 : targetIndex;
      displayHand.splice(Math.max(0, Math.min(insertAt, displayHand.length)), 0, card);
    }
    SoundFX.click();
    selectedUid = -1;
  }
  dragState = null;
  setTimeout(() => { dragMoved = false; renderMyHand(); }, 50);
}

// ---------- 结算 ----------
function showResult() {
  const overlay = document.getElementById("result-overlay");
  overlay.classList.remove("hidden");
  const title = document.getElementById("result-title");
  const detail = document.getElementById("result-detail");
  const scores = document.getElementById("result-scores");

  const deltas = state.players.map(p => p.score_delta);
  const totals = addTotalScores(deltas);

  if (state.huangzhuang) {
    title.textContent = "黄庄";
    detail.innerHTML = "牌堆剩余不足, 本局流局, 杠分不结算";
    scores.innerHTML = "";
    return;
  }
  title.textContent = (state.winner === 0 ? "你胡了!" : SEAT_NAMES[state.winner] + " 胡了");
  const KIND_NAME = { gangshang: "杠上花", tianhu: "天胡", zimo: "自摸" };
  const kindName = KIND_NAME[state.win_kind] || "自摸";
  let d = `胡牌方式: ${kindName}<br>159翻牌: `;
  d += '<div id="fan-tiles">';
  for (const t of state.fan_159) d += makeTileEl(t, "small").outerHTML;
  d += "</div>";
  d += `中出 1/5/9 共 ${state.n_159} 张, 每家赔 ${state.n_159 + 1} 分`;
  detail.innerHTML = d;

  let s = "";
  state.players.forEach((p, i) => {
    const cls = p.score_delta > 0 ? "score-row win" : "score-row";
    const sd = p.score_delta > 0 ? "+" + p.score_delta : "" + p.score_delta;
    const tt = totals[i] > 0 ? "+" + totals[i] : "" + totals[i];
    s += `<div class="${cls}"><span>${SEAT_NAMES[i]}</span><span>本局 ${sd} | 累计 ${tt}</span></div>`;
  });
  scores.innerHTML = s;
}

// ---------- 分析面板 ----------
function toggleAnalysis() {
  document.getElementById("analysis-panel").classList.toggle("hidden");
}

async function refreshAnalysis() {
  const rho = document.getElementById("rho-slider")?.value ?? 1.0;
  const data = await api("analyze?rho=" + rho);
  if (!data) return;
  lastAnalysis = data;
  const panel = document.getElementById("analysis-panel");
  if (!panel.classList.contains("hidden")) renderAnalysis(data);
  if (state && state.phase !== "game_over") {
    renderMyHand();
    updateTingHint();
  }
}

function renderAnalysis(data) {
  const box = document.getElementById("analysis-content");
  const h = data.hand;
  let html = '<div class="ana-section"><div class="ana-title">手牌状态</div>';
  html += `<div class="ana-row">向听数: ${h.shanten}${h.is_ting ? " (已听牌)" : ""}</div>`;
  if (h.is_ting && h.waits.length) {
    html += '<div class="ana-row">听口:</div><div class="ana-wait-tiles">';
    for (const w of h.waits) html += makeTileEl(w.tile, "mini").outerHTML;
    html += `</div><div class="ana-row">进张剩余: ${h.wait_count} 张</div>`;
  }
  html += `<div class="ana-row">当前胡牌预期159: ${h.expected_fan159} 张</div>`;
  html += `<div class="ana-row">胡牌预期收益: +${h.expected_score_if_win} 分</div></div>`;

  html += '<div class="ana-section"><div class="ana-title">对手威胁</div>';
  for (const o of h.opponents) {
    const pct = Math.round(o.threat * 100);
    const cls = o.threat > 0.6 ? "risk-high" : (o.threat > 0.35 ? "risk-mid" : "risk-low");
    html += `<div class="ana-row">${SEAT_NAMES[o.seat]}: ${pct}%</div>`;
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
      html += makeTileEl(d.tile, "mini").outerHTML;
      html += '<div class="info">';
      if (d.hand_value != null) {
        html += `<span class="hv-badge">期望 ${d.hand_value} 巡胡</span>`;
        if (d.hand_value_np != null && Math.abs(d.hand_value_np - d.hand_value) > 0.01)
          html += `<span class="hv-np">(纯自摸 ${d.hand_value_np})</span>`;
        html += "<br>";
      }
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
function toggleReview() {
  document.getElementById("review-panel").classList.toggle("hidden");
}

async function loadReview() {
  const data = await api("review");
  if (!data) return;
  const sum = document.getElementById("review-summary");
  const box = document.getElementById("review-content");
  if (!data.total) {
    sum.textContent = "本局还没有出牌记录";
    box.innerHTML = "";
    return;
  }
  sum.innerHTML = `共 ${data.total} 手 | 与AI一致 ${data.matched} 手 | 一致率 ${Math.round(data.match_rate * 100)}%`;
  let html = "";
  for (const s of data.steps) {
    const cls = s.match ? "match" : "mismatch";
    const verdict = s.match ? '<span class="verdict good">一致</span>' : '<span class="verdict bad">不符</span>';
    html += `<div class="review-step ${cls}">`;
    html += `<div class="step-head"><span class="step-no">第${s.step}手</span>${verdict}</div>`;
    html += '<div class="hand-row">';
    for (const t of s.hand) html += makeTileEl(t, "mini").outerHTML;
    html += "</div>";
    html += `<div class="opt-row">你打: ${makeTileEl(s.actual, "mini").outerHTML} &nbsp; AI推荐: ${makeTileEl(s.recommended, "mini").outerHTML}</div>`;
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
const BOT_KINDS = [
  ["default", "默认(阵容)"],
  ["v1", "菜鸟(v1)"],
  ["v10", "中鸟(v10)"],
  ["v31", "老鸟(v31)"],
  ["scholar", "学者(牌型价值)"],
  ["target", "目标概率"],
  ["cheat_wall", "挂哥(看牌墙)"],
  ["cheat_opp", "挂王(看牌+手牌)"],
  ["cheat_full", "神挂(全信息)"],
];
for (const i of [1, 2, 3]) {
  const sel = document.getElementById("bot-kind-" + i);
  if (!sel) continue;
  for (const [v, label] of BOT_KINDS) {
    const op = document.createElement("option");
    op.value = v; op.textContent = label;
    sel.appendChild(op);
  }
}
const rhoSlider = document.getElementById("rho-slider");
if (rhoSlider) rhoSlider.oninput = () => {
  document.getElementById("rho-val").textContent = Number(rhoSlider.value).toFixed(1);
  refreshAnalysis();
};

document.getElementById("btn-new").onclick = newGame;
document.getElementById("btn-sort").onclick = sortHand;
document.getElementById("btn-analysis").onclick = () => { toggleAnalysis(); refreshAnalysis(); };
document.getElementById("btn-review").onclick = () => { toggleReview(); loadReview(); };
const againBtn = document.getElementById("btn-again");
if (againBtn) againBtn.onclick = newGame;
newGame();
