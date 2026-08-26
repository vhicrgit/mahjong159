/* 安康159麻将 - 前端交互逻辑 */

const SUITS = ["条", "饼", "万"];
let state = null;
let selectedTile = null;
let lastAnalysis = null;   // 最近一次分析结果(用于推荐牌高亮)

// ---------- 工具 ----------
function tileInfo(t) {
  if (t === 27) return { suit: -1, num: "中", cls: "hz", name: "红中" };
  const suit = Math.floor(t / 9);
  const num = (t % 9) + 1;
  return { suit, num, cls: "suit-" + suit, name: num + SUITS[suit] };
}

function makeTileEl(t, size, clickable) {
  const info = tileInfo(t);
  const el = document.createElement("div");
  el.className = "tile " + info.cls + (size ? " " + size : "");
  if (t === 27) {
    el.innerHTML = '<span class="t-num">中</span>';
  } else {
    el.innerHTML = `<span class="t-num">${info.num}</span><span class="t-suit">${SUITS[info.suit]}</span>`;
  }
  if (clickable) {
    el.onclick = () => onTileClick(t, el);
  }
  return el;
}

function makeBackEl(size) {
  const el = document.createElement("div");
  el.className = "tile back" + (size ? " " + size : "");
  return el;
}

// ---------- API ----------
async function api(path, body) {
  const opt = body !== undefined
    ? { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }
    : {};
  const r = await fetch("/api/" + path, opt);
  if (!r.ok) {
    const e = await r.json().catch(() => ({}));
    console.error("API error", path, e);
    return null;
  }
  return r.json();
}

async function newGame() {
  selectedTile = null;
  lastAnalysis = null;
  document.getElementById("result-overlay").classList.add("hidden");
  document.getElementById("review-panel").classList.add("hidden");
  const botKind = document.getElementById("bot-kind")?.value || "v10";
  const botParam = Number(document.getElementById("bot-param")?.value || 0);
  state = await api("new_game", { dealer: 0, bot_kind: botKind, bot_param: botParam });
  render();
  refreshAnalysis();
}

async function refresh() {
  state = await api("state");
  render();
}

// ---------- 渲染 ----------
function render() {
  if (!state) return;
  document.getElementById("wall-info").textContent = "牌堆: " + state.wall_remaining;
  const dealerName = ["我", "下家", "对家", "上家"][state.dealer];
  document.getElementById("dealer-info").textContent = "庄家: " + dealerName;

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

  // 弃牌区
  for (let s = 0; s <= 3; s++) {
    const area = document.querySelector("#discard-" + s + " .discards");
    area.innerHTML = "";
    for (const t of state.players[s].discards) area.appendChild(makeTileEl(t, "mini"));
  }

  // 我的手牌
  const me = state.players[0];
  const myHand = document.getElementById("my-hand");
  myHand.innerHTML = "";
  const myMelds = document.getElementById("my-melds");
  myMelds.innerHTML = "";
  for (const m of me.melds) myMelds.appendChild(makeMeldEl(m, "small"));
  if (me.hand) {
    // 推荐牌(Akagi式高亮): 分析器最优打法
    const recTile = (lastAnalysis && lastAnalysis.discards && lastAnalysis.discards.length)
      ? lastAnalysis.discards[0].tile : null;
    me.hand.forEach((t, i) => {
      const el = makeTileEl(t, "", true);
      if (selectedTile === t) el.classList.add("selected");
      if (recTile !== null && t === recTile) el.classList.add("recommended");
      myHand.appendChild(el);
    });
  }

  // 状态提示
  const statusEl = document.getElementById("my-status");
  if (state.phase === "game_over") {
    statusEl.textContent = "";
    showResult();
  } else if (state.phase === "discard_wait" && state.turn === 0) {
    statusEl.textContent = "轮到你出牌 - 点击手牌打出";
  } else if (state.phase === "react_wait" && state.pending_actions && state.pending_actions["0"]) {
    statusEl.textContent = "有人打出牌, 你可以选择碰/杠/过";
  } else {
    statusEl.textContent = "等待其他玩家...";
  }

  // 操作按钮
  updateActionButtons();
}

function makeMeldEl(m, size) {
  const wrap = document.createElement("div");
  wrap.style.display = "flex";
  wrap.style.gap = "1px";
  const n = m.type === "peng" ? 3 : 4;
  for (let i = 0; i < n; i++) wrap.appendChild(makeTileEl(m.tile, size));
  return wrap;
}

function updateActionButtons() {
  const bp = document.getElementById("btn-peng");
  const bg = document.getElementById("btn-gang");
  const bs = document.getElementById("btn-pass");
  bp.classList.add("hidden");
  bg.classList.add("hidden");
  bs.classList.add("hidden");
  document.getElementById("gang-choices").classList.add("hidden");

  if (!state || state.phase === "game_over") return;

  // 碰杠响应
  if (state.phase === "react_wait" && state.pending_actions && state.pending_actions["0"]) {
    const pa = state.pending_actions["0"];
    if (pa.peng) bp.classList.remove("hidden");
    if (pa.gang) bg.classList.remove("hidden");
    bs.classList.remove("hidden");
  }
  // 摸牌后的暗杠/补杠
  if (state.phase === "discard_wait" && state.turn === 0) {
    const gopts = state.gang_options || [];
    if (gopts.length > 0) {
      bg.classList.remove("hidden");
      const gc = document.getElementById("gang-choices");
      gc.innerHTML = "";
      gc.classList.remove("hidden");
      for (const t of gopts) {
        const el = makeTileEl(t, "small");
        el.onclick = () => doGangTile(t);
        gc.appendChild(el);
      }
    }
  }
}

// ---------- 交互 ----------
function onTileClick(t, el) {
  if (!state) return;
  if (state.phase === "discard_wait" && state.turn === 0) {
    if (selectedTile === t) {
      // 再次点击 = 确认打出
      doDiscard(t);
    } else {
      selectedTile = t;
      render();
    }
  }
}

async function doDiscard(t) {
  selectedTile = null;
  state = await api("discard", { tile: t });
  render();
  refreshAnalysis();
}

async function doPeng() {
  state = await api("peng", {});
  render();
  refreshAnalysis();
}

async function doGang() {
  // react阶段的明杠
  state = await api("gang", {});
  render();
  refreshAnalysis();
}

async function doGangTile(t) {
  state = await api("gang", { tile: t });
  render();
  refreshAnalysis();
}

async function doPass() {
  state = await api("pass", {});
  render();
  refreshAnalysis();
}

// ---------- 结果 ----------
function showResult() {
  const overlay = document.getElementById("result-overlay");
  overlay.classList.remove("hidden");
  const title = document.getElementById("result-title");
  const detail = document.getElementById("result-detail");
  const scores = document.getElementById("result-scores");

  if (state.huangzhuang) {
    title.textContent = "黄庄";
    detail.innerHTML = "牌堆剩余不足, 本局流局, 杠分不结算";
    scores.innerHTML = "";
    return;
  }
  const wname = ["我", "下家", "对家", "上家"][state.winner];
  title.textContent = (state.winner === 0 ? "你胡了!" : wname + " 胡了");
  const kindName = state.win_kind === "gangshang" ? "杠上花" : "自摸";
  let d = `胡牌方式: ${kindName}<br>159翻牌: `;
  d += '<div id="fan-tiles">';
  for (const t of state.fan_159) d += makeTileEl(t, "small").outerHTML;
  d += "</div>";
  d += `中出 1/5/9 共 ${state.n_159} 张, 每家赔 ${state.n_159 + 1} 分`;
  detail.innerHTML = d;

  let s = "";
  state.players.forEach((p, i) => {
    const nm = ["我", "下家", "对家", "上家"][i];
    const cls = p.score_delta > 0 ? "score-row win" : "score-row";
    s += `<div class="${cls}"><span>${nm}</span><span>${p.score_delta > 0 ? "+" : ""}${p.score_delta}</span></div>`;
  });
  scores.innerHTML = s;
}

// ---------- 分析面板 ----------
function toggleAnalysis() {
  document.getElementById("analysis-panel").classList.toggle("hidden");
}

async function refreshAnalysis() {
  // 始终拉取分析(用于推荐牌高亮), 面板开着才更新内容
  const data = await api("analyze");
  if (!data) return;
  lastAnalysis = data;
  const panel = document.getElementById("analysis-panel");
  if (!panel.classList.contains("hidden")) renderAnalysis(data);
  // 推荐高亮需要重绘手牌
  if (state && state.phase !== "game_over") renderHandOnly();
}

function renderHandOnly() {
  const me = state.players[0];
  const myHand = document.getElementById("my-hand");
  myHand.innerHTML = "";
  const recTile = (lastAnalysis && lastAnalysis.discards && lastAnalysis.discards.length)
    ? lastAnalysis.discards[0].tile : null;
  if (me.hand) {
    me.hand.forEach((t) => {
      const el = makeTileEl(t, "", true);
      if (selectedTile === t) el.classList.add("selected");
      if (recTile !== null && t === recTile) el.classList.add("recommended");
      myHand.appendChild(el);
    });
  }
}

// ---------- 复盘(mjai式赛后检讨) ----------
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
    const verdict = s.match
      ? '<span class="verdict good">一致</span>'
      : '<span class="verdict bad">不符</span>';
    html += `<div class="review-step ${cls}">`;
    html += `<div class="step-head"><span class="step-no">第${s.step}手</span>${verdict}</div>`;
    html += '<div class="hand-row">';
    for (const t of s.hand) html += makeTileEl(t, "mini").outerHTML;
    html += "</div>";
    const actual = makeTileEl(s.actual, "mini").outerHTML;
    const rec = makeTileEl(s.recommended, "mini").outerHTML;
    html += `<div class="opt-row">你打: ${actual} &nbsp; AI推荐: ${rec}</div>`;
    if (s.options && s.options.length) {
      const optText = s.options.map(o =>
        `${o.name}(风险${Math.round(o.gang_risk * 100)}%)`).join(" ");
      html += `<div class="opt-row">候选: ${optText}</div>`;
    }
    html += `<div class="opt-row">向听数 ${s.shanten_before} -> ${s.shanten_after}</div>`;
    html += "</div>";
  }
  box.innerHTML = html;
}

function renderAnalysis(data) {
  const box = document.getElementById("analysis-content");
  const h = data.hand;
  let html = "";

  // 手牌概况
  html += '<div class="ana-section"><div class="ana-title">手牌状态</div>';
  html += `<div class="ana-row">向听数: ${h.shanten}${h.is_ting ? " (已听牌)" : ""}</div>`;
  if (h.is_ting && h.waits.length) {
    html += '<div class="ana-row">听口:</div><div class="ana-wait-tiles">';
    for (const w of h.waits) {
      html += makeTileEl(w.tile, "mini").outerHTML;
    }
    html += "</div>";
    html += `<div class="ana-row">进张剩余: ${h.wait_count} 张</div>`;
  }
  html += `<div class="ana-row">当前胡牌预期159: ${h.expected_fan159} 张</div>`;
  html += `<div class="ana-row">胡牌预期收益: +${h.expected_score_if_win} 分</div>`;
  html += "</div>";

  // 对手威胁
  html += '<div class="ana-section"><div class="ana-title">对手威胁</div>';
  for (const o of h.opponents) {
    const nm = ["我", "下家", "对家", "上家"][o.seat];
    const pct = Math.round(o.threat * 100);
    const cls = o.threat > 0.6 ? "risk-high" : (o.threat > 0.35 ? "risk-mid" : "risk-low");
    html += `<div class="ana-row">${nm}: ${pct}%</div>`;
    html += `<div class="risk-bar"><div class="risk-fill ${cls}" style="width:${pct}%"></div></div>`;
  }
  html += "</div>";

  // 出牌建议
  if (data.discards && data.discards.length) {
    html += '<div class="ana-section"><div class="ana-title">出牌建议(最优在前)</div>';
    for (let i = 0; i < Math.min(data.discards.length, 8); i++) {
      const d = data.discards[i];
      const rpct = Math.round(d.gang_risk * 100);
      const rcls = d.gang_risk > 0.5 ? "risk-high" : (d.gang_risk > 0.2 ? "risk-mid" : "risk-low");
      html += `<div class="discard-suggest${i === 0 ? " best" : ""}">`;
      html += makeTileEl(d.tile, "mini").outerHTML;
      html += '<div class="info">';
      if (d.shanten === 0 && d.waits.length) {
        const wnames = d.waits.map(w => tileInfo(w.tile).name + "(" + w.remain + ")").join(" ");
        html += `打后听: ${wnames}<br>进张 ${d.wait_remain} 张 | 被杠风险 ${rpct}%`;
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

// ---------- 启动 ----------
document.getElementById("btn-new").onclick = newGame;
document.getElementById("btn-analysis").onclick = () => {
  toggleAnalysis();
  refreshAnalysis();
};
document.getElementById("btn-review").onclick = () => {
  toggleReview();
  loadReview();
};
newGame();
