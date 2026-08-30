/* 安康159 - wasm 规则核心的加载与接入
 *
 * 提供 MJWasm.init() (返回 Promise)。初始化成功后:
 *   1. 把全局 shanten / isWin / usefulSet 替换为 wasm 版本(所有 Bot 与引擎都受益)
 *   2. 学者Bot 走 mj_hv_* 整体入 wasm 的快路径(一次决策只跨界一次)
 * 初始化失败(旧 WebView / wasm 被禁)时静默降级为纯 JS, 功能不受影响。
 *
 * 实测提速(Node, 见 mobile/wasm/mjcore.c 头部):
 *   shanten            0.0147ms -> 0.0027ms   5.5x
 *   学者 chooseDiscard  中盘均值 4781ms -> 162ms   29x
 *                      最坏     28410ms -> 1095ms  26x
 *
 * 必须用异步实例化: 主线程上 new WebAssembly.Module() 对 >4KB 的字节数组
 * 会被浏览器拒绝, 我们的模块 44KB。
 */

const MJWasm = (function () {
  let X = null;            // wasm exports
  let mem8 = null;         // Int8Array 视图
  let P1 = 0, P2 = 0;      // 两个共享缓冲区的地址
  let ready = false;
  let initPromise = null;

  // 保留纯 JS 实现的引用, 作为降级兜底与对拍基准
  const jsImpl = {};

  function b64ToBytes(b64) {
    // atob 在 WebView 里可用; Node 下没有, 用 Buffer
    if (typeof atob === "function") {
      const s = atob(b64);
      const out = new Uint8Array(s.length);
      for (let i = 0; i < s.length; i++) out[i] = s.charCodeAt(i);
      return out;
    }
    return new Uint8Array(Buffer.from(b64, "base64"));
  }

  function put(p, counts) {
    for (let i = 0; i < 28; i++) mem8[p + i] = counts[i];
  }

  /* ---- wasm 版规则接口 ---- */
  function wShanten(counts) { put(P1, counts); return X.mj_shanten(); }
  function wIsWin(counts) { put(P1, counts); return !!X.mj_is_win(); }

  const _usCache = new Map();
  const US_LIMIT = 200000;
  function wUsefulSet(counts) {
    // 仍然缓存: 命中时连跨界都省了
    let key = "";
    for (let i = 0; i < 28; i++) key += String.fromCharCode(counts[i]);
    const hit = _usCache.get(key);
    if (hit !== undefined) return hit;
    put(P1, counts);
    const mask = X.mj_useful_mask();
    const out = [];
    for (let t = 0; t < 28; t++) if ((mask >> t) & 1) out.push(t);
    if (_usCache.size >= US_LIMIT) _usCache.clear();
    _usCache.set(key, out);
    return out;
  }

  /* ---- 学者Bot 的 wasm 快路径 ---- */
  // 把某座位的手牌与可见牌写入 wasm, 口径与 bot_hv.js 的 _analyzer() 一致
  // kaiMax: 换型预算(0=关)。需要新版 wasm 的 mj_hv_set2; 旧版 wasm 的换型
  // 语义是"连续次数重置", 高向听手牌会 DP 爆炸, 所以旧版一律退回 kaizen=0。
  function hvSetup(game, seat, rho, kaiMax) {
    const visible = new Array(28).fill(0);
    for (const q of game.players) {
      for (const t of q.discards) visible[t]++;
      for (const m of q.melds) visible[m.tile] += m.type === "peng" ? 3 : 4;
    }
    const hc = game.players[seat].handCounts();
    for (let t = 0; t < 28; t++) visible[t] += hc[t];
    put(P1, hc);
    put(P2, visible);
    if (typeof X.mj_hv_set2 === "function") {
      // (rho, kaizen, kaiMargin, kaiMax, kaiTopk)
      X.mj_hv_set2(rho, kaiMax > 0 ? 1 : 0, 2, kaiMax, 6);
    } else {
      X.mj_hv_set(rho, 0);
    }
  }

  const GANG_KIND = { ming: 0, an: 1, bu: 2 };

  return {
    /** 是否已就绪(可用 wasm) */
    get ok() { return ready; },

    /** 载入并实例化 wasm。总是 resolve; 失败时 ok 为 false 并降级为纯 JS。 */
    init() {
      if (initPromise) return initPromise;
      initPromise = (async () => {
        try {
          if (typeof WebAssembly === "undefined" || typeof MJ_WASM_B64 === "undefined") {
            return false;
          }
          const bytes = b64ToBytes(MJ_WASM_B64);
          const r = await WebAssembly.instantiate(bytes, {});
          X = r.instance.exports;
          mem8 = new Int8Array(X.memory.buffer);
          P1 = X.mj_buf_ptr();
          P2 = X.mj_buf2_ptr();

          // 自检: 拿几个已知答案验证一遍, 不对就不启用
          //   13张 111饼222饼333饼444饼+1条 -> 听牌(向听0), 听 1条 与 红中
          const probe = new Array(28).fill(0);
          for (const t of [9, 9, 9, 10, 10, 10, 11, 11, 11, 12, 12, 12, 0]) probe[t]++;
          if (wShanten(probe) !== 0) return false;
          probe[0]++;                       // 补成一对 -> 胡
          if (!wIsWin(probe)) return false;

          // 备份纯 JS 实现后替换全局
          jsImpl.shanten = shanten;
          jsImpl.isWin = isWin;
          if (typeof usefulSet === "function") jsImpl.usefulSet = usefulSet;
          shanten = wShanten;
          isWin = wIsWin;
          if (typeof usefulSet === "function") usefulSet = wUsefulSet;

          ready = true;
          return true;
        } catch (e) {
          ready = false;
          return false;
        }
      })();
      return initPromise;
    },

    /** 退回纯 JS(排查用) */
    disable() {
      if (!ready) return;
      shanten = jsImpl.shanten;
      isWin = jsImpl.isWin;
      if (jsImpl.usefulSet) usefulSet = jsImpl.usefulSet;
      ready = false;
    },

    /* ---- 学者Bot 快路径; 未就绪时返回 null, 调用方退回 JS 实现 ---- */
    hvChooseDiscard(game, seat, rho) {
      if (!ready) return null;
      hvSetup(game, seat, rho);
      const t = X.mj_hv_choose_discard();
      return t >= 0 ? t : null;
    },
    hvDecidePeng(game, seat, rho, tile) {
      if (!ready) return null;
      hvSetup(game, seat, rho);
      return !!X.mj_hv_decide_peng(tile);
    },
    hvDecideGang(game, seat, rho, tile, kind) {
      if (!ready) return null;
      hvSetup(game, seat, rho);
      return !!X.mj_hv_decide_gang(tile, GANG_KIND[kind] !== undefined ? GANG_KIND[kind] : 0);
    },
    /** 打出某张后的期望巡数(分析面板可用); kaiMax=换型预算(0/1/2); 未就绪返回 null */
    hvEAfterDiscard(game, seat, rho, tile, kaiMax) {
      if (!ready) return null;
      hvSetup(game, seat, rho, kaiMax || 0);
      return X.mj_hv_e_after_discard(tile);
    },
    /** E 的通道分解 + 明细(自摸/换型/碰)。需要新版 wasm; 未就绪返回 null */
    hvExplain(game, seat, rho, tile, kaiMax) {
      if (!ready || typeof X.mj_hv_explain !== "function") return null;
      hvSetup(game, seat, rho, kaiMax || 0);
      if (X.mj_hv_explain(tile) !== 0) return null;
      // 视图直指 wasm 内存, 必须立即把值拷出来(下一次调用会覆盖)
      const f8 = new Float64Array(X.memory.buffer, X.mj_outf_ptr(), 8);
      const i32 = new Int32Array(X.memory.buffer, X.mj_outi_ptr(), 192);
      const out = { E: f8[0], wait: f8[1], cUseful: f8[2], cKai: f8[3],
                    cPeng: f8[4], useful: [], kai: [], peng: [] };
      let q = 0;
      const nu = i32[q++];
      for (let i = 0; i < nu; i++) { out.useful.push([i32[q], i32[q + 1]]); q += 2; }
      const nk = i32[q++];
      for (let i = 0; i < nk; i++) {
        out.kai.push([i32[q], i32[q + 1], i32[q + 2]]); q += 3;
      }
      const np = i32[q++];
      for (let i = 0; i < np; i++) { out.peng.push([i32[q], i32[q + 1] / 1000]); q += 2; }
      return out;
    },

    /** 从 IndexedDB 读出前沿日志(没有或失败返回 null) */
    async sfReadIDB() {
      const db = await sfOpen();
      if (!db) return null;
      const rec = await new Promise((resolve) => {
        try {
          const rq = db.transaction(SF_STORE, "readonly").objectStore(SF_STORE).get(SF_KEY);
          rq.onsuccess = () => resolve(rq.result || null);
          rq.onerror = () => resolve(null);
        } catch (e) { resolve(null); }
      });
      if (!rec) return null;
      const bytes = rec instanceof Uint8Array ? rec : new Uint8Array(rec);
      return bytes.length >= SF_REC ? bytes : null;
    },
    /** 把日志 blob 回放进主线程实例的缓存 */
    sfLoadBlob(bytes) {
      if (!ready || typeof X.mj_sf_load !== "function" || !bytes) return;
      const n = Math.min(bytes.length - (bytes.length % SF_REC), SF_CAP);
      if (n <= 0) return;
      mem8.set(bytes.subarray(0, n), X.mj_sf_in_ptr());
      X.mj_sf_load(n / SF_REC);
    },
    /** 导出主线程实例的当前日志(拷出; 没有或不可用返回 null) */
    sfDumpLog() {
      if (!ready || typeof X.mj_sf_log_len !== "function") return null;
      const n = X.mj_sf_log_len();
      if (n <= 0) return null;
      const view = new Uint8Array(X.memory.buffer, X.mj_sf_log_ptr(), n * SF_REC);
      return new Uint8Array(view);   // 拷出来, 免得后续写入被改
    },
    /** 把日志 blob 写回 IndexedDB */
    async sfWriteIDB(bytes) {
      if (!bytes || !bytes.length) return;
      const db = await sfOpen();
      if (!db) return;
      try {
        db.transaction(SF_STORE, "readwrite").objectStore(SF_STORE).put(bytes, SF_KEY);
      } catch (e) { /* 写失败无妨 */ }
    },

    /** 暴露纯 JS 实现, 供对拍测试 */
    _js: jsImpl,
  };
})();

/* ================= 分花色前沿缓存落盘的辅助(IndexedDB) ================= */
const SF_DB = "mj159_sf", SF_STORE = "kv", SF_KEY = "suit_front_log_v1";
const SF_REC = 15, SF_CAP = (1 << 18) * SF_REC;   // 与 C 侧 SF_LOG_MAX 一致

function sfOpen() {
  return new Promise((resolve) => {
    if (typeof indexedDB === "undefined") return resolve(null);
    try {
      const req = indexedDB.open(SF_DB, 1);
      req.onupgradeneeded = () => req.result.createObjectStore(SF_STORE);
      req.onsuccess = () => resolve(req.result);
      req.onerror = () => resolve(null);
    } catch (e) { resolve(null); }
  });
}

