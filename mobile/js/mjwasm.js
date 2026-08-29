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
  function hvSetup(game, seat, rho) {
    const visible = new Array(28).fill(0);
    for (const q of game.players) {
      for (const t of q.discards) visible[t]++;
      for (const m of q.melds) visible[m.tile] += m.type === "peng" ? 3 : 4;
    }
    const hc = game.players[seat].handCounts();
    for (let t = 0; t < 28; t++) visible[t] += hc[t];
    put(P1, hc);
    put(P2, visible);
    X.mj_hv_set(rho, 0);            // kaizen=0: 与 bot_hv 对战口径一致
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
    /** 打出某张后的期望巡数(分析面板可用); 未就绪返回 null */
    hvEAfterDiscard(game, seat, rho, tile) {
      if (!ready) return null;
      hvSetup(game, seat, rho);
      return X.mj_hv_e_after_discard(tile);
    },

    /** 暴露纯 JS 实现, 供对拍测试 */
    _js: jsImpl,
  };
})();
