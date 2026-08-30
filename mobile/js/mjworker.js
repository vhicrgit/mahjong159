/* 安康159 - 期望巡数(E)的后台计算服务
 *
 * 为什么需要它: mj_hv_e_after_discard 是一次不可中断的同步 wasm 调用。真机实测
 * 单张候选的耗时随换型档位急剧上升:
 *   kaiMax=0   3ms      kaiMax=1  53ms      kaiMax=2  322ms
 * 主线程上算的话, 每张候选都会整块卡住 UI —— 在 requestAnimationFrame 之间让帧
 * 也没用, 因为卡的是那一次调用本身。14 张候选累计约 4.5 秒, 期间换型滑块、关闭
 * 按钮全都点不动。搬到 Worker 后主线程只负责收结果, 始终可响应。
 *
 * 为什么用 Blob URL 而不是 new Worker("js/xxx.js"):
 * Android WebView 从 file:///android_asset/ 加载页面时, 普通 Worker 构造会被拒:
 *   SecurityError: Failed to construct 'Worker': Script at 'file:///android_asset/js/...'
 *   cannot be accessed from origin 'null'
 * Blob URL 的来源继承自创建它的文档, 不受此限制(真机已验证可用)。
 * 同理 worker 里也不能 importScripts 那个 file:// 的 base64 文件, 所以 wasm 字节
 * 由主线程通过 postMessage 传进去, 只传一次。
 *
 * 取消语义: 单次 wasm 调用无法中断, 所以"取消"= 不再派发新任务 + 丢弃已在途结果
 * (靠 gen 代数号判定)。另外重要: 任何时刻只往 worker 发一条消息(见 pump), 因为
 * postMessage 进去的消息拦不住 —— 若随手 post, 快速拖滑块会堆出一串废任务,
 * 真机实测 12 次切换要 61 秒才能算到最新档位。现在废工作恒为 1 条,
 * 它在 worker 线程上跑, 不影响主线程交互。
 *
 * 与 MJWasm 的关系: 两者各持一个独立 wasm 实例(各自的记忆化表)。MJWasm 仍负责
 * 主线程上的同步快路径(shanten/isWin/Bot 决策), 本模块只接管分析面板的 E 计算。
 * worker 不可用时调用方应回退到 MJWasm 的同步版本。
 */
const MJWorker = (() => {
  let worker = null;
  let blobUrl = null;
  let ready = false;
  let initPromise = null;
  let lastError = null;            // 初始化失败原因, 供真机排查(页面 console 到不了 logcat)
  let idCounter = 0;
  let gen = 0;                     // 代数号: bump 一次就作废所有在途任务

  /* 单飞行调度: 任何时刻只往 worker 发一条消息。
   *
   * 为何不能随手 post: postMessage 进的是 worker 的消息队列, 进去就一定会被跑完 ——
   * cancelAll() 只能丢弃主线程这边的 promise, 拦不住已排队的活。早期写法是每个
   * 任务直接 post, 结果快速拖滑块 12 次会在队列里堆出 12 条废任务, 真机实测要
   * 61 秒才能算到最新那一档(而且队列里 0/1/2 交替还会互相冲掉 wasm 的记忆化表,
   * 每条都变成冷算)。限住“只一条在飞”后, 废工作恒为 1 条, 与切换次数无关。 */
  const queue = [];               // 还没派发的任务
  let inflight = null;            // {id, entry, abandoned}
  const pendingById = new Map();  // id -> entry(仅在飞的那一条)

  function pump() {
    if (inflight || !queue.length || !worker) return;
    const entry = queue.shift();
    const id = ++idCounter;
    entry.payload.id = id;
    entry.timer = setTimeout(() => {
      if (inflight && inflight.id === id) {
        inflight = null;
        pendingById.delete(id);
        entry.reject(new Error("worker 超时"));
        pump();
      }
    }, entry.timeoutMs);
    inflight = { id, entry, abandoned: false };
    pendingById.set(id, entry);
    worker.postMessage(entry.payload);
  }

  function post(payload, timeoutMs) {
    return new Promise((resolve, reject) => {
      queue.push({ payload, resolve, reject, timeoutMs: timeoutMs || 60000, timer: null });
      pump();
    });
  }

  function onMessage(ev) {
    const d = ev.data;
    const entry = pendingById.get(d.id);
    const wasInflight = inflight && inflight.id === d.id;
    const abandoned = wasInflight && inflight.abandoned;
    if (wasInflight) inflight = null;
    if (entry) {
      pendingById.delete(d.id);
      clearTimeout(entry.timer);
      if (!abandoned) {
        if (d.err) entry.reject(new Error(d.err));
        else entry.resolve(d);
      }
    }
    pump();                       // 废任务回来也要推动下一条
  }

  /* worker 线程的全部代码。写成函数再 toString, 免去手写模板字符串的转义。
   * 注意: 这个函数体在 worker 里执行, 不能引用外层任何变量。 */
  function workerMain() {
    let X = null;

    function put(ptr, arr) {
      const m = new Int8Array(X.memory.buffer);
      for (let i = 0; i < 28; i++) m[ptr + i] = arr[i];
    }

    // 与 mjwasm.js 的 hvSetup 同口径
    function setup(hc, vis, rho, kai) {
      put(X.mj_buf_ptr(), hc);
      put(X.mj_buf2_ptr(), vis);
      if (typeof X.mj_hv_set2 === "function") {
        X.mj_hv_set2(rho, kai > 0 ? 1 : 0, 2, kai, 6);
      } else {
        X.mj_hv_set(rho, 0);      // 旧版 wasm 没有预算式换型, 一律关掉
      }
    }

    // 与 mjwasm.js 的 hvExplain 逐字段一致
    function readExplain() {
      const f8 = new Float64Array(X.memory.buffer, X.mj_outf_ptr(), 8);
      const i32 = new Int32Array(X.memory.buffer, X.mj_outi_ptr(), 192);
      const out = { E: f8[0], wait: f8[1], cUseful: f8[2], cKai: f8[3],
                    cPeng: f8[4], useful: [], kai: [], peng: [] };
      let q = 0;
      const nu = i32[q++];
      for (let i = 0; i < nu; i++) { out.useful.push([i32[q], i32[q + 1]]); q += 2; }
      const nk = i32[q++];
      for (let i = 0; i < nk; i++) { out.kai.push([i32[q], i32[q + 1], i32[q + 2]]); q += 3; }
      const np = i32[q++];
      for (let i = 0; i < np; i++) { out.peng.push([i32[q], i32[q + 1] / 1000]); q += 2; }
      return out;
    }

    self.onmessage = async (ev) => {
      const d = ev.data;
      try {
        if (d.cmd === "init") {
          const s = atob(d.b64);
          const bytes = new Uint8Array(s.length);
          for (let i = 0; i < s.length; i++) bytes[i] = s.charCodeAt(i);
          const r = await WebAssembly.instantiate(bytes, {});
          X = r.instance.exports;
          // 自检: 111饼222饼333饼444饼+1条 -> 向听 0
          const probe = new Array(28).fill(0);
          probe[9] = 3; probe[10] = 3; probe[11] = 3; probe[12] = 3; probe[0] = 1;
          put(X.mj_buf_ptr(), probe);
          const okSh = X.mj_shanten() === 0;
          self.postMessage({ id: d.id, ok: okSh, set2: typeof X.mj_hv_set2 === "function" });
        } else if (d.cmd === "e") {
          const t0 = Date.now();
          setup(d.hc, d.vis, d.rho, d.kai);
          const e = X.mj_hv_e_after_discard(d.tile);
          self.postMessage({ id: d.id, e, ms: Date.now() - t0 });
        } else if (d.cmd === "sf_load") {
          // 主线程转交的分花色前沿日志: 写入回放区并回放(暖 worker 自己的缓存)
          if (typeof X.mj_sf_load === "function") {
            const bytes = new Uint8Array(d.bytes);
            const m = new Int8Array(X.memory.buffer);
            const ptr = X.mj_sf_in_ptr();
            const cap = typeof X.mj_sf_in_cap === "function"
              ? X.mj_sf_in_cap() : (1 << 18) * 15;
            const n = Math.min(bytes.length, cap);
            for (let i = 0; i < n; i++) m[ptr + i] = bytes[i];
            X.mj_sf_load((n / 15) | 0);
          }
          self.postMessage({ id: d.id, ok: true });
        } else if (d.cmd === "sf_dump") {
          // 导出当前缓存日志(拷出再传, wasm 内存还会继续被写)
          if (typeof X.mj_sf_log_len === "function") {
            const n = X.mj_sf_log_len() * 15;
            const out = new Uint8Array(n);
            out.set(new Uint8Array(X.memory.buffer, X.mj_sf_log_ptr(), n));
            self.postMessage({ id: d.id, bytes: out.buffer }, [out.buffer]);
          } else {
            self.postMessage({ id: d.id, bytes: null });
          }
        } else if (d.cmd === "explain") {
          setup(d.hc, d.vis, d.rho, d.kai);
          if (typeof X.mj_hv_explain !== "function" || X.mj_hv_explain(d.tile) !== 0) {
            self.postMessage({ id: d.id, data: null });
          } else {
            self.postMessage({ id: d.id, data: readExplain() });
          }
        }
      } catch (err) {
        self.postMessage({ id: d.id, err: String(err) });
      }
    };
  }

  /** 把当前局面折算成 (hc, vis) —— 与 mjwasm.js 的 hvSetup 同口径 */
  function snapshot(game, seat) {
    const vis = new Array(28).fill(0);
    for (const q of game.players) {
      for (const t of q.discards) vis[t]++;
      for (const m of q.melds) vis[m.tile] += m.type === "peng" ? 3 : 4;
    }
    const hc = game.players[seat].handCounts();
    for (let t = 0; t < 28; t++) vis[t] += hc[t];
    return { hc: Array.from(hc), vis };
  }

  return {
    get ok() { return ready; },
    get gen() { return gen; },
    /** 初始化失败的原因(字符串); 正常时为 null */
    get lastError() { return lastError; },

    /**
     * 作废所有在途任务。未派发的直接丢, 已在飞的标记为弃用
     * (它还会跑完 —— 单次 wasm 调用无法中断, 但回复会被丢弃)。
     * 因为同时只有一条在飞, 废掉的计算量与切换次数无关。
     */
    cancelAll() {
      gen++;
      while (queue.length) {
        const e = queue.shift();
        clearTimeout(e.timer);
        e.reject(new Error("已取消"));
      }
      if (inflight) {
        inflight.abandoned = true;
        const e = inflight.entry;
        pendingById.delete(inflight.id);
        clearTimeout(e.timer);
        e.reject(new Error("已取消"));
      }
      return gen;
    },

    /** 总是 resolve; 失败时 ok 为 false, 调用方应回退到 MJWasm 同步版。 */
    init() {
      if (initPromise) return initPromise;
      initPromise = (async () => {
        try {
          if (typeof Worker === "undefined" || typeof Blob === "undefined"
              || typeof URL === "undefined" || !URL.createObjectURL
              || typeof MJ_WASM_B64 === "undefined") {
            lastError = "环境不支持 Worker/Blob/URL 或 MJ_WASM_B64 不可见";
            return false;
          }
          const src = "(" + workerMain.toString() + ")()";
          blobUrl = URL.createObjectURL(new Blob([src], { type: "text/javascript" }));
          worker = new Worker(blobUrl);
          worker.addEventListener("message", onMessage);
          worker.addEventListener("error", (e) => {
            lastError = "worker error: " + (e.message || "?")
              + " @" + (e.filename || "") + ":" + (e.lineno || "");
            console.warn("MJWorker " + lastError);
          });
          const r = await post({ cmd: "init", b64: MJ_WASM_B64 }, 30000);
          if (!r.ok) {
            lastError = "worker 内自检未通过: " + JSON.stringify(r);
            return false;
          }
          ready = true;
          return true;
        } catch (e) {
          lastError = String(e);
          console.warn("MJWorker init 失败, 回退主线程: " + lastError);
          ready = false;
          return false;
        }
      })();
      return initPromise;
    },

    /**
     * 打出某张后的期望巡数。返回 Promise<number|null>。
     * myGen 传入调用方持有的代数号; 与当前 gen 不符则直接判为过期。
     */
    async eAfterDiscard(game, seat, rho, tile, kaiMax, myGen) {
      if (!ready) return null;
      if (myGen !== undefined && myGen !== gen) return null;
      const { hc, vis } = snapshot(game, seat);
      if (hc[tile] <= 0) return null;
      const r = await post({ cmd: "e", hc, vis, rho, kai: kaiMax || 0, tile });
      return r.e;
    },

    /** 载入分花色前沿日志(暖 worker 的缓存; 静默失败) */
    async sfLoad(bytes) {
      if (!ready || !bytes) return;
      try { await post({ cmd: "sf_load", bytes }, 30000); } catch (e) {}
    },
    /** 导出 worker 的前沿日志(供落盘); 未就绪或失败返回 null */
    async sfDump() {
      if (!ready) return null;
      try {
        const r = await post({ cmd: "sf_dump" }, 30000);
        return r.bytes ? new Uint8Array(r.bytes) : null;
      } catch (e) { return null; }
    },

    /** 候选行的展开解释。返回 Promise<object|null>。 */
    async explain(game, seat, rho, tile, kaiMax) {
      if (!ready) return null;
      const { hc, vis } = snapshot(game, seat);
      if (hc[tile] <= 0) return null;
      const r = await post({ cmd: "explain", hc, vis, rho, kai: kaiMax || 0, tile });
      return r.data;
    },
  };
})();
