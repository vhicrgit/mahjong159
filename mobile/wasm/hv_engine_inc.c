/* 安康159 - 牌型价值 E 引擎(共享实现, 不单独编译)
 *
 * 被两个宿主各自 #include:
 *   mobile/wasm/mjcore.c   (自包含 DFS 向听, 供 wasm/手机)
 *   backend/native/mj159.c (93MB 查表向听, 供桌面 ctypes)
 *
 * 宿主在包含前必须定义:
 *   HV_SHANTEN(c28) -> int    向听(含红中)
 *   HV_IS_WIN(c28)  -> int    判胡
 *   HV_KEY27(c28)   -> u64    前 27 张(非红中)的 63 位打包编码
 * 以及类型 i8/u8/u32/u64、常量 RED/NTILE、mix64()。
 *
 * 可选覆盖: HV_E_BITS(默认 wasm19/原生21), HV_US_BITS(默认19)。
 *
 * 语义必须与 backend/analysis/hand_value.py / mobile|android/js/hand_value.js
 * 逐位一致(浮点累加顺序: wait -> 自摸 -> 换型 -> 碰, 通道内按既定排序)。
 * 改任何一处, 跑 tools/perf/test_hv_c_parity.py 与跨端对拍。
 */

#ifndef HV_E_BITS
#ifdef __wasm__
#define HV_E_BITS 19
#else
#define HV_E_BITS 21
#endif
#endif
#ifndef HV_US_BITS
#define HV_US_BITS 19
#endif

/* 性能计数器(调优用) */
static u64 ct_e_miss, ct_e_hit, ct_us_miss, ct_shanten, ct_fast;
static void hve_stats(u64 *out) {
    out[0] = ct_e_miss; out[1] = ct_e_hit; out[2] = ct_us_miss;
    out[3] = ct_shanten; out[4] = ct_fast;
}
static void hve_stats_reset(void) {
    ct_e_miss = ct_e_hit = ct_us_miss = ct_shanten = ct_fast = 0;
}

static inline int hve_shanten(const i8 *c) { ct_shanten++; return HV_SHANTEN(c); }

/* ---- 有效张掩码缓存(对齐 hand_value.py 的 _USEFUL_SET_CACHE) ---- */
#define HV_US_SIZE (1u << HV_US_BITS)
typedef struct { u64 key; u32 mask; u8 red, used; } HvUsEnt;
static HvUsEnt hve_us_memo[HV_US_SIZE];

static u32 hve_useful_mask(const i8 *c28) {
    u64 key = HV_KEY27(c28);
    int red = c28[RED];
    u32 h = (u32)(mix64(key ^ (0xBF58476D1CE4E5B9ULL * (u64)(red + 1)))
                  & (HV_US_SIZE - 1));
    HvUsEnt *e = &hve_us_memo[h];
    if (e->used && e->key == key && e->red == (u8)red) return e->mask;
    ct_us_miss++;
    i8 t28[NTILE];
    for (int i = 0; i < NTILE; i++) t28[i] = c28[i];
    int base = hve_shanten(t28);
    u32 mask = 0;
    for (int t = 0; t < NTILE; t++) {
        if (t28[t] >= 4) continue;
        t28[t]++;
        int s = hve_shanten(t28);
        t28[t]--;
        if (s < base) mask |= 1u << t;
    }
    e->used = 1; e->key = key; e->red = (u8)red; e->mask = mask;
    return mask;
}

static int hve_ukeire(const i8 *hand, const i8 *u) {
    u32 m = hve_useful_mask(hand);
    int sum = 0;
    while (m) {
        int t = __builtin_ctz(m);
        m &= m - 1;
        if (u[t] > 0) sum += u[t];
    }
    return sum;
}

/* 摸到有效张后打哪张: v10 牌效(只保留向听+进张, 关掉两步推演与放杠风险)。
 * 等价于 native.choose_discard_v10(h14, u, zeros, 0, 100, 1, 0, 0, -1):
 *   向听非最小的候选 score = -10*SW - SW*s (恒为负, 必然落选)
 *   向听最小的候选   score = UW * ukeire
 *   取最大, 严格 > 比较且按 tile 升序扫描 -> 同分取最小 tile */
static int hve_fast_discard(i8 *h14, const i8 *u) {
    ct_fast++;
    int cand[NTILE], cs[NTILE], nc = 0, minsh = 99;
    for (int t = 0; t < NTILE; t++) {
        if (h14[t] <= 0) continue;
        h14[t]--;
        int s = hve_shanten(h14);
        h14[t]++;
        cand[nc] = t; cs[nc] = s; nc++;
        if (s < minsh) minsh = s;
    }
    if (nc == 0) return -1;
    int bestT = -1;
    double best = -1e18;
    for (int k = 0; k < nc; k++) {
        int t = cand[k], s = cs[k];
        double sc;
        if (s > minsh) {
            sc = -10.0 * 100.0 - 100.0 * (double)s;
        } else {
            h14[t]--;
            sc = 1.0 * (double)hve_ukeire(h14, u);
            h14[t]++;
        }
        if (sc > best) { best = sc; bestT = t; }
    }
    return bestT;
}

/* ---- 分析器上下文 ---- */
typedef struct {
    i8 hand0[NTILE];
    i8 u0[NTILE];
    double rho;
    int kaizen, kai_margin, kai_max, kai_topk;
} HvCtx;
static HvCtx hv = { {0}, {0}, 1.0, 0, 2, 1, 6 };

/* ---- E 的记忆化: 键 = (hand, u, kai)。直映射覆盖式: 未命中只是重算 ---- */
#define HV_E_SIZE (1u << HV_E_BITS)
typedef struct { u64 k1, k2; double val; u8 rh, ru, kai, used; } HvEEnt;
static HvEEnt hve_memo[HV_E_SIZE];

/* rho/kaizen/kai 参数变了会让旧条目失效(键里不含这些参数), 所以切换时清表 */
static void hve_memo_clear(void) {
    for (u32 i = 0; i < HV_E_SIZE; i++) hve_memo[i].used = 0;
}

static void hve_set_params(double rho, int kaizen, int kai_margin,
                           int kai_max, int kai_topk) {
    if (hv.rho != rho || hv.kaizen != kaizen || hv.kai_margin != kai_margin
        || hv.kai_max != kai_max || hv.kai_topk != kai_topk)
        hve_memo_clear();
    hv.rho = rho;
    hv.kaizen = kaizen;
    hv.kai_margin = kai_margin;
    hv.kai_max = kai_max;
    hv.kai_topk = kai_topk;
}

static void hve_set_ctx(const i8 *hand, const i8 *visible, double rho,
                        int kaizen, int kai_margin, int kai_max,
                        int kai_topk) {
    hve_set_params(rho, kaizen, kai_margin, kai_max, kai_topk);
    for (int i = 0; i < NTILE; i++) {
        hv.hand0[i] = hand[i];
        int v = 4 - visible[i];
        hv.u0[i] = (i8)(v > 0 ? v : 0);
    }
}

/* ---- 单状态通道枚举(E_rec 与 explain 共用, 保证口径一致) ---- */
typedef struct {
    int s;                          /* 当前向听 */
    int nu, nk, np;
    int ut[NTILE], uw[NTILE];       /* 自摸: tile, 剩余张数 */
    int kt[NTILE], kw[NTILE], kg[NTILE];  /* 换型: tile, 剩余, 进张净增 */
    i8 kh[NTILE][NTILE];            /* 换型后的手牌形态 */
    int pt[27], pd[27];              /* 碰: tile, 碰后最优弃牌 */
    double pw[27];                  /* 碰权重 */
    int N;                          /* 池子总量 */
    double U;                       /* 有效事件总权重 */
} HvCh;

static void hve_channels(const i8 *hand, const i8 *u, int kai, HvCh *ch) {
    int s = hve_shanten(hand);
    ch->s = s;

    /* 自摸通道: 能降向听(或胡)的进张, t 升序 */
    int nu = 0;
    u32 um = hve_useful_mask(hand);
    for (int t = 0; t < NTILE; t++)
        if (((um >> t) & 1u) && u[t] > 0) { ch->ut[nu] = t; ch->uw[nu] = u[t]; nu++; }
    ch->nu = nu;

    /* 碰通道: 可碰且碰后最优向听严格降低的对子, t 升序。
     * 碰后弃牌 = 最小向听优先, 同向听取进张最宽(与 hand_value.py 同口径)。
     * 只判"向听严格更小"会在同向听组里取到牌号最小的那张: 实测碰3条后
     * 选打6条(进张7), 而打2饼进张10, 碰分支 E 被系统性高估。 */
    int np = 0;
    if (hv.rho > 0) {
        i8 h2[NTILE];
        for (int t = 0; t < 27; t++) {
            if (hand[t] != 2 || u[t] <= 0) continue;
            for (int i = 0; i < NTILE; i++) h2[i] = hand[i];
            h2[t] -= 2;
            int bs = 99;
            for (int d = 0; d < NTILE; d++) {
                if (h2[d] <= 0) continue;
                h2[d]--;
                int sd = hve_shanten(h2);
                h2[d]++;
                if (sd < bs) bs = sd;
            }
            if (bs >= s) continue;          /* 碰不降向听 -> 无价值 */
            int bd = -1, bu = -1;
            for (int d = 0; d < NTILE; d++) {
                if (h2[d] <= 0) continue;
                h2[d]--;
                int ud = (hve_shanten(h2) == bs) ? hve_ukeire(h2, u) : -1;
                h2[d]++;
                if (ud > bu) { bu = ud; bd = d; }   /* 同进张保留牌号小的 */
            }
            if (bd < 0) continue;
            ch->pt[np] = t; ch->pw[np] = (double)u[t] * 3.0 * hv.rho;
            ch->pd[np] = bd; np++;
        }
    }
    ch->np = np;

    /* 换型通道(kaizen): 不降向听但让有效张变宽 >= kai_margin。
     * 只保留进张净增最多的 kai_topk 个分支(状态爆炸的保险丝),
     * 排序 (-gain, -w, t升序) 与 Python 端完全一致(浮点累加顺序敏感)。 */
    int nk = 0;
    if (hv.kaizen && kai < hv.kai_max) {
        int uk0 = hve_ukeire(hand, u);
        i8 h14[NTILE];
        for (int t = 0; t < NTILE; t++) {
            if (u[t] <= 0 || ((um >> t) & 1u)) continue;
            for (int i = 0; i < NTILE; i++) h14[i] = hand[i];
            h14[t]++;
            int d = hve_fast_discard(h14, u);
            h14[d]--;
            if (hve_shanten(h14) != s) continue;
            int gain = hve_ukeire(h14, u) - uk0;
            if (gain >= hv.kai_margin) {
                ch->kt[nk] = t; ch->kw[nk] = u[t]; ch->kg[nk] = gain;
                for (int i = 0; i < NTILE; i++) ch->kh[nk][i] = h14[i];
                nk++;
            }
        }
        /* 选择排序出 (-gain, -w, t) 前 kai_topk 名, 剩余丢弃 */
        int cap = hv.kai_topk > 0 && hv.kai_topk < nk ? hv.kai_topk : nk;
        for (int i = 0; i < nk; i++) {
            int bi = i;
            for (int j = i + 1; j < nk; j++)
                if (ch->kg[j] > ch->kg[bi]
                    || (ch->kg[j] == ch->kg[bi] && ch->kw[j] > ch->kw[bi])
                    || (ch->kg[j] == ch->kg[bi] && ch->kw[j] == ch->kw[bi]
                        && ch->kt[j] < ch->kt[bi]))
                    bi = j;
            if (bi != i) {
                int ti;
                ti = ch->kg[i]; ch->kg[i] = ch->kg[bi]; ch->kg[bi] = ti;
                ti = ch->kt[i]; ch->kt[i] = ch->kt[bi]; ch->kt[bi] = ti;
                ti = ch->kw[i]; ch->kw[i] = ch->kw[bi]; ch->kw[bi] = ti;
                for (int j = 0; j < NTILE; j++) {
                    i8 tc = ch->kh[i][j];
                    ch->kh[i][j] = ch->kh[bi][j]; ch->kh[bi][j] = tc;
                }
            }
        }
        nk = cap;
    }
    ch->nk = nk;

    int N = 0;
    for (int i = 0; i < NTILE; i++) N += u[i];
    ch->N = N;
    /* 与 Python 的 (sum(useful) + sum(kai)) + sum(peng) 同序 */
    double sU = 0.0, sK = 0.0, sP = 0.0;
    for (int i = 0; i < nu; i++) sU += (double)ch->uw[i];
    for (int i = 0; i < nk; i++) sK += (double)ch->kw[i];
    for (int i = 0; i < np; i++) sP += ch->pw[i];
    ch->U = sU + sK + sP;
}

static double hve_E_rec(const i8 *hand, const i8 *u, int kai) {
    u64 k1 = HV_KEY27(hand), k2 = HV_KEY27(u);
    u8 rh = (u8)hand[RED], ru = (u8)u[RED];
    u32 h = (u32)(mix64(k1 ^ (k2 * 0x9E3779B97F4A7C15ULL)
                        ^ ((u64)(rh * 40 + ru * 8 + kai) << 3)) & (HV_E_SIZE - 1));
    HvEEnt *e = &hve_memo[h];
    if (e->used && e->k1 == k1 && e->k2 == k2 && e->rh == rh && e->ru == ru
        && e->kai == (u8)kai) { ct_e_hit++; return e->val; }
    ct_e_miss++;

    HvCh ch;
    hve_channels(hand, u, kai, &ch);

    double val;
    if (ch.U <= 0) {
        val = (double)ch.N + 2.0 * (double)ch.s;  /* 有效张耗尽的死手 */
    } else {
        val = ((double)ch.N + 1.0) / (ch.U + 1.0); /* 无放回首次命中的精确期望 */
        i8 h[NTILE], u2[NTILE];
        for (int i = 0; i < ch.nu; i++) {          /* 自摸通道 */
            double p = (double)ch.uw[i] / ch.U;
            int t = ch.ut[i];
            for (int j = 0; j < NTILE; j++) h[j] = hand[j];
            h[t]++;
            if (HV_IS_WIN(h)) continue;          /* 这一摸直接胡, 无后续 */
            for (int j = 0; j < NTILE; j++) u2[j] = u[j];
            u2[t]--;
            int d = hve_fast_discard(h, u2);
            h[d]--;
            /* kai 透传: 换型预算按整条路径计(不重置), 否则每个状态都能
             * 花一次预算, 高向听手牌的 DP 图会爆炸 */
            val += p * hve_E_rec(h, u2, kai);
        }
        for (int i = 0; i < ch.nk; i++) {          /* 换型通道 */
            double p = (double)ch.kw[i] / ch.U;
            for (int j = 0; j < NTILE; j++) u2[j] = u[j];
            u2[ch.kt[i]]--;
            val += p * hve_E_rec(ch.kh[i], u2, kai + 1);
        }
        for (int i = 0; i < ch.np; i++) {          /* 碰通道 */
            double p = ch.pw[i] / ch.U;
            int t = ch.pt[i];
            for (int j = 0; j < NTILE; j++) h[j] = hand[j];
            h[t] -= 2;
            for (int j = 0; j < NTILE; j++) u2[j] = u[j];
            u2[t]--;
            h[ch.pd[i]]--;
            val += p * hve_E_rec(h, u2, kai);
        }
    }
    e->used = 1; e->k1 = k1; e->k2 = k2; e->rh = rh; e->ru = ru;
    e->kai = (u8)kai; e->val = val;
    return val;
}

/* ---- 学者Bot 决策(对齐 backend/ai/bot_hv.py) ---- */
static int hve_choose_discard(void) {
    int bestT = -1;
    double bestE = 1e18;
    i8 h[NTILE];
    for (int t = 0; t < NTILE; t++) {
        if (hv.hand0[t] <= 0) continue;
        for (int i = 0; i < NTILE; i++) h[i] = hv.hand0[i];
        h[t]--;
        double e = hve_E_rec(h, hv.u0, 0);
        if (e < bestE) { bestE = e; bestT = t; }
    }
    return bestT;
}

static int hve_decide_peng(int tile) {
    double eBefore = hve_E_rec(hv.hand0, hv.u0, 0);
    i8 h2[NTILE], h3[NTILE];
    for (int i = 0; i < NTILE; i++) h2[i] = hv.hand0[i];
    h2[tile] -= 2;
    double best = 1e18;
    for (int d = 0; d < NTILE; d++) {
        if (h2[d] <= 0) continue;
        for (int i = 0; i < NTILE; i++) h3[i] = h2[i];
        h3[d]--;
        double e = hve_E_rec(h3, hv.u0, 0);
        if (e < best) best = e;
    }
    return best < eBefore;
}

/* kind: 0=ming, 1=an, 2=bu */
static int hve_decide_gang(int tile, int kind) {
    int before = hve_shanten(hv.hand0);
    i8 c[NTILE];
    for (int i = 0; i < NTILE; i++) c[i] = hv.hand0[i];
    if (kind == 0) c[tile] -= 3;
    else if (kind == 1) c[tile] -= 4;
    else c[tile] -= 1;
    int after = hve_shanten(c);
    return !(before == 0 && after > 0);
}

static double hve_e_after_discard(int tile) {
    i8 h[NTILE];
    for (int i = 0; i < NTILE; i++) h[i] = hv.hand0[i];
    if (h[tile] <= 0) return -1.0;
    h[tile]--;
    return hve_E_rec(h, hv.u0, 0);
}

/* ---- 可解释输出: E 的通道分解 + 明细列表(由宿主转交调用方缓冲) ----
 * outf[0..4] = [E, 等待分量, 自摸分量, 换型分量, 碰分量]
 *   E 与 hve_e_after_discard 逐位一致(相同的顺序累加);
 *   各分量单独求和, 仅供展示, 浮点末位可能与 E 差 1ulp。
 * outi = [nu, (t,剩余)*nu, nk, (t,剩余,净增)*nk, np, (t,权重x1000)*np]
 * outi 至少 192 个 int。 */
static int hve_explain(int tile, double *outf, int *outi) {
    i8 h[NTILE];
    for (int i = 0; i < NTILE; i++) h[i] = hv.hand0[i];
    if (h[tile] <= 0) return -1;
    h[tile]--;
    HvCh ch;
    hve_channels(h, hv.u0, 0, &ch);

    double wait, cU = 0.0, cK = 0.0, cP = 0.0, total;
    if (ch.U <= 0) {
        wait = (double)ch.N + 2.0 * (double)ch.s;
        total = wait;
    } else {
        wait = ((double)ch.N + 1.0) / (ch.U + 1.0);
        total = wait;
        i8 hc[NTILE], u2[NTILE];
        for (int i = 0; i < ch.nu; i++) {          /* 自摸通道 */
            double p = (double)ch.uw[i] / ch.U;
            int t = ch.ut[i];
            for (int j = 0; j < NTILE; j++) hc[j] = h[j];
            hc[t]++;
            if (HV_IS_WIN(hc)) continue;
            for (int j = 0; j < NTILE; j++) u2[j] = hv.u0[j];
            u2[t]--;
            int d = hve_fast_discard(hc, u2);
            hc[d]--;
            double c = p * hve_E_rec(hc, u2, 0);
            total += c;
            cU += c;
        }
        for (int i = 0; i < ch.nk; i++) {          /* 换型通道 */
            double p = (double)ch.kw[i] / ch.U;
            for (int j = 0; j < NTILE; j++) u2[j] = hv.u0[j];
            u2[ch.kt[i]]--;
            double c = p * hve_E_rec(ch.kh[i], u2, 1);
            total += c;
            cK += c;
        }
        for (int i = 0; i < ch.np; i++) {          /* 碰通道 */
            double p = ch.pw[i] / ch.U;
            int t = ch.pt[i];
            for (int j = 0; j < NTILE; j++) hc[j] = h[j];
            hc[t] -= 2;
            for (int j = 0; j < NTILE; j++) u2[j] = hv.u0[j];
            u2[t]--;
            hc[ch.pd[i]]--;
            double c = p * hve_E_rec(hc, u2, 0);
            total += c;
            cP += c;
        }
    }
    outf[0] = total; outf[1] = wait;
    outf[2] = cU; outf[3] = cK; outf[4] = cP;
    int q = 0;
    outi[q++] = ch.nu;
    for (int i = 0; i < ch.nu; i++) { outi[q++] = ch.ut[i]; outi[q++] = ch.uw[i]; }
    outi[q++] = ch.nk;
    for (int i = 0; i < ch.nk; i++) {
        outi[q++] = ch.kt[i]; outi[q++] = ch.kw[i]; outi[q++] = ch.kg[i];
    }
    outi[q++] = ch.np;
    for (int i = 0; i < ch.np; i++) {
        outi[q++] = ch.pt[i]; outi[q++] = (int)(ch.pw[i] * 1000 + 0.5);
    }
    return 0;
}
