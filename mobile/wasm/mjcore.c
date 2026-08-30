/* 安康159 - 规则核心的自包含 C 实现(可编 wasm32-freestanding)
 *
 * 与 backend/rules/win.py 逐行对齐, 三方(Python / JS / 本文件)必须保持一致,
 * 用 mobile/_gen_cases.js 生成的样例交叉验证 0 不一致。
 *
 * 与 backend/native/mj159.c 的区别: 那个版本靠 112MB 预计算大表(mmap),
 * 装不进 APK 且 wasm 没有 mmap。本文件是纯 DFS + 记忆化, 完全自包含,
 * 不依赖 libc, 不需要任何数据文件, 编出来只有几十 KB。
 *
 * 算法: 标准胡牌型 = 4面子(顺/刻) + 1对将, 红中可当任意牌。
 *   mj_is_win  : 枚举将的位置(含红中凑将), 余牌用 all_melds 检查能否全组面子
 *   mj_shanten : DFS 求 (面子m, 搭子t, 将p) 的 Pareto 前沿,
 *                shanten = min over 前沿 of 2*need - 2m - min(t,need-m) - min(p,1)
 *                need 由手牌张数推导, 原生支持副露手
 *
 * 编译(原生, 供对拍测试):
 *   cc -O3 -fPIC -shared -o libmjcore.so mjcore.c
 * 编译(wasm32, 无 libc):
 *   zig cc -target wasm32-freestanding -O3 -nostdlib \
 *       -Wl,--no-entry -Wl,--export-dynamic -o mjcore.wasm mjcore.c
 */

typedef unsigned char u8;
typedef signed char i8;
typedef unsigned int u32;
typedef unsigned long long u64;

#define RED 27
#define NTILE 28

/* 性能计数器(原生对拍/调优用; wasm 也可用) */
static u64 ct_e_miss, ct_e_hit, ct_us_miss, ct_shanten, ct_fast;

/* ---------------- 牌面工具(对齐 rules/tiles.py) ---------------- */
/* 0-8 条, 9-17 饼, 18-26 万, 27 红中 */
static inline int suit_of(int t) { return t / 9; }        /* t<27 时 0/1/2 */
static inline int rank_of(int t) { return t % 9 + 1; }    /* 1-based */

/* ---------------- Pareto 前沿: 256 位集合 ----------------
 * idx = (m<<5) | (t<<2) | p, m/t 各 0..7, p 0..3
 * prune 后 m<=need<=4, t<=need<=4, p<=1; 加 delta 后最多 m=5 t=5 p=2, 不越界 */
typedef struct { u64 w[4]; } Front;

static inline void fr_clear(Front *f) { f->w[0] = f->w[1] = f->w[2] = f->w[3] = 0; }
static inline void fr_set(Front *f, int idx) { f->w[idx >> 6] |= 1ULL << (idx & 63); }
static inline int fr_test(const Front *f, int idx) {
    return (f->w[idx >> 6] >> (idx & 63)) & 1ULL;
}
static inline int idx_of(int m, int t, int p) { return (m << 5) | (t << 2) | p; }

/* 把 sub 前沿整体加上 (dm,dt,dp) 后并入 dst */
static void fr_add(Front *dst, const Front *sub, int dm, int dt, int dp) {
    for (int b = 0; b < 4; b++) {
        u64 w = sub->w[b];
        while (w) {
            int bit = __builtin_ctzll(w);
            w &= w - 1;
            int idx = (b << 6) | bit;
            int m = (idx >> 5) & 7, t = (idx >> 2) & 7, p = idx & 3;
            fr_set(dst, idx_of(m + dm, t + dt, p + dp));
        }
    }
}

/* 截断到公式上限后保留分量支配意义下的 Pareto 前沿。
 *
 * 实现注意: 前沿实际只有 3-8 个点, 所以先把点收集到小数组再做 O(k^2) 支配判断。
 * 早期版本直接在 5x5x2 全格上做双重扫描(最多 2500 次迭代), 成了主要热点。 */
static void fr_prune(Front *f, int need) {
    int pm[64], pt[64], pp[64], k = 0;
    for (int b = 0; b < 4; b++) {
        u64 w = f->w[b];
        while (w) {
            int bit = __builtin_ctzll(w);
            w &= w - 1;
            int idx = (b << 6) | bit;
            int m = (idx >> 5) & 7, t = (idx >> 2) & 7, p = idx & 3;
            if (m > need) m = need;
            if (t > need) t = need;
            if (p > 1) p = 1;
            /* 去重(截断后可能撞在一起) */
            int dup = 0;
            for (int i = 0; i < k; i++)
                if (pm[i] == m && pt[i] == t && pp[i] == p) { dup = 1; break; }
            if (!dup && k < 64) { pm[k] = m; pt[k] = t; pp[k] = p; k++; }
        }
    }
    Front out;
    fr_clear(&out);
    for (int i = 0; i < k; i++) {
        int dominated = 0;
        for (int j = 0; j < k; j++) {
            if (j == i) continue;
            if (pm[j] >= pm[i] && pt[j] >= pt[i] && pp[j] >= pp[i]) { dominated = 1; break; }
        }
        if (!dominated) fr_set(&out, idx_of(pm[i], pt[i], pp[i]));
    }
    *f = out;
}

/* ---------------- 记忆化(直接映射, 冲突即覆盖; 纯函数故永不失效) ---------------- */
#define DFS_BITS 19
#define DFS_SIZE (1u << DFS_BITS)
#define AM_BITS 17
#define AM_SIZE (1u << AM_BITS)

typedef struct { u64 key; Front f; u8 red, need, used; } DfsEnt;
typedef struct { u64 key; u8 red, val, used; } AmEnt;

static DfsEnt dfs_memo[DFS_SIZE];
static AmEnt am_memo[AM_SIZE];

void mjc_hv_stats(u64 *out) {
    out[0] = ct_e_miss; out[1] = ct_e_hit; out[2] = ct_us_miss;
    out[3] = ct_shanten; out[4] = ct_fast;
}
void mjc_hv_stats_reset(void) {
    ct_e_miss = ct_e_hit = ct_us_miss = ct_shanten = ct_fast = 0;
}

static inline u64 mix64(u64 x) {
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33;
    x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33;
    return x;
}

/* 27 张计数编码成 63 位: 每花色 9 张做 5 进制(5^9 = 1953125 < 2^21) */
static u64 enc27(const i8 *c) {
    u32 a = 0, b = 0, d = 0;
    for (int i = 8; i >= 0; i--) a = a * 5 + (u32)c[i];
    for (int i = 17; i >= 9; i--) b = b * 5 + (u32)c[i];
    for (int i = 26; i >= 18; i--) d = d * 5 + (u32)c[i];
    return (u64)a | ((u64)b << 21) | ((u64)d << 42);
}

/* ---------------- all_melds: counts(3的倍数)能否在红中辅助下全组面子 ---------------- */
static int all_melds(i8 *c, int red);

static int all_melds_cached(i8 *c, int red) {
    u64 key = enc27(c);
    u32 h = (u32)(mix64(key ^ (0x9E3779B97F4A7C15ULL * (u64)(red + 1))) & (AM_SIZE - 1));
    AmEnt *e = &am_memo[h];
    if (e->used && e->key == key && e->red == (u8)red) return e->val;
    int v = all_melds(c, red);
    e->used = 1; e->key = key; e->red = (u8)red; e->val = (u8)v;
    return v;
}

static int all_melds(i8 *c, int red) {
    int t = -1;
    for (int i = 0; i < 27; i++)
        if (c[i] > 0) { t = i; break; }
    if (t == -1) return red % 3 == 0;

    /* 刻子 */
    if (c[t] >= 3) {
        c[t] -= 3;
        int ok = all_melds_cached(c, red);
        c[t] += 3;
        if (ok) return 1;
    }
    if (c[t] >= 2 && red >= 1) {
        c[t] -= 2;
        int ok = all_melds_cached(c, red - 1);
        c[t] += 2;
        if (ok) return 1;
    }
    if (c[t] >= 1 && red >= 2) {
        c[t] -= 1;
        int ok = all_melds_cached(c, red - 2);
        c[t] += 1;
        if (ok) return 1;
    }
    /* 顺子: t 可为头/中/尾张。t 是最小现存牌, 起点低于 t 的位置必然缺牌、由红中补
     * (历史bug: 只试以 t 为起点 -> "89+红中"等永不识别) */
    int s = suit_of(t);
    for (int d = -2; d <= 0; d++) {
        int start = t + d;
        if (start < 0 || suit_of(start) != s || rank_of(start) > 7) continue;
        int x0 = start, x1 = start + 1, x2 = start + 2;
        int u0 = c[x0] >= 1, u1 = c[x1] >= 1, u2 = c[x2] >= 1;
        int need = 3 - u0 - u1 - u2;
        if (need > red) continue;
        c[x0] -= u0; c[x1] -= u1; c[x2] -= u2;
        int ok = all_melds_cached(c, red - need);
        c[x0] += u0; c[x1] += u1; c[x2] += u2;
        if (ok) return 1;
    }
    return 0;
}

/* ---------------- DFS: (m,t,p) 的 Pareto 前沿 ---------------- */
static void dfs_front(i8 *c, int red_left, int need, Front *out);

static void dfs_cached(i8 *c, int red_left, int need, Front *out) {
    u64 key = enc27(c);
    u32 h = (u32)(mix64(key ^ (0xD6E8FEB86659FD93ULL * (u64)(red_left * 8 + need)))
                  & (DFS_SIZE - 1));
    DfsEnt *e = &dfs_memo[h];
    if (e->used && e->key == key && e->red == (u8)red_left && e->need == (u8)need) {
        *out = e->f;
        return;
    }
    dfs_front(c, red_left, need, out);
    e->used = 1; e->key = key; e->red = (u8)red_left; e->need = (u8)need; e->f = *out;
}

static void dfs_front(i8 *c, int red_left, int need, Front *out) {
    int t = -1;
    for (int i = 0; i < 27; i++)
        if (c[i] > 0) { t = i; break; }

    Front cands;
    fr_clear(&cands);

    if (t == -1) {
        int m = red_left / 3, rem = red_left % 3;
        if (rem == 2) {
            fr_set(&cands, idx_of(m, 0, 1));
            fr_set(&cands, idx_of(m, 1, 0));
        } else {
            fr_set(&cands, idx_of(m, 0, 0));
        }
        fr_prune(&cands, need);
        *out = cands;
        return;
    }

    Front sub;

    /* 选项1: 孤张跳过 */
    c[t] -= 1;
    dfs_cached(c, red_left, need, &sub);
    c[t] += 1;
    fr_add(&cands, &sub, 0, 0, 0);

    /* 选项2: 对子(将 或 刻子搭子; 历史bug4: 旧版对子只当将) */
    if (c[t] >= 2) {
        c[t] -= 2;
        dfs_cached(c, red_left, need, &sub);
        c[t] += 2;
        fr_add(&cands, &sub, 0, 0, 1);
        fr_add(&cands, &sub, 0, 1, 0);
    }
    if (c[t] >= 1 && red_left >= 1) {
        c[t] -= 1;
        dfs_cached(c, red_left - 1, need, &sub);
        c[t] += 1;
        fr_add(&cands, &sub, 0, 0, 1);
    }

    /* 选项3: 刻子 */
    if (c[t] >= 3) {
        c[t] -= 3;
        dfs_cached(c, red_left, need, &sub);
        c[t] += 3;
        fr_add(&cands, &sub, 1, 0, 0);
    }
    if (c[t] >= 2 && red_left >= 1) {
        c[t] -= 2;
        dfs_cached(c, red_left - 1, need, &sub);
        c[t] += 2;
        fr_add(&cands, &sub, 1, 0, 0);
    }

    int s = suit_of(t);
    /* 选项4: 顺子面子(t 可为头/中/尾张, 缺牌由红中补) */
    for (int d = -2; d <= 0; d++) {
        int start = t + d;
        if (start < 0 || suit_of(start) != s || rank_of(start) > 7) continue;
        int x0 = start, x1 = start + 1, x2 = start + 2;
        int u0 = c[x0] >= 1, u1 = c[x1] >= 1, u2 = c[x2] >= 1;
        int need_red = 3 - u0 - u1 - u2;
        if (need_red > red_left) continue;
        c[x0] -= u0; c[x1] -= u1; c[x2] -= u2;
        dfs_cached(c, red_left - need_red, need, &sub);
        c[x0] += u0; c[x1] += u1; c[x2] += u2;
        fr_add(&cands, &sub, 1, 0, 0);
    }

    /* 选项5: 搭子(历史bug2: 旧代码 r<=7 门禁把 8/9 点搭子全部跳过) */
    int r = rank_of(t);
    if (r <= 8 && c[t + 1] >= 1) {          /* 两面/边张 t,t+1 */
        c[t] -= 1; c[t + 1] -= 1;
        dfs_cached(c, red_left, need, &sub);
        c[t] += 1; c[t + 1] += 1;
        fr_add(&cands, &sub, 0, 1, 0);
    }
    if (r <= 7 && c[t + 2] >= 1) {          /* 嵌张 t,t+2 */
        c[t] -= 1; c[t + 2] -= 1;
        dfs_cached(c, red_left, need, &sub);
        c[t] += 1; c[t + 2] += 1;
        fr_add(&cands, &sub, 0, 1, 0);
    }
    if (red_left >= 1) {                    /* 红中搭子 */
        c[t] -= 1;
        dfs_cached(c, red_left - 1, need, &sub);
        c[t] += 1;
        fr_add(&cands, &sub, 0, 1, 0);
    }

    fr_prune(&cands, need);
    *out = cands;
}

/* ---------------- 对外接口 ---------------- */

/* JS 通过这块共享缓冲区传入手牌(28 个 int8) */
static i8 g_buf[NTILE];

__attribute__((export_name("mj_buf_ptr")))
int mj_buf_ptr(void) { return (int)(long)g_buf; }

/* shanten 顶层结果缓存: E 负载里同一手牌被 fast_discard/kaizen 反复评估,
 * 直映射 2^21 槽(约 24MB)。键 = enc27(前27张) + 红中数。 */
#define SH_BITS 21
#define SH_SIZE (1u << SH_BITS)
typedef struct { u64 key; i8 val; u8 red, used; } ShEnt;
static ShEnt sh_memo[SH_SIZE];

static int shanten_slow(const i8 *cin);

static int shanten_impl(const i8 *cin) {
    ct_shanten++;
    u64 key = enc27(cin);
    int red = cin[RED];
    u32 h = (u32)(mix64(key ^ (0xC2B2AE3D27D4EB4FULL * (u64)(red + 1)))
                  & (SH_SIZE - 1));
    ShEnt *e = &sh_memo[h];
    if (e->used && e->key == key && e->red == (u8)red) return e->val;
    int v = shanten_slow(cin);
    e->used = 1; e->key = key; e->red = (u8)red; e->val = (i8)v;
    return v;
}

static int shanten_slow(const i8 *cin) {
    i8 c[27];
    int red = cin[RED], total = red;
    for (int i = 0; i < 27; i++) { c[i] = cin[i]; total += cin[i]; }

    /* need 由手牌张数推导: 13张->4, 副露1副的10张->3。C 的整数除法向零取整,
     * total=0 时 (0-1)/3=0 也进特判, 与 Python 的地板除结果一致(都走 total<=1) */
    int need = (total - 1) / 3;
    /* need<=0 特判(四副露只差将): 公式 2*need-2m-t-p 表达不了"只差将"。
     * 不能用 max(1,need) 兜底 —— 那会把 1 张暗牌算成向听 2, 与判胡自相矛盾 */
    if (need <= 0) {
        if (total <= 1) return 0;
        if (red >= 1) return -1;
        for (int i = 0; i < 27; i++)
            if (c[i] >= 2) return -1;
        return 1;
    }

    Front f;
    dfs_cached(c, red, need, &f);
    int best = 99;
    for (int b = 0; b < 4; b++) {
        u64 w = f.w[b];
        while (w) {
            int bit = __builtin_ctzll(w);
            w &= w - 1;
            int idx = (b << 6) | bit;
            int m = (idx >> 5) & 7, t = (idx >> 2) & 7, p = idx & 3;
            int tcap = need - m;
            if (t > tcap) t = tcap;
            int v = 2 * need - 2 * m - t - p;
            if (v < best) best = v;
        }
    }
    return best;
}

static int is_win_impl(const i8 *cin) {
    int total = 0;
    for (int i = 0; i < NTILE; i++) total += cin[i];
    if (total % 3 != 2) return 0;
    int red = cin[RED];
    i8 base[27];
    for (int i = 0; i < 27; i++) base[i] = cin[i];

    /* 将 = 普通对子, 可用 0/1/2 张红中凑 */
    for (int t = 0; t < 27; t++) {
        for (int need = 0; need <= 2; need++) {
            if (need > red) continue;
            int take = 2 - need;
            if (base[t] + need >= 2 && base[t] >= take) {
                base[t] -= take;
                int ok = all_melds_cached(base, red - need);
                base[t] += take;
                if (ok) return 1;
            }
        }
    }
    /* 将 = 两张红中 */
    if (red >= 2 && all_melds_cached(base, red - 2)) return 1;
    return 0;
}

__attribute__((export_name("mj_shanten")))
int mj_shanten(void) { return shanten_impl(g_buf); }

__attribute__((export_name("mj_is_win")))
int mj_is_win(void) { return is_win_impl(g_buf); }

/* 摸到能降向听(含直接胡)的牌集合, 返回 28 位掩码。
 * 对齐 hand_value.py 的 useful_set / analyzer 的有效进张口径。
 * 一次调用省掉 JS 侧 28 次跨界调用, 是学者Bot 的主要热点。 */
__attribute__((export_name("mj_useful_mask")))
int mj_useful_mask(void) {
    int base = shanten_impl(g_buf);
    u32 mask = 0;
    for (int t = 0; t < NTILE; t++) {
        if (g_buf[t] >= 4) continue;
        g_buf[t]++;
        int s = shanten_impl(g_buf);
        g_buf[t]--;
        if (s < base) mask |= 1u << t;
    }
    return (int)mask;
}

/* ================= 牌型价值分析器(HandAnalyzer) =================
 *
 * 与 backend/analysis/hand_value.py 逐行对齐。浮点累加顺序必须一致
 * (wait -> 自摸通道 -> 换型通道 -> 碰通道, 每条通道内按 t 升序),
 * 否则末位会出现差异。禁用 -ffast-math。
 *
 * 为什么把整个 E 递归搬进 C: 学者Bot 单次决策要算 22 万次 shanten。
 * 如果只把 shanten 搬进 wasm, 就会发生 22 万次 JS-wasm 跨界调用, 开销吃掉收益。
 * 整体搬进来后一次决策只跨界一次。
 */

/* ---- 有效张掩码缓存(对齐 hand_value.py 的 _USEFUL_SET_CACHE) ---- */
#define US_BITS 19
#define US_SIZE (1u << US_BITS)
typedef struct { u64 key; u32 mask; u8 red, used; } UsEnt;
static UsEnt us_memo[US_SIZE];

static u32 useful_mask_of(const i8 *c28) {
    u64 key = enc27(c28);
    int red = c28[RED];
    u32 h = (u32)(mix64(key ^ (0xBF58476D1CE4E5B9ULL * (u64)(red + 1))) & (US_SIZE - 1));
    UsEnt *e = &us_memo[h];
    if (e->used && e->key == key && e->red == (u8)red) return e->mask;
    ct_us_miss++;

    i8 t28[NTILE];
    for (int i = 0; i < NTILE; i++) t28[i] = c28[i];
    int base = shanten_impl(t28);
    u32 mask = 0;
    for (int t = 0; t < NTILE; t++) {
        if (t28[t] >= 4) continue;
        t28[t]++;
        int s = shanten_impl(t28);
        t28[t]--;
        if (s < base) mask |= 1u << t;
    }
    e->used = 1; e->key = key; e->red = (u8)red; e->mask = mask;
    return mask;
}

static int ukeire_of(const i8 *hand, const i8 *u) {
    u32 m = useful_mask_of(hand);
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
static int fast_discard(i8 *h14, const i8 *u) {
    ct_fast++;
    int cand[NTILE], cs[NTILE], nc = 0, minsh = 99;
    for (int t = 0; t < NTILE; t++) {
        if (h14[t] <= 0) continue;
        h14[t]--;
        int s = shanten_impl(h14);
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
            sc = 1.0 * (double)ukeire_of(h14, u);
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

/* ---- E 的记忆化: 键 = (hand, u, kai) ----
 * 换型层的 DP 状态数单手可达数万~几十万(全表共享时), 2^17 会严重抖动,
 * 给到 2^19(约 16MB 静态内存)。直映射覆盖式缓存: 未命中只是重算, 不影响正确性 */
#ifdef __wasm__
#define E_BITS 19
#else
#define E_BITS 21   /* 原生端内存宽裕, 直映射冲突少 -> 命中率高 */
#endif
#define E_SIZE (1u << E_BITS)
typedef struct { u64 k1, k2; double val; u8 rh, ru, kai, used; } EEnt;
static EEnt e_memo[E_SIZE];


/* rho/kaizen/kai 参数变了会让旧条目失效(键里不含这些参数), 所以切换时清表 */
static void e_memo_clear(void) {
    for (u32 i = 0; i < E_SIZE; i++) e_memo[i].used = 0;
}

/* ---- 单状态通道枚举(E_rec 与 hv_explain 共用, 保证口径一致) ---- */
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

static void hv_channels(const i8 *hand, const i8 *u, int kai, HvCh *ch) {
    int s = shanten_impl(hand);
    ch->s = s;

    /* 自摸通道: 能降向听(或胡)的进张, t 升序 */
    int nu = 0;
    u32 um = useful_mask_of(hand);
    for (int t = 0; t < NTILE; t++)
        if (((um >> t) & 1u) && u[t] > 0) { ch->ut[nu] = t; ch->uw[nu] = u[t]; nu++; }
    ch->nu = nu;

    /* 碰通道: 可碰且碰后最优向听严格降低的对子, t 升序 */
    int np = 0;
    if (hv.rho > 0) {
        i8 h2[NTILE];
        for (int t = 0; t < 27; t++) {
            if (hand[t] != 2 || u[t] <= 0) continue;
            for (int i = 0; i < NTILE; i++) h2[i] = hand[i];
            h2[t] -= 2;
            int bd = -1, bs = 99;
            for (int d = 0; d < NTILE; d++) {
                if (h2[d] <= 0) continue;
                h2[d]--;
                int sd = shanten_impl(h2);
                h2[d]++;
                if (sd < bs) { bs = sd; bd = d; }
            }
            if (bs < s) {
                ch->pt[np] = t; ch->pw[np] = (double)u[t] * 3.0 * hv.rho;
                ch->pd[np] = bd; np++;
            }
        }
    }
    ch->np = np;

    /* 换型通道(kaizen): 不降向听但让有效张变宽 >= kai_margin。
     * 只保留进张净增最多的 kai_topk 个分支(状态爆炸的保险丝),
     * 排序 (-gain, -w, t升序) 与 Python 端完全一致(浮点累加顺序敏感)。 */
    int nk = 0;
    if (hv.kaizen && kai < hv.kai_max) {
        int uk0 = ukeire_of(hand, u);
        i8 h14[NTILE];
        for (int t = 0; t < NTILE; t++) {
            if (u[t] <= 0 || ((um >> t) & 1u)) continue;
            for (int i = 0; i < NTILE; i++) h14[i] = hand[i];
            h14[t]++;
            int d = fast_discard(h14, u);
            h14[d]--;
            if (shanten_impl(h14) != s) continue;
            int gain = ukeire_of(h14, u) - uk0;
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

static double E_rec(const i8 *hand, const i8 *u, int kai) {
    u64 k1 = enc27(hand), k2 = enc27(u);
    u8 rh = (u8)hand[RED], ru = (u8)u[RED];
    u32 h = (u32)(mix64(k1 ^ (k2 * 0x9E3779B97F4A7C15ULL)
                        ^ ((u64)(rh * 40 + ru * 8 + kai) << 3)) & (E_SIZE - 1));
    EEnt *e = &e_memo[h];
    if (e->used && e->k1 == k1 && e->k2 == k2 && e->rh == rh && e->ru == ru
        && e->kai == (u8)kai) { ct_e_hit++; return e->val; }
    ct_e_miss++;

    HvCh ch;
    hv_channels(hand, u, kai, &ch);

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
            if (is_win_impl(h)) continue;        /* 这一摸直接胡, 无后续 */
            for (int j = 0; j < NTILE; j++) u2[j] = u[j];
            u2[t]--;
            int d = fast_discard(h, u2);
            h[d]--;
            /* kai 透传: 换型预算按整条路径计(不重置), 否则每个状态都能
             * 花一次预算, 高向听手牌的 DP 图会爆炸 */
            val += p * E_rec(h, u2, kai);
        }
        for (int i = 0; i < ch.nk; i++) {          /* 换型通道 */
            double p = (double)ch.kw[i] / ch.U;
            for (int j = 0; j < NTILE; j++) u2[j] = u[j];
            u2[ch.kt[i]]--;
            val += p * E_rec(ch.kh[i], u2, kai + 1);
        }
        for (int i = 0; i < ch.np; i++) {          /* 碰通道 */
            double p = ch.pw[i] / ch.U;
            int t = ch.pt[i];
            for (int j = 0; j < NTILE; j++) h[j] = hand[j];
            h[t] -= 2;
            for (int j = 0; j < NTILE; j++) u2[j] = u[j];
            u2[t]--;
            h[ch.pd[i]]--;
            val += p * E_rec(h, u2, kai);
        }
    }
    e->used = 1; e->k1 = k1; e->k2 = k2; e->rh = rh; e->ru = ru;
    e->kai = (u8)kai; e->val = val;
    return val;
}

/* ---- 可解释输出: E 的通道分解 + 明细列表 ----
 * g_outf[0..4] = [E, 等待分量, 自摸分量, 换型分量, 碰分量]
 *   E 与 mj_hv_e_after_discard 逐位一致(相同的顺序累加);
 *   各分量单独求和, 仅供展示, 浮点末位可能与 E 差 1ulp。
 * g_outi = [nu, (t,剩余)*nu, nk, (t,剩余,净增)*nk, np, (t,权重x1000)*np] */
static double g_outf[8];
static int g_outi[192];

__attribute__((export_name("mj_outf_ptr")))
int mj_outf_ptr(void) { return (int)(long)g_outf; }
__attribute__((export_name("mj_outi_ptr")))
int mj_outi_ptr(void) { return (int)(long)g_outi; }

static int hv_explain_tile(int tile) {
    i8 h[NTILE];
    for (int i = 0; i < NTILE; i++) h[i] = hv.hand0[i];
    if (h[tile] <= 0) return -1;
    h[tile]--;
    HvCh ch;
    hv_channels(h, hv.u0, 0, &ch);

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
            if (is_win_impl(hc)) continue;
            for (int j = 0; j < NTILE; j++) u2[j] = hv.u0[j];
            u2[t]--;
            int d = fast_discard(hc, u2);
            hc[d]--;
            double c = p * E_rec(hc, u2, 0);
            total += c;
            cU += c;
        }
        for (int i = 0; i < ch.nk; i++) {          /* 换型通道 */
            double p = (double)ch.kw[i] / ch.U;
            for (int j = 0; j < NTILE; j++) u2[j] = hv.u0[j];
            u2[ch.kt[i]]--;
            double c = p * E_rec(ch.kh[i], u2, 1);
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
            double c = p * E_rec(hc, u2, 0);
            total += c;
            cP += c;
        }
    }
    g_outf[0] = total; g_outf[1] = wait;
    g_outf[2] = cU; g_outf[3] = cK; g_outf[4] = cP;
    int q = 0;
    g_outi[q++] = ch.nu;
    for (int i = 0; i < ch.nu; i++) { g_outi[q++] = ch.ut[i]; g_outi[q++] = ch.uw[i]; }
    g_outi[q++] = ch.nk;
    for (int i = 0; i < ch.nk; i++) {
        g_outi[q++] = ch.kt[i]; g_outi[q++] = ch.kw[i]; g_outi[q++] = ch.kg[i];
    }
    g_outi[q++] = ch.np;
    for (int i = 0; i < ch.np; i++) {
        g_outi[q++] = ch.pt[i]; g_outi[q++] = (int)(ch.pw[i] * 1000 + 0.5);
    }
    return 0;
}

__attribute__((export_name("mj_hv_explain")))
int mj_hv_explain(int tile) { return hv_explain_tile(tile); }

/* 原生对拍入口: 指针返回值在原生 64 位会被 int 截断, 改为调用方传缓冲 */
int mjc_hv_explain_buf(int tile, double *outf, int *outi) {
    int rc = hv_explain_tile(tile);
    if (rc == 0) {
        for (int i = 0; i < 8; i++) outf[i] = g_outf[i];
        for (int i = 0; i < 192; i++) outi[i] = g_outi[i];
    }
    return rc;
}

/* ---- 学者Bot 决策(对齐 backend/ai/bot_hv.py) ---- */
static int hv_choose_discard(void) {
    int bestT = -1;
    double bestE = 1e18;
    i8 h[NTILE];
    for (int t = 0; t < NTILE; t++) {
        if (hv.hand0[t] <= 0) continue;
        for (int i = 0; i < NTILE; i++) h[i] = hv.hand0[i];
        h[t]--;
        double e = E_rec(h, hv.u0, 0);
        if (e < bestE) { bestE = e; bestT = t; }
    }
    return bestT;
}

static int hv_decide_peng(int tile) {
    double eBefore = E_rec(hv.hand0, hv.u0, 0);
    i8 h2[NTILE], h3[NTILE];
    for (int i = 0; i < NTILE; i++) h2[i] = hv.hand0[i];
    h2[tile] -= 2;
    double best = 1e18;
    for (int d = 0; d < NTILE; d++) {
        if (h2[d] <= 0) continue;
        for (int i = 0; i < NTILE; i++) h3[i] = h2[i];
        h3[d]--;
        double e = E_rec(h3, hv.u0, 0);
        if (e < best) best = e;
    }
    return best < eBefore;
}

/* kind: 0=ming, 1=an, 2=bu */
static int hv_decide_gang(int tile, int kind) {
    int before = shanten_impl(hv.hand0);
    i8 c[NTILE];
    for (int i = 0; i < NTILE; i++) c[i] = hv.hand0[i];
    if (kind == 0) c[tile] -= 3;
    else if (kind == 1) c[tile] -= 4;
    else c[tile] -= 1;
    int after = shanten_impl(c);
    return !(before == 0 && after > 0);
}

/* ---- 导出: g_buf=手牌(28), g_buf2=可见牌(28) ---- */
static i8 g_buf2[NTILE];

__attribute__((export_name("mj_buf2_ptr")))
int mj_buf2_ptr(void) { return (int)(long)g_buf2; }

static void hv_set_params(double rho, int kaizen, int kai_margin,
                          int kai_max, int kai_topk) {
    if (hv.rho != rho || hv.kaizen != kaizen || hv.kai_margin != kai_margin
        || hv.kai_max != kai_max || hv.kai_topk != kai_topk)
        e_memo_clear();
    hv.rho = rho;
    hv.kaizen = kaizen;
    hv.kai_margin = kai_margin;
    hv.kai_max = kai_max;
    hv.kai_topk = kai_topk;
}

__attribute__((export_name("mj_hv_set2")))
void mj_hv_set2(double rho, int kaizen, int kai_margin, int kai_max,
                int kai_topk) {
    hv_set_params(rho, kaizen, kai_margin, kai_max, kai_topk);
    for (int i = 0; i < NTILE; i++) {
        hv.hand0[i] = g_buf[i];
        int v = 4 - g_buf2[i];
        hv.u0[i] = (i8)(v > 0 ? v : 0);
    }
}

__attribute__((export_name("mj_hv_set")))
void mj_hv_set(double rho, int kaizen) {
    /* 旧接口: 换型参数用默认(2/1/6) */
    hv_set_params(rho, kaizen, 2, 1, 6);
    for (int i = 0; i < NTILE; i++) {
        hv.hand0[i] = g_buf[i];
        int v = 4 - g_buf2[i];
        hv.u0[i] = (i8)(v > 0 ? v : 0);
    }
}

__attribute__((export_name("mj_hv_choose_discard")))
int mj_hv_choose_discard(void) { return hv_choose_discard(); }

__attribute__((export_name("mj_hv_decide_peng")))
int mj_hv_decide_peng(int tile) { return hv_decide_peng(tile); }

__attribute__((export_name("mj_hv_decide_gang")))
int mj_hv_decide_gang(int tile, int kind) { return hv_decide_gang(tile, kind); }

/* 返回指定弃牌后的期望巡数(供分析面板与对拍测试用) */
__attribute__((export_name("mj_hv_e_after_discard")))
double mj_hv_e_after_discard(int tile) {
    i8 h[NTILE];
    for (int i = 0; i < NTILE; i++) h[i] = hv.hand0[i];
    if (h[tile] <= 0) return -1.0;
    h[tile]--;
    return E_rec(h, hv.u0, 0);
}

/* 给原生对拍测试用的直传版本(wasm 侧不用) */
int mjc_shanten(const i8 *c) { return shanten_impl(c); }
int mjc_is_win(const i8 *c) { return is_win_impl(c); }
int mjc_useful_mask(const i8 *cin) {
    i8 c[NTILE];
    for (int i = 0; i < NTILE; i++) c[i] = cin[i];
    int base = shanten_impl(c);
    u32 mask = 0;
    for (int t = 0; t < NTILE; t++) {
        if (c[t] >= 4) continue;
        c[t]++;
        int s = shanten_impl(c);
        c[t]--;
        if (s < base) mask |= 1u << t;
    }
    return (int)mask;
}

/* 分析器的原生对拍入口 */
void mjc_hv_set2(const i8 *hand, const i8 *visible, double rho, int kaizen,
                 int kai_margin, int kai_max, int kai_topk) {
    hv_set_params(rho, kaizen, kai_margin, kai_max, kai_topk);
    for (int i = 0; i < NTILE; i++) {
        hv.hand0[i] = hand[i];
        int v = 4 - visible[i];
        hv.u0[i] = (i8)(v > 0 ? v : 0);
    }
}
void mjc_hv_set(const i8 *hand, const i8 *visible, double rho, int kaizen) {
    mjc_hv_set2(hand, visible, rho, kaizen, 2, 1, 6);
}
int mjc_hv_choose_discard(void) { return hv_choose_discard(); }
int mjc_hv_decide_peng(int tile) { return hv_decide_peng(tile); }
int mjc_hv_decide_gang(int tile, int kind) { return hv_decide_gang(tile, kind); }
double mjc_hv_e_after_discard(int tile) { return mj_hv_e_after_discard(tile); }
