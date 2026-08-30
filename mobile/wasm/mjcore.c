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

/* ---------------- 分花色前沿(懒计算 + 直映射缓存 + 计算日志) ----------------
 * 整手向听 = 三花色前沿按红中分配合并(与 mj159.c 的 LUT 合并逐位同构)。
 * 花色前沿对 (code, red_used) 懒计算: 用 dfs_front 跑单花色(9 张填 0..8 位),
 * 压成 (m,p)->max_t 十格。缓存条目降到花色级(实战工作集几万条), 命中率
 * 远高于整手级缓存。计算日志供 JS 侧落盘(IndexedDB), 下次启动回放暖机。 */
#define SF_BITS 18
#define SF_SIZE (1u << SF_BITS)
typedef struct { u64 key; i8 tab[10]; u8 used; } SfEnt;
static SfEnt sf_memo[SF_SIZE];

#define SF_LOG_MAX (1u << 18)
#define SF_REC 15                     /* code u32 + red u8 + tab 10B */
static u8 sf_log[SF_LOG_MAX * SF_REC];
static u32 sf_log_n = 0;
static int sf_loading = 0;            /* 回放期间不写日志(否则每轮启停翻倍) */

static void suit_front_compute(int code, int red, i8 tab[10]) {
    i8 c[NTILE];
    for (int i = 0; i < NTILE; i++) c[i] = 0;
    int x = code;
    for (int i = 0; i < 9; i++) { c[i] = (i8)(x % 5); x /= 5; }
    Front f;
    dfs_front(c, red, 4, &f);
    for (int i = 0; i < 10; i++) tab[i] = -1;
    for (int b = 0; b < 4; b++) {
        u64 w = f.w[b];
        while (w) {
            int bit = __builtin_ctzll(w);
            w &= w - 1;
            int idx = (b << 6) | bit;
            int m = (idx >> 5) & 7, t = (idx >> 2) & 7, p2 = idx & 3;
            if (m > 4) m = 4;
            if (t > 4) t = 4;
            if (p2 > 1) p2 = 1;
            int ci = m * 2 + p2;
            if (t > tab[ci]) tab[ci] = (i8)t;
        }
    }
}

static void sf_store(u64 key, const i8 tab[10]) {
    u32 h = (u32)(mix64(key) & (SF_SIZE - 1));
    SfEnt *e = &sf_memo[h];
    e->used = 1; e->key = key;
    for (int i = 0; i < 10; i++) e->tab[i] = tab[i];
}

static void suit_front(int code, int red, i8 tab[10]) {
    u64 key = (u64)(u32)code | ((u64)(u8)red << 21);
    u32 h = (u32)(mix64(key) & (SF_SIZE - 1));
    SfEnt *e = &sf_memo[h];
    if (e->used && e->key == key) {
        for (int i = 0; i < 10; i++) tab[i] = e->tab[i];
        return;
    }
    suit_front_compute(code, red, tab);
    e->used = 1; e->key = key;
    for (int i = 0; i < 10; i++) e->tab[i] = tab[i];
    if (!sf_loading && sf_log_n < SF_LOG_MAX) {
        u8 *r = sf_log + (u64)sf_log_n * SF_REC;
        r[0] = (u8)(code & 0xFF); r[1] = (u8)((code >> 8) & 0xFF);
        r[2] = (u8)((code >> 16) & 0xFF); r[3] = (u8)((code >> 24) & 0xFF);
        r[4] = (u8)red;
        for (int i = 0; i < 10; i++) r[5 + i] = (u8)tab[i];
        sf_log_n++;
    }
}

/* ---- 落盘接口(wasm 用指针版; 原生对拍用直传版) ---- */
static u8 g_sf_in[SF_LOG_MAX * SF_REC];   /* JS 写入的回放暂存区 */

__attribute__((export_name("mj_sf_log_ptr")))
int mj_sf_log_ptr(void) { return (int)(long)sf_log; }
__attribute__((export_name("mj_sf_log_len")))
int mj_sf_log_len(void) { return (int)sf_log_n; }
__attribute__((export_name("mj_sf_in_ptr")))
int mj_sf_in_ptr(void) { return (int)(long)g_sf_in; }
__attribute__((export_name("mj_sf_in_cap")))
int mj_sf_in_cap(void) { return (int)(SF_LOG_MAX * SF_REC); }

static void sf_load_records(const u8 *buf, u32 n) {
    sf_loading = 1;
    u32 room = n > SF_LOG_MAX ? SF_LOG_MAX : n;
    for (u32 i = 0; i < room; i++) {
        const u8 *r = buf + (u64)i * SF_REC;
        int code = (int)r[0] | ((int)r[1] << 8) | ((int)r[2] << 16)
                 | ((int)r[3] << 24);
        int red = (int)r[4];
        i8 tab[10];
        for (int j = 0; j < 10; j++) tab[j] = (i8)r[5 + j];
        sf_store((u64)(u32)code | ((u64)(u8)red << 21), tab);
        if (sf_log_n < SF_LOG_MAX) {
            for (int j = 0; j < SF_REC; j++)
                sf_log[(u64)sf_log_n * SF_REC + j] = r[j];
            sf_log_n++;
        }
    }
    sf_loading = 0;
}

__attribute__((export_name("mj_sf_load")))
void mj_sf_load(int n) { sf_load_records(g_sf_in, (u32)n); }

/* 原生直传版 */
void mjc_sf_load(const u8 *buf, int n) { sf_load_records(buf, (u32)n); }
int mjc_sf_log_len(void) { return (int)sf_log_n; }
const u8 *mjc_sf_log_ptr(void) { return sf_log; }

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

    if (need > 4) {
        /* 超过 14 张的非法手牌: 分花色表截断在 4, 退回整手 DFS(保持语义) */
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

    /* 分花色: 每花色各红中档前沿(懒计算+缓存) */
    int code[3] = {0, 0, 0};
    for (int s3 = 0; s3 < 3; s3++)
        for (int i = 8; i >= 0; i--) code[s3] = code[s3] * 5 + c[s3 * 9 + i];
    i8 F[3][5][10];
    for (int s3 = 0; s3 < 3; s3++)
        for (int r = 0; r <= red; r++) suit_front(code[s3], r, F[s3][r]);

    /* 阶段合并(与 mj159.c shanten_core 逐位同构): 先合 0+1, 再合 2 */
    i8 G01[5][10], G012[5][10];
    for (int i = 0; i < 50; i++) { G01[0][i] = -1; G012[0][i] = -1; }
    for (int i = 1; i < 5; i++)
        for (int j = 0; j < 10; j++) { G01[i][j] = -1; G012[i][j] = -1; }
    for (int r0 = 0; r0 <= red; r0++) {
        for (int r1 = 0; r0 + r1 <= red; r1++) {
            i8 *g = G01[r0 + r1];
            for (int i = 0; i < 10; i++) {
                int t0 = F[0][r0][i];
                if (t0 < 0) continue;
                int m0 = i >> 1, p0 = i & 1;
                for (int j = 0; j < 10; j++) {
                    int t1 = F[1][r1][j];
                    if (t1 < 0) continue;
                    int m = m0 + (j >> 1); if (m > 4) m = 4;
                    int p2 = p0 + (j & 1); if (p2 > 1) p2 = 1;
                    int t = t0 + t1; if (t > 4) t = 4;
                    int idx = m * 2 + p2;
                    if (t > g[idx]) g[idx] = (i8)t;
                }
            }
        }
    }
    for (int R = 0; R <= red; R++) {
        i8 *g = G01[R];
        for (int i = 0; i < 10; i++) {
            int t0 = g[i];
            if (t0 < 0) continue;
            int m0 = i >> 1, p0 = i & 1;
            for (int r2 = 0; R + r2 <= red; r2++) {
                i8 *o = G012[R + r2];
                for (int j = 0; j < 10; j++) {
                    int t1 = F[2][r2][j];
                    if (t1 < 0) continue;
                    int m = m0 + (j >> 1); if (m > 4) m = 4;
                    int p2 = p0 + (j & 1); if (p2 > 1) p2 = 1;
                    int t = t0 + t1; if (t > 4) t = 4;
                    int idx = m * 2 + p2;
                    if (t > o[idx]) o[idx] = (i8)t;
                }
            }
        }
    }
    int best = 99;
    for (int R = 0; R <= red; R++) {
        int left = red - R, q = left / 3, rem = left % 3;
        i8 *g = G012[R];
        for (int i = 0; i < 10; i++) {
            int t = g[i];
            if (t < 0) continue;
            int m = (i >> 1) + q, p2 = i & 1;
            int mm = m < need ? m : need;
            int room = need - mm;
            if (rem == 2) {
                int tA = t + 1; if (tA > room) tA = room;
                int v = 2 * need - 2 * mm - tA - p2;
                if (v < best) best = v;
                int tB = t < room ? t : room;
                int pB = p2 + 1; if (pB > 1) pB = 1;
                v = 2 * need - 2 * mm - tB - pB;
                if (v < best) best = v;
            } else {
                int t2 = t < room ? t : room;
                int v = 2 * need - 2 * mm - t2 - p2;
                if (v < best) best = v;
            }
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

/* ================= 牌型价值 E 引擎(共享实现) =================
 * 引擎本体在 hv_engine_inc.c, 与桌面版 backend/native/mj159.c 共用同一份;
 * 本宿主注入自包含 DFS 版原语。语义与 Python/JS 逐位一致。 */
#define HV_SHANTEN(c) shanten_impl(c)
#define HV_IS_WIN(c)  is_win_impl(c)
#define HV_KEY27(c)   enc27(c)
#include "hv_engine_inc.c"

/* wasm 解释输出缓冲(JS 经 mj_outf_ptr/mj_outi_ptr 读) */
static double g_outf[8];
static int g_outi[192];

__attribute__((export_name("mj_outf_ptr")))
int mj_outf_ptr(void) { return (int)(long)g_outf; }
__attribute__((export_name("mj_outi_ptr")))
int mj_outi_ptr(void) { return (int)(long)g_outi; }

__attribute__((export_name("mj_hv_explain")))
int mj_hv_explain(int tile) { return hve_explain(tile, g_outf, g_outi); }

/* 原生对拍入口: 指针返回值在原生 64 位会被 int 截断, 改为调用方传缓冲 */
int mjc_hv_explain_buf(int tile, double *outf, int *outi) {
    return hve_explain(tile, outf, outi);
}
void mjc_hv_stats(u64 *out) { hve_stats(out); }
void mjc_hv_stats_reset(void) { hve_stats_reset(); }

/* ---- 导出: g_buf=手牌(28), g_buf2=可见牌(28) ---- */
static i8 g_buf2[NTILE];

__attribute__((export_name("mj_buf2_ptr")))
int mj_buf2_ptr(void) { return (int)(long)g_buf2; }

__attribute__((export_name("mj_hv_set2")))
void mj_hv_set2(double rho, int kaizen, int kai_margin, int kai_max,
                int kai_topk) {
    hve_set_ctx(g_buf, g_buf2, rho, kaizen, kai_margin, kai_max, kai_topk);
}

__attribute__((export_name("mj_hv_set")))
void mj_hv_set(double rho, int kaizen) {
    /* 旧接口: 换型参数用默认(2/1/6) */
    hve_set_ctx(g_buf, g_buf2, rho, kaizen, 2, 1, 6);
}

__attribute__((export_name("mj_hv_choose_discard")))
int mj_hv_choose_discard(void) { return hve_choose_discard(); }

__attribute__((export_name("mj_hv_decide_peng")))
int mj_hv_decide_peng(int tile) { return hve_decide_peng(tile); }

__attribute__((export_name("mj_hv_decide_gang")))
int mj_hv_decide_gang(int tile, int kind) { return hve_decide_gang(tile, kind); }

/* 返回指定弃牌后的期望巡数(供分析面板与对拍测试用) */
__attribute__((export_name("mj_hv_e_after_discard")))
double mj_hv_e_after_discard(int tile) { return hve_e_after_discard(tile); }

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
    hve_set_ctx(hand, visible, rho, kaizen, kai_margin, kai_max, kai_topk);
}
void mjc_hv_set(const i8 *hand, const i8 *visible, double rho, int kaizen) {
    mjc_hv_set2(hand, visible, rho, kaizen, 2, 1, 6);
}
int mjc_hv_choose_discard(void) { return hve_choose_discard(); }
int mjc_hv_decide_peng(int tile) { return hve_decide_peng(tile); }
int mjc_hv_decide_gang(int tile, int kind) { return hve_decide_gang(tile, kind); }
double mjc_hv_e_after_discard(int tile) { return hve_e_after_discard(tile); }
