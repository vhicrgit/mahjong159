/* 安康159 规则/Bot 的原生实现 —— 用 C 替掉 Python 里最热的 shanten/is_win 递归。
 *
 * 为什么用 C 而不是向量化(numpy/torch/jax):
 *   v10/v31 的出牌评估是"分支 + 记忆化"型负载 —— 单次决策要算 ~2.4 万次 shanten,
 *   但其中大量是重复手牌(记忆化命中率 ~75%)。向量化后台(GPU/SIMD)无法利用记忆化,
 *   必须把所有分支都算满, 反而更慢; 而 C + 查表 + 哈希记忆化可以两头都吃到。
 *
 * 查表: A[code][r_used][m*2+p] = 该单花色组合可达的最大搭子数 t (255=不可达),
 *       W[code][r_used][0/1]   = 能否"全面子" / "1将+全面子"。
 * 由 backend/native/build_tables.py 从已对拍验证的 suit_front_table.npz 生成。
 *
 * 编译: cc -O3 -march=native -fPIC -shared -o libmj159.so mj159.c
 */
#include <fcntl.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

#define NCODE 1953125
#define PAD 255
#define RED 27

static const uint8_t *A_TBL = NULL; /* [code][ru][10] */
static const uint8_t *W_TBL = NULL; /* [code][ru][2]  */

/* ---------------- 记忆化(直接映射, 冲突即覆盖; shanten/is_win 是纯函数, 永不失效) */
#define SH_BITS 21
#define SH_SIZE (1u << SH_BITS)
#define TI_BITS 21
#define TI_SIZE (1u << TI_BITS)

typedef struct {
    uint64_t key;
    int16_t val;
    uint8_t red;
    uint8_t used;
} Ent;

typedef struct {
    uint64_t key;
    uint32_t mask;
    int16_t sh;
    uint8_t red;
    uint8_t used;
} EntTI;

static Ent *sh_memo = NULL;
static Ent *win_memo = NULL;
static EntTI *ti_memo = NULL;

static inline uint64_t mix64(uint64_t x) {
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33;
    x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33;
    return x;
}

static void *map_ro(const char *path, size_t n) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) return NULL;
    void *p = mmap(NULL, n, PROT_READ, MAP_SHARED, fd, 0);
    close(fd);
    return (p == MAP_FAILED) ? NULL : p;
}

int mj_init(const char *front_path, const char *win_path) {
    if (A_TBL) return 0;
    A_TBL = (const uint8_t *)map_ro(front_path, (size_t)NCODE * 5 * 10);
    if (!A_TBL) return -1;
    W_TBL = (const uint8_t *)map_ro(win_path, (size_t)NCODE * 5 * 2);
    if (!W_TBL) return -2;
    sh_memo = (Ent *)calloc(SH_SIZE, sizeof(Ent));
    win_memo = (Ent *)calloc(SH_SIZE, sizeof(Ent));
    ti_memo = (EntTI *)calloc(TI_SIZE, sizeof(EntTI));
    if (!sh_memo || !win_memo || !ti_memo) return -3;
    return 0;
}

/* ---------------- 编码 ---------------- */
typedef struct {
    int c0, c1, c2, red, total;
    uint64_t key;
} Code;

static inline void encode(const int8_t *c, Code *o) {
    int a = 0, b = 0, d = 0, tt = 0;
    for (int i = 8; i >= 0; i--) a = a * 5 + c[i];
    for (int i = 17; i >= 9; i--) b = b * 5 + c[i];
    for (int i = 26; i >= 18; i--) d = d * 5 + c[i];
    for (int i = 0; i < 28; i++) tt += c[i];
    o->c0 = a; o->c1 = b; o->c2 = d; o->red = c[27]; o->total = tt;
    o->key = (uint64_t)a | ((uint64_t)b << 21) | ((uint64_t)d << 42);
}

/* ---------------- 向听 ----------------
 * 适用范围: total <= 14。表把单花色的 m/t 截断在 4, 只有 need<=4 时截断才无损;
 * total>=16 会让 Python 侧 need=5, 本实现(与 jax159/shanten.py 相同)会 clamp 到 4
 * 从而偏小。手牌最多 13+1=14 张, 副露只会更短, 所以 need>4 不可达。 */
static int shanten_core(int c0, int c1, int c2, int red, int total) {
    int need = (total - 1) / 3;
    if (need < 1) need = 1;
    if (need > 4) need = 4;

    int8_t G01[5][10], G012[5][10];
    memset(G01, -1, sizeof(G01));
    for (int r0 = 0; r0 <= red; r0++) {
        const uint8_t *a0 = A_TBL + ((size_t)c0 * 5 + r0) * 10;
        for (int r1 = 0; r0 + r1 <= red; r1++) {
            const uint8_t *a1 = A_TBL + ((size_t)c1 * 5 + r1) * 10;
            int8_t *g = G01[r0 + r1];
            for (int i = 0; i < 10; i++) {
                if (a0[i] == PAD) continue;
                int m0 = i >> 1, p0 = i & 1, t0 = a0[i];
                for (int j = 0; j < 10; j++) {
                    if (a1[j] == PAD) continue;
                    int m = m0 + (j >> 1); if (m > 4) m = 4;
                    int p = p0 + (j & 1); if (p > 1) p = 1;
                    int t = t0 + a1[j]; if (t > 4) t = 4;
                    int idx = m * 2 + p;
                    if (t > g[idx]) g[idx] = (int8_t)t;
                }
            }
        }
    }
    memset(G012, -1, sizeof(G012));
    for (int R = 0; R <= red; R++) {
        const int8_t *g = G01[R];
        for (int i = 0; i < 10; i++) {
            int t0 = g[i];
            if (t0 < 0) continue;
            int m0 = i >> 1, p0 = i & 1;
            for (int r2 = 0; R + r2 <= red; r2++) {
                const uint8_t *a2 = A_TBL + ((size_t)c2 * 5 + r2) * 10;
                int8_t *o = G012[R + r2];
                for (int j = 0; j < 10; j++) {
                    if (a2[j] == PAD) continue;
                    int m = m0 + (j >> 1); if (m > 4) m = 4;
                    int p = p0 + (j & 1); if (p > 1) p = 1;
                    int t = t0 + a2[j]; if (t > 4) t = 4;
                    int idx = m * 2 + p;
                    if (t > o[idx]) o[idx] = (int8_t)t;
                }
            }
        }
    }

    int best = 99;
    for (int R = 0; R <= red; R++) {
        int left = red - R, q = left / 3, rem = left % 3;
        const int8_t *g = G012[R];
        for (int i = 0; i < 10; i++) {
            int t = g[i];
            if (t < 0) continue;
            int m = (i >> 1) + q, p = i & 1;
            int mm = m < need ? m : need;
            int room = need - mm;
            if (rem == 2) {
                int tA = t + 1; if (tA > room) tA = room;
                int s = 2 * need - 2 * mm - tA - p;
                if (s < best) best = s;
                int tB = t < room ? t : room;
                int pB = p + 1; if (pB > 1) pB = 1;
                s = 2 * need - 2 * mm - tB - pB;
                if (s < best) best = s;
            } else {
                int t2 = t < room ? t : room;
                int s = 2 * need - 2 * mm - t2 - p;
                if (s < best) best = s;
            }
        }
    }
    return best;
}

static inline int shanten_code(const Code *k) {
    uint32_t h = (uint32_t)(mix64(k->key ^ (0x9E3779B97F4A7C15ULL * (uint64_t)k->red))
                            & (SH_SIZE - 1));
    Ent *e = &sh_memo[h];
    if (e->used && e->key == k->key && e->red == (uint8_t)k->red) return e->val;
    int v = shanten_core(k->c0, k->c1, k->c2, k->red, k->total);
    e->used = 1; e->key = k->key; e->red = (uint8_t)k->red; e->val = (int16_t)v;
    return v;
}

int mj_shanten(const int8_t *c) {
    Code k;
    encode(c, &k);
    /* need==0 特判(四副露只差将), 与 rules/win.py shanten() 逐位一致。
       C 的整数除法向零取整, total=0 时 (0-1)/3=0 也进特判, 无害。 */
    int need = (k.total - 1) / 3;
    if (need <= 0) {
        if (k.total <= 1) return 0;
        if (k.red >= 1) return -1;
        for (int i = 0; i < 27; i++)
            if (c[i] >= 2) return -1;
        return 1;
    }
    return shanten_code(&k);
}

/* 与 mj_shanten 同口径, 但供只有原始计数的内部调用链使用(tiles_info/ukeire)。
   少了这层特判, 四副露的单钓会被 shanten_core 的 need clamp 当成搭子:
   暗牌 1 张时算出向听 2, 于是"补成对"和"补成两面/嵌张"都被记作进张
   (实测 400 手四副露里 97.8% 的进张集与 Python 参考不一致)。 */
static inline int shanten_raw(const int8_t *c) {
    Code k;
    encode(c, &k);
    if ((k.total - 1) / 3 > 0) return shanten_code(&k);
    if (k.total <= 1) return 0;
    if (k.red >= 1) return -1;
    for (int i = 0; i < 27; i++)
        if (c[i] >= 2) return -1;
    return 1;
}

/* ---------------- 判胡 ---------------- */
static int is_win_core(int c0, int c1, int c2, int red, int total) {
    if (total % 3 != 2) return 0;
    const uint8_t *w0 = W_TBL + (size_t)c0 * 10;
    const uint8_t *w1 = W_TBL + (size_t)c1 * 10;
    const uint8_t *w2 = W_TBL + (size_t)c2 * 10;
    for (int r0 = 0; r0 <= red; r0++) {
        int A0 = w0[r0 * 2], B0 = w0[r0 * 2 + 1];
        if (!A0 && !B0) continue;
        for (int r1 = 0; r0 + r1 <= red; r1++) {
            int A1 = w1[r1 * 2], B1 = w1[r1 * 2 + 1];
            if (!A1 && !B1) continue;
            for (int r2 = 0; r0 + r1 + r2 <= red; r2++) {
                int A2 = w2[r2 * 2], B2 = w2[r2 * 2 + 1];
                int left = red - r0 - r1 - r2;
                if (left % 3 == 0) {
                    if ((B0 && A1 && A2) || (A0 && B1 && A2) || (A0 && A1 && B2))
                        return 1;
                }
                if (left == 2 && A0 && A1 && A2) return 1;
            }
        }
    }
    return 0;
}

static inline int is_win_code(const Code *k) {
    uint32_t h = (uint32_t)(mix64(k->key ^ (0xD6E8FEB86659FD93ULL * (uint64_t)(k->red + 1)))
                            & (SH_SIZE - 1));
    Ent *e = &win_memo[h];
    if (e->used && e->key == k->key && e->red == (uint8_t)k->red) return e->val;
    int v = is_win_core(k->c0, k->c1, k->c2, k->red, k->total);
    e->used = 1; e->key = k->key; e->red = (uint8_t)k->red; e->val = (int16_t)v;
    return v;
}

int mj_is_win(const int8_t *c) {
    Code k;
    encode(c, &k);
    return is_win_code(&k);
}

/* ---------------- 有效张(听口 或 降向听进张) ---------------- */
/* 与 rules.ting.waiting_tiles / useful_draws 同口径: s==0 时取听口, 否则取降向听进张 */
static void tiles_info(const int8_t *c, int *out_sh, uint32_t *out_mask) {
    Code k;
    encode(c, &k);
    uint32_t h = (uint32_t)(mix64(k.key ^ (0xA24BAED4963EE407ULL * (uint64_t)(k.red + 7)))
                            & (TI_SIZE - 1));
    EntTI *e = &ti_memo[h];
    if (e->used && e->key == k.key && e->red == (uint8_t)k.red) {
        *out_sh = e->sh;
        *out_mask = e->mask;
        return;
    }
    int s = shanten_raw(c);
    uint32_t mask = 0;
    int8_t g[28];
    memcpy(g, c, 28);
    for (int t = 0; t < 28; t++) {
        if (c[t] >= 4) continue;
        g[t]++;
        Code k2;
        encode(g, &k2);
        if (s == 0) {
            if (is_win_code(&k2)) mask |= 1u << t;
        } else {
            if (shanten_raw(g) < s) mask |= 1u << t;
        }
        g[t]--;
    }
    e->used = 1; e->key = k.key; e->red = (uint8_t)k.red;
    e->sh = (int16_t)s; e->mask = mask;
    *out_sh = s;
    *out_mask = mask;
}

static inline int ukeire(const int8_t *c, const int8_t *unseen) {
    int s;
    uint32_t mask;
    tiles_info(c, &s, &mask);
    int u = 0;
    while (mask) {
        int t = __builtin_ctz(mask);
        mask &= mask - 1;
        u += unseen[t];
    }
    return u;
}

/* ---------------- 两步推演(v10 _second_step_value / v31 _second_step_m) ---------------- */
static double second_step(const int8_t *c13, const int8_t *unseen) {
    int total = 0;
    for (int i = 0; i < 28; i++) total += unseen[i];
    if (total <= 0) return 0.0;
    int base_s = mj_shanten(c13);
    double v = 0.0;
    int8_t h14[28], h13[28];
    memcpy(h14, c13, 28);
    for (int draw = 0; draw < 28; draw++) {
        int n = unseen[draw];
        if (n <= 0) continue;
        h14[draw]++;
        if (base_s == 0 && mj_is_win(h14)) {
            v += (double)n / total * 50.0;
            h14[draw]--;
            continue;
        }
        int best_s = 99, best_u = 0;
        for (int d = 0; d < 28; d++) {
            if (h14[d] <= 0) continue;
            memcpy(h13, h14, 28);
            h13[d]--;
            int s = mj_shanten(h13);
            int u = ukeire(h13, unseen);
            if (s < best_s || (s == best_s && u > best_u)) {
                best_s = s;
                best_u = u;
            }
        }
        int drop = base_s - best_s;
        if (drop < 0) drop = 0;
        v += (double)n / total * (20.0 * drop + 0.15 * best_u);
        h14[draw]--;
    }
    return v;
}

static inline double risk_of(int remain) {
    if (remain == 3) return 0.4;
    if (remain == 2) return 0.2;
    if (remain == 1) return 0.05;
    if (remain == 0) return 0.0;
    return 0.4; /* dict.get(默认) */
}

/* ---------------- v10 / v31 出牌 (两者公式等价, 见 native.py 说明) ---------------- */
/* 逐候选打分: out[k] = {tile, shanten, ukeire, cont, score}。
   chooser 与 trace 共用, 保证记录下来的分数就是 bot 实际用的分数。 */
static int score_discards_v10(const int8_t *hand, const int8_t *unseen,
                              const uint8_t *penged, double eg,
                              double SW, double UW, double CW, double RW,
                              int cont_max, double *out) {
    int nc = 0, min_sh = 99;
    int8_t h[28];
    for (int t = 0; t < 28; t++) {
        if (hand[t] <= 0) continue;
        memcpy(h, hand, 28);
        h[t]--;
        int s = mj_shanten(h);
        double *o = out + nc * 5;
        o[0] = t;
        o[1] = s;
        o[2] = 0;
        o[3] = 0;
        o[4] = 0;
        nc++;
        if (s < min_sh) min_sh = s;
    }
    for (int k = 0; k < nc; k++) {
        double *o = out + k * 5;
        int t = (int)o[0], s = (int)o[1];
        double score;
        if (s > min_sh) {
            score = -10.0 * SW - SW * s;
        } else {
            memcpy(h, hand, 28);
            h[t]--;
            int u = ukeire(h, unseen);
            double cont = (s <= cont_max) ? second_step(h, unseen) : 0.0;
            o[2] = u;
            o[3] = cont;
            score = UW * u + CW * cont;
        }
        if (t != RED) {
            double risk = penged[t] ? 1.0 : risk_of(unseen[t]);
            score -= RW * (1.0 + 1.5 * eg) * risk;
        }
        o[4] = score;
    }
    return nc;
}

int mj_score_discards_v10(const int8_t *hand, const int8_t *unseen,
                          const uint8_t *penged, double eg,
                          double SW, double UW, double CW, double RW,
                          int cont_max, double *out) {
    return score_discards_v10(hand, unseen, penged, eg,
                              SW, UW, CW, RW, cont_max, out);
}

int mj_choose_discard_v10(const int8_t *hand, const int8_t *unseen,
                          const uint8_t *penged, double eg,
                          double SW, double UW, double CW, double RW,
                          int cont_max) {
    double sc[28 * 5];
    int nc = score_discards_v10(hand, unseen, penged, eg,
                                SW, UW, CW, RW, cont_max, sc);
    if (nc == 0) return -1;
    int best_t = -1;
    double best_score = -1e18;
    for (int k = 0; k < nc; k++) {
        double score = sc[k * 5 + 4];
        if (score > best_score) {
            best_score = score;
            best_t = (int)sc[k * 5];
        }
    }
    return best_t;
}

/* ---------------- v1 出牌 ---------------- */
int mj_choose_discard_v1(const int8_t *hand, const int8_t *visible,
                         const uint8_t *penged) {
    int best_t = -1;
    double best_score = -1e9;
    int8_t h[28], g[28];
    for (int t = 0; t < 28; t++) {
        if (hand[t] <= 0) continue;
        memcpy(h, hand, 28);
        h[t]--;
        int s = mj_shanten(h);
        int wr = 0;
        if (s == 0) { /* waits 只在听牌时非空 */
            memcpy(g, h, 28);
            for (int w = 0; w < 28; w++) {
                if (h[w] >= 4) continue;
                g[w]++;
                if (mj_is_win(g)) {
                    int x = 4 - visible[w] - hand[w];
                    if (x < 0) x = 0;
                    wr += x;
                }
                g[w]--;
            }
        }
        double risk = 0.0;
        if (t != RED) {
            if (penged[t]) {
                risk = 1.0;
            } else {
                int remain = 4 - visible[t] - hand[t];
                if (remain < 0) remain = 0;
                risk = risk_of(remain);
            }
        }
        double score = -100.0 * s + 3.0 * wr - 25.0 * risk;
        if (score > best_score) {
            best_score = score;
            best_t = t;
        }
    }
    return best_t;
}

/* ---------------- Oracle beam search (bot_oracle.search_first_discard_detail)
 * cheat_full 96% 的时间花在这里。必须与 Python 逐位一致, 所以要复刻:
 *   - next_nodes 的构造顺序(beam_nodes 顺序 × tile 升序)
 *   - 每个 depth 内按 (hand13, first_t) 去重, 保留首次出现
 *   - 按 first_t 分组, 组的顺序 = first_t 在 next_nodes 里的首次出现顺序
 *   - 组内按 shanten 稳定排序后截断到 beam
 * 排序若不稳定, 同向听的不同手牌会被换掉, 后续 depth 就会走偏。
 */
#define BS_MAX_BEAM 32
#define BS_MAX_CUR (28 * BS_MAX_BEAM)
#define BS_MAX_NEXT (BS_MAX_CUR * 28)
#define BS_SEEN_BITS 18
#define BS_SEEN_SIZE (1u << BS_SEEN_BITS)

typedef struct {
    int8_t h[28];
    uint8_t ft;
    int16_t sh;
} BNode;

typedef struct {
    uint64_t key;
    uint32_t gen;
    uint8_t red;
    uint8_t ft;
} SeenEnt;

static BNode *bs_cur = NULL, *bs_next = NULL;
static int *bs_order = NULL;
static SeenEnt *bs_seen = NULL;
static uint32_t bs_gen = 0;

static int bs_alloc(void) {
    if (bs_cur) return 0;
    bs_cur = (BNode *)malloc(sizeof(BNode) * BS_MAX_CUR);
    bs_next = (BNode *)malloc(sizeof(BNode) * BS_MAX_NEXT);
    bs_order = (int *)malloc(sizeof(int) * BS_MAX_NEXT * 2);
    bs_seen = (SeenEnt *)calloc(BS_SEEN_SIZE, sizeof(SeenEnt));
    return (bs_cur && bs_next && bs_order && bs_seen) ? 0 : -1;
}

static inline int bs_seen_insert(uint64_t key, int red, int ft) {
    uint64_t hh = mix64(key ^ (0x2545F4914F6CDD1DULL *
                               (uint64_t)(red * 32 + ft + 1)));
    uint32_t i = (uint32_t)(hh & (BS_SEEN_SIZE - 1));
    for (int probe = 0; probe < 256; probe++) {
        SeenEnt *e = &bs_seen[i];
        if (e->gen != bs_gen) {
            e->gen = bs_gen;
            e->key = key;
            e->red = (uint8_t)red;
            e->ft = (uint8_t)ft;
            return 1;
        }
        if (e->key == key && e->red == (uint8_t)red && e->ft == (uint8_t)ft)
            return 0;
        i = (i + 1) & (BS_SEEN_SIZE - 1);
    }
    return 1;
}

/* out_wd[t]: 胡牌所需摸牌数; -1 = Python 的 None; -2 = 该 tile 不是合法首出牌。
   out_sh[t]: 视野内可达的最小向听(胡了记 -1)。 */
int mj_beam_detail(const int8_t *counts14, const int8_t *future, int horizon,
                   int beam, int32_t *out_wd, int32_t *out_sh) {
    if (bs_alloc() != 0) return -1;
    if (beam > BS_MAX_BEAM) beam = BS_MAX_BEAM;
    if (beam < 1) beam = 1;

    uint8_t won[28];
    memset(won, 0, sizeof(won));
    for (int t = 0; t < 28; t++) {
        out_wd[t] = -2;
        out_sh[t] = 99;
    }

    int ncur = 0;
    int8_t c[28];
    for (int t = 0; t < 28; t++) {
        if (counts14[t] <= 0) continue;
        memcpy(c, counts14, 28);
        c[t]--;
        int s = mj_shanten(c);
        BNode *n = &bs_cur[ncur++];
        memcpy(n->h, c, 28);
        n->ft = (uint8_t)t;
        n->sh = (int16_t)s;
        out_wd[t] = -1;
        out_sh[t] = s;
    }
    if (ncur == 0) return 0;

    int *ord = bs_order;
    int *tmp = bs_order + BS_MAX_NEXT;

    for (int depth = 0; depth < horizon; depth++) {
        int draw = future[depth];
        int nnext = 0;
        bs_gen++;
        for (int k = 0; k < ncur; k++) {
            int ft = bs_cur[k].ft;
            if (won[ft]) continue;
            int8_t h14[28];
            memcpy(h14, bs_cur[k].h, 28);
            h14[draw]++;
            if (mj_is_win(h14)) {
                won[ft] = 1;
                out_wd[ft] = depth;
                out_sh[ft] = -1;
                continue;
            }
            for (int t = 0; t < 28; t++) {
                if (h14[t] <= 0) continue;
                h14[t]--;
                Code kk;
                encode(h14, &kk);
                if (bs_seen_insert(kk.key, kk.red, ft) && nnext < BS_MAX_NEXT) {
                    BNode *m = &bs_next[nnext++];
                    memcpy(m->h, h14, 28);
                    m->ft = (uint8_t)ft;
                    m->sh = (int16_t)shanten_code(&kk);
                }
                h14[t]++;
            }
        }
        if (nnext == 0) break;

        /* first_t 的首次出现顺序 */
        int ft_pos[28], ft_order[28], nft = 0;
        for (int t = 0; t < 28; t++) ft_pos[t] = -1;
        for (int i = 0; i < nnext; i++) {
            int ft = bs_next[i].ft;
            if (ft_pos[ft] < 0) {
                ft_pos[ft] = nft;
                ft_order[nft++] = ft;
            }
        }
        /* 计数排序按组归拢, 组内保持插入序 */
        int cnt[29] = {0}, start[29], fill[29];
        for (int i = 0; i < nnext; i++) cnt[ft_pos[bs_next[i].ft]]++;
        start[0] = 0;
        for (int g = 1; g <= nft; g++) start[g] = start[g - 1] + cnt[g - 1];
        for (int g = 0; g <= nft; g++) fill[g] = start[g];
        for (int i = 0; i < nnext; i++)
            ord[fill[ft_pos[bs_next[i].ft]]++] = i;

        ncur = 0;
        for (int g = 0; g < nft; g++) {
            int ft = ft_order[g];
            int base = start[g], m = cnt[g];
            /* 组内按 shanten 稳定排序: 向听范围小, 计数排序即稳定 */
            int sc[18] = {0}, ss[19];
            for (int j = 0; j < m; j++) {
                int s = bs_next[ord[base + j]].sh;
                if (s < 0) s = 0;
                if (s > 17) s = 17;
                sc[s]++;
            }
            ss[0] = 0;
            for (int s = 1; s <= 18; s++) ss[s] = ss[s - 1] + sc[s - 1];
            for (int j = 0; j < m; j++) {
                int idx = ord[base + j];
                int s = bs_next[idx].sh;
                if (s < 0) s = 0;
                if (s > 17) s = 17;
                tmp[ss[s]++] = idx;
            }
            int keep = m < beam ? m : beam;
            for (int j = 0; j < keep && ncur < BS_MAX_CUR; j++)
                bs_cur[ncur++] = bs_next[tmp[j]];
            if (!won[ft]) {
                int best_s = bs_next[tmp[0]].sh;
                if (out_wd[ft] == -1 && best_s < out_sh[ft])
                    out_sh[ft] = best_s;
            }
        }
        if (ncur == 0) break;
    }
    return 0;
}

/* ---------------- 碰/杠 ---------------- */
/* bot: 1=v1, 10=v10, 31=v31 */
int mj_decide_peng(int bot, const int8_t *hand, int tile) {
    int before = mj_shanten(hand);
    int8_t c[28];
    memcpy(c, hand, 28);
    c[tile] -= 2;
    int after;
    if (bot == 10) {
        after = mj_shanten(c);
        if (after < before) return (before != 0) || (after == 0);
        return 0;
    }
    after = 99;
    int8_t g[28];
    for (int d = 0; d < 28; d++) {
        if (c[d] <= 0) continue;
        memcpy(g, c, 28);
        g[d]--;
        int s = mj_shanten(g);
        if (s < after) after = s;
    }
    if (bot == 1) return after < before;
    if (after < before) return (before != 0) || (after == 0);
    return 0;
}

/* kind: 0=ming, 1=an, 2=bu —— v1/v10/v31 三者公式一致 */
int mj_decide_gang(int bot, const int8_t *hand, int tile, int kind) {
    (void)bot;
    int before = mj_shanten(hand);
    int8_t c[28];
    memcpy(c, hand, 28);
    if (kind == 0) c[tile] -= 3;
    else if (kind == 1) c[tile] -= 4;
    else c[tile] -= 1;
    int after = mj_shanten(c);
    return !(before == 0 && after > 0);
}

/* ---------------- cheat_full 用的小工具 ---------------- */
/* rules.ting.discard_options 里 (tile, shanten) 那部分。
   _ranked_oracle_discards 只读 tile/shanten, waits 白算(每次 ~280 次判胡)。*/
int mj_discard_shanten(const int8_t *hand14, int32_t *out_tiles,
                       int32_t *out_sh) {
    int n = 0;
    int8_t h[28];
    for (int t = 0; t < 28; t++) {
        if (hand14[t] <= 0) continue;
        memcpy(h, hand14, 28);
        h[t]--;
        out_tiles[n] = t;
        out_sh[n] = mj_shanten(h);
        n++;
    }
    return n;
}

/* BotCheat._heuristic_score 的 ukeire: sum(unseen[w] for w in waiting_tiles(hand13)) */
int mj_waits_ukeire(const int8_t *hand13, const int8_t *unseen) {
    int u = 0;
    int8_t g[28];
    memcpy(g, hand13, 28);
    for (int w = 0; w < 28; w++) {
        if (hand13[w] >= 4) continue;
        g[w]++;
        if (mj_is_win(g)) u += unseen[w];
        g[w]--;
    }
    return u;
}

/* ---------------- 批量对拍入口 ---------------- */
void mj_shanten_batch(const int8_t *hands, int n, int32_t *out) {
    for (int i = 0; i < n; i++) out[i] = mj_shanten(hands + (size_t)i * 28);
}

void mj_is_win_batch(const int8_t *hands, int n, int32_t *out) {
    for (int i = 0; i < n; i++) out[i] = mj_is_win(hands + (size_t)i * 28);
}

/* ================= 牌型价值 E 引擎 =================
 * 引擎本体与 mobile/wasm/mjcore.c 共享同一份 hv_engine_inc.c,
 * 本宿主注入查表版原语(mj_shanten/mj_is_win 走 93MB LUT, 比 DFS 快 5-10 倍)。
 * 语义与 backend/analysis/hand_value.py 逐位一致;
 * 改动后跑 tools/perf/test_hv_c_parity.py 对拍。 */
typedef int8_t i8;
typedef uint8_t u8;
typedef uint32_t u32;
typedef uint64_t u64;
#define NTILE 28

static inline u64 key27(const i8 *c) {
    int a = 0, b = 0, d = 0;
    for (int i = 8; i >= 0; i--) a = a * 5 + c[i];
    for (int i = 17; i >= 9; i--) b = b * 5 + c[i];
    for (int i = 26; i >= 18; i--) d = d * 5 + c[i];
    return (u64)a | ((u64)b << 21) | ((u64)d << 42);
}

#define HV_SHANTEN(c) mj_shanten(c)
#define HV_IS_WIN(c)  mj_is_win(c)
#define HV_KEY27(c)   key27(c)
#include "../../mobile/wasm/hv_engine_inc.c"

int mj_hv_set2(const int8_t *hand, const int8_t *visible, double rho,
               int kaizen, int kai_margin, int kai_max, int kai_topk) {
    hve_set_ctx(hand, visible, rho, kaizen, kai_margin, kai_max, kai_topk);
    return 0;
}
double mj_hv_e_after_discard(int tile) { return hve_e_after_discard(tile); }
int mj_hv_choose_discard(void) { return hve_choose_discard(); }
int mj_hv_decide_peng(int tile) { return hve_decide_peng(tile); }
int mj_hv_decide_gang(int tile, int kind) { return hve_decide_gang(tile, kind); }
int mj_hv_explain_buf(int tile, double *outf, int *outi) {
    return hve_explain(tile, outf, outi);
}
void mj_hv_stats(uint64_t *out) { hve_stats(out); }
void mj_hv_stats_reset(void) { hve_stats_reset(); }
