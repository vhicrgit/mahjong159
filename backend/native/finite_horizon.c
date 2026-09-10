/* Finite-horizon, without-replacement solitary Mahjong Bellman solver.
 * No Mahjong implementation is duplicated here: the caller supplies the
 * native is_win and shanten functions. Counts include red as tile 27.
 *
 * Chance: draw every available tile and consume it, including useless draws.
 * Decision: maximize over EVERY legal discard after observing that draw.
 * discount < 1 values completion at draw T as discount^(T-1), rather than
 * treating all completions within the horizon equally. It is a constant
 * survival-discount model, not the actual multiplayer win probability.
 * Only exact values enter the transposition table. A node budget returns
 * rigorous [lower, upper] bounds, never an estimated value marked exact.
 */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define NT 28

/* Bump whenever exported function signatures or result-buffer layouts change. */
int fh_abi_version(void) { return 2; }

typedef int (*TileFn)(const int8_t *);
typedef struct { double lo, hi; } Bound;
typedef struct {
    uint64_t hash;
    double value;
    int8_t hand[NT], unseen[NT];
    uint8_t k, phase, used;
} Entry;
typedef struct {
    TileFn win, shanten;
    Entry *memo;
    size_t mask;
    uint64_t nodes, hits, cutoffs, budget;
    double discount; /* Fixed for this context, so not needed in memo keys. */
} Context;

static Bound exact(double x) { Bound b = {x, x}; return b; }

/* Fully compare keys after hashing: hash collisions affect time, not values. */
static Entry *lookup(Context *c, const int8_t *h, const int8_t *u,
                     int k, int phase, uint64_t *hash) {
    uint64_t x = UINT64_C(1469598103934665603);
    for (int i = 0; i < NT; ++i) {
        x = (x ^ (uint8_t)h[i]) * UINT64_C(1099511628211);
        x = (x ^ (uint8_t)u[i]) * UINT64_C(1099511628211);
    }
    x = (x ^ (uint8_t)k) * UINT64_C(1099511628211);
    x = (x ^ (uint8_t)phase) * UINT64_C(1099511628211);
    /* Mix low bits before indexing the power-of-two table. */
    x ^= x >> 33; x *= UINT64_C(0xff51afd7ed558ccd);
    x ^= x >> 33; x *= UINT64_C(0xc4ceb9fe1a85ec53);
    x ^= x >> 33;
    *hash = x;
    return &c->memo[x & c->mask];
}

static int matches(const Entry *e, uint64_t hash, const int8_t *h,
                   const int8_t *u, int k, int phase) {
    return e->used && e->hash == hash && e->k == k && e->phase == phase
        && memcmp(e->hand, h, NT) == 0 && memcmp(e->unseen, u, NT) == 0;
}

static Bound finish(Entry *e, uint64_t hash, const int8_t *h,
                    const int8_t *u, int k, int phase, Bound b) {
    if (b.lo == b.hi) {
        e->hash = hash; e->value = b.lo; e->k = (uint8_t)k;
        e->phase = (uint8_t)phase; e->used = 1;
        memcpy(e->hand, h, NT); memcpy(e->unseen, u, NT);
    }
    return b;
}

static int exhausted(Context *c) {
    if (c->budget && c->nodes >= c->budget) { c->cutoffs++; return 1; }
    c->nodes++;
    return 0;
}

static Bound chance(Context *, int8_t *, int8_t *, int);

static Bound decision(Context *c, int8_t *h, int8_t *u, int k) {
    if (k <= 0) return exact(0.0); /* Winning draws were absorbed by chance. */
    uint64_t hash;
    Entry *e = lookup(c, h, u, k, 1, &hash);
    if (matches(e, hash, h, u, k, 1)) { c->hits++; return exact(e->value); }
    if (exhausted(c)) { Bound b = {0.0, 1.0}; return b; }
    Bound best = {0.0, 0.0};
    for (int d = 0; d < NT; ++d) {
        if (h[d] <= 0) continue;
        h[d]--;
        Bound b = chance(c, h, u, k);
        h[d]++;
        if (b.lo > best.lo) best.lo = b.lo;
        if (b.hi > best.hi) best.hi = b.hi;
        if (best.lo == 1.0) { best.hi = 1.0; break; }
    }
    return finish(e, hash, h, u, k, 1, best);
}

static Bound chance(Context *c, int8_t *h, int8_t *u, int k) {
    if (k <= 0) return exact(0.0);
    int n = 0;
    for (int t = 0; t < NT; ++t) n += u[t];
    if (n <= 0) return exact(0.0);
    if (k > n) k = n;
    uint64_t hash;
    Entry *e = lookup(c, h, u, k, 0, &hash);
    if (matches(e, hash, h, u, k, 0)) { c->hits++; return exact(e->value); }
    /* Shanten is only an impossibility bound, never an action ranking. */
    if (c->shanten(h) >= k)
        return finish(e, hash, h, u, k, 0, exact(0.0));
    if (exhausted(c)) { Bound b = {0.0, 1.0}; return b; }
    Bound sum = {0.0, 0.0};
    for (int t = 0; t < NT; ++t) {
        int w = u[t];
        if (w <= 0) continue;
        h[t]++; u[t]--;
        Bound b;
        if (c->win(h)) {
            b = exact(1.0);
        } else {
            b = decision(c, h, u, k - 1);
            b.lo *= c->discount; b.hi *= c->discount;
        }
        h[t]--; u[t]++;
        sum.lo += (double)w * b.lo;
        sum.hi += (double)w * b.hi;
    }
    sum.lo /= (double)n; sum.hi /= (double)n;
    return finish(e, hash, h, u, k, 0, sum);
}

void *fh_create(void *win_fn, void *shanten_fn, int cache_bits, double discount) {
    if (!win_fn || !shanten_fn || cache_bits < 8 || cache_bits > 22
            || !(discount > 0.0 && discount <= 1.0)) return NULL;
    Context *c = (Context *)calloc(1, sizeof(Context));
    if (!c) return NULL;
    size_t size = (size_t)1 << cache_bits;
    c->memo = (Entry *)calloc(size, sizeof(Entry));
    if (!c->memo) { free(c); return NULL; }
    c->win = (TileFn)win_fn; c->shanten = (TileFn)shanten_fn;
    c->mask = size - 1;
    c->discount = discount;
    return c;
}

void fh_destroy(void *ptr) {
    Context *c = (Context *)ptr;
    if (c) { free(c->memo); free(c); }
}

int fh_solve(void *ptr, const int8_t *hand, const int8_t *unseen, int k,
             uint64_t max_nodes, double *bounds, uint64_t *stats) {
    if (!ptr || !hand || !unseen || !bounds || !stats || k < 0) return -1;
    Context *c = (Context *)ptr;
    int8_t h[NT], u[NT];
    int total = 0, n = 0;
    for (int i = 0; i < NT; ++i) {
        if (hand[i] < 0 || unseen[i] < 0 || hand[i] + unseen[i] > 4) return -2;
        h[i] = hand[i]; u[i] = unseen[i]; total += h[i]; n += u[i];
    }
    if (total < 1 || total > 13 || total % 3 != 1) return -3;
    if (k > n) k = n;
    c->nodes = c->hits = c->cutoffs = 0; c->budget = max_nodes;
    Bound b = chance(c, h, u, k);
    bounds[0] = b.lo; bounds[1] = b.hi;
    stats[0] = c->nodes; stats[1] = c->hits; stats[2] = c->cutoffs;
    return 0;
}
