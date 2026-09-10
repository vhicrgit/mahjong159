/* Complete production-rule v31 continuation. No tile LUT is duplicated here.
 * Native primitive function pointers are supplied by the Python host once per
 * call. These are C function addresses, never Python callbacks. */
#include <stdint.h>
#include <stddef.h>
#include <string.h>

#define S159_TILES 28
#define S159_SEATS 4
#define S159_WALL 112
#define S159_DISCARDS 112
#define S159_MELDS 4
#define S159_GANGS 16

enum { S159_DISCARD = 0, S159_REACT = 1, S159_OVER = 2 };
enum { S159_PENG = 0, S159_GANG = 1 };
enum { S159_MING = 0, S159_AN = 1, S159_BU = 2 };
enum { S159_OK = 0, S159_BAD_STATE = 1, S159_GUARD = 2,
       S159_BAD_ACTION = 3, S159_CAPACITY = 4 };

typedef struct {
    int32_t tile, type, kind, wr;
} S159Meld;

typedef struct {
    int8_t hand[S159_TILES];
    int8_t discards[S159_DISCARDS];
    int32_t n_discards;
    S159Meld melds[S159_MELDS];
    int32_t n_melds;
    int32_t score;
} S159Player;

typedef struct {
    int32_t seat, kind, tile, from_seat;
} S159Gang;

typedef struct {
    S159Player players[S159_SEATS];
    int8_t wall[S159_WALL];
    int32_t head, tail;
    S159Gang gangs[S159_GANGS];
    int32_t n_gangs;
    int32_t phase, turn, last_discard, last_discarder;
    uint8_t pending[S159_SEATS]; /* peng bit 1, gang bit 2 */
    int8_t pending_order[S159_SEATS];
    int32_t n_pending;
    int32_t winner, win_tile, win_kind, huangzhuang, n_159;
    int8_t fan_159[6];
    int32_t n_fan;
    int32_t last_drawn_seat, last_drawn_tile;
    /* 0: preserve initial text, 1: discard, 2: normal draw, 3: gang draw */
    int32_t last_action_kind, last_action_seat, last_action_tile;
    int32_t steps;
    double expected_scores[S159_SEATS];
} S159State;

typedef int (*S159IsWin)(const int8_t *);
typedef int (*S159Choose)(const int8_t *, const int8_t *, const uint8_t *,
                          double, double, double, double, double, int);
typedef int (*S159PengDecision)(int, const int8_t *, int);
typedef int (*S159GangDecision)(int, const int8_t *, int, int);

typedef struct {
    S159IsWin is_win;
    S159Choose choose;
    S159PengDecision peng;
    S159GangDecision gang;
} S159Functions;

typedef struct {
    double sw, uw, cw, rw;
    int32_t cont_max;
} S159Params;

int search159_abi_version(void) { return 1; }
size_t search159_sizeof_state(void) { return sizeof(S159State); }
size_t search159_sizeof_params(void) { return sizeof(S159Params); }

static int s159_remaining(const S159State *s) { return s->tail - s->head; }
static int s159_is_159(int t) {
    return t < 27 && (t % 9 == 0 || t % 9 == 4 || t % 9 == 8);
}

static void s159_clear_pending(S159State *s) {
    memset(s->pending, 0, sizeof(s->pending));
    s->n_pending = 0;
}

static void s159_gang_scores(const S159State *s, double out[4]) {
    for (int i = 0; i < s->n_gangs; i++) {
        const S159Gang *g = &s->gangs[i];
        out[g->seat] += 3.0;
        if (g->kind == S159_MING) out[g->from_seat] -= 3.0;
        else for (int p = 0; p < 4; p++) if (p != g->seat) out[p] -= 1.0;
    }
}

static void s159_expected(S159State *s) {
    for (int p = 0; p < 4; p++) s->expected_scores[p] = 0.0;
    if (s->winner < 0) return; /* No gang settlement on a drawn game. */
    s159_gang_scores(s, s->expected_scores);
    int n = s159_remaining(s), m = 0;
    for (int i = s->head; i < s->tail; i++) m += s159_is_159(s->wall[i]);
    double per = 1.0 + (n >= 6 ? 6.0 * m / n : 0.0);
    for (int p = 0; p < 4; p++)
        s->expected_scores[p] += (p == s->winner ? 3.0 : -1.0) * per;
}

static void s159_hu(S159State *s, int seat, int tile, int kind) {
    s->winner = seat;
    s->win_tile = tile;
    s->win_kind = kind;
    s->n_159 = 0;
    s->n_fan = 0;
    if (s159_remaining(s) >= 6) {
        s->n_fan = 6;
        for (int i = 0; i < 6; i++) {
            s->fan_159[i] = s->wall[s->head + i];
            s->n_159 += s159_is_159(s->fan_159[i]);
        }
    }
    double scores[4] = {0, 0, 0, 0};
    s159_gang_scores(s, scores);
    for (int p = 0; p < 4; p++) {
        scores[p] += (p == seat ? 3 : -1) * (s->n_159 + 1);
        s->players[p].score = (int32_t)scores[p];
    }
    s->phase = S159_OVER;
}

static void s159_normal_draw(S159State *s, const S159Functions *f) {
    s->turn = (s->last_discarder + 1) % 4;
    if (s159_remaining(s) <= 6) {
        s->huangzhuang = 1;
        s->phase = S159_OVER;
        return;
    }
    int seat = s->turn, tile = s->wall[s->head++];
    s->players[seat].hand[tile]++;
    s->last_drawn_seat = seat;
    s->last_drawn_tile = tile;
    s->last_action_kind = 2;
    s->last_action_seat = seat;
    s->last_action_tile = tile;
    if (f->is_win(s->players[seat].hand)) s159_hu(s, seat, tile, 1);
    else s->phase = S159_DISCARD;
}

static void s159_gang_draw(S159State *s, int seat, const S159Functions *f) {
    if (s159_remaining(s) == 0) {
        s->huangzhuang = 1;
        s->phase = S159_OVER;
        return;
    }
    int tile = s->wall[--s->tail];
    s->players[seat].hand[tile]++;
    s->last_drawn_seat = seat;
    s->last_drawn_tile = tile;
    s->last_action_kind = 3;
    s->last_action_seat = seat;
    s->last_action_tile = tile;
    if (f->is_win(s->players[seat].hand)) s159_hu(s, seat, tile, 2);
    else {
        s->phase = S159_DISCARD;
        s->turn = seat;
    }
}

static int s159_discard(S159State *s, int seat, int tile, const S159Functions *f) {
    S159Player *p = &s->players[seat];
    if (tile < 0 || tile >= 28 || p->hand[tile] <= 0) return S159_BAD_ACTION;
    if (p->n_discards == S159_DISCARDS) return S159_CAPACITY;
    p->hand[tile]--;
    p->discards[p->n_discards++] = (int8_t)tile;
    s->last_discard = tile;
    s->last_discarder = seat;
    s->last_drawn_seat = -1;
    s->last_drawn_tile = -1;
    s->last_action_kind = 1;
    s->last_action_seat = seat;
    s->last_action_tile = tile;
    s159_clear_pending(s);
    if (tile != 27) {
        for (int other = 0; other < 4; other++) {
            int n = s->players[other].hand[tile];
            if (other != seat && n >= 2) {
                s->pending[other] = (uint8_t)(1 | (n >= 3 ? 2 : 0));
                s->pending_order[s->n_pending++] = (int8_t)other;
            }
        }
    }
    if (s->n_pending > 0) s->phase = S159_REACT;
    else s159_normal_draw(s, f);
    return S159_OK;
}

static void s159_pop_claimed_discard(S159State *s, int tile) {
    S159Player *p = &s->players[s->last_discarder];
    if (p->n_discards > 0 && p->discards[p->n_discards - 1] == tile) p->n_discards--;
}

static int s159_peng(S159State *s, int seat) {
    S159Player *p = &s->players[seat];
    int tile = s->last_discard;
    if (p->n_melds == S159_MELDS) return S159_CAPACITY;
    if (tile < 0 || tile >= 27 || p->hand[tile] < 2) return S159_BAD_ACTION;
    p->hand[tile] -= 2;
    p->melds[p->n_melds++] = (S159Meld){tile, S159_PENG, -1, s159_remaining(s)};
    s159_pop_claimed_discard(s, tile);
    s159_clear_pending(s);
    s->turn = seat;
    s->phase = S159_DISCARD;
    return S159_OK;
}

static int s159_gang(S159State *s, int seat, int tile, const S159Functions *f) {
    S159Player *p = &s->players[seat];
    if (s->n_gangs == S159_GANGS) return S159_CAPACITY;
    if (tile < 0 || tile >= 27) return S159_BAD_ACTION;
    int kind;
    if (s->phase == S159_REACT) {
        if (p->n_melds == S159_MELDS) return S159_CAPACITY;
        if (p->hand[tile] < 3) return S159_BAD_ACTION;
        p->hand[tile] -= 3;
        kind = S159_MING;
        p->melds[p->n_melds++] = (S159Meld){tile, S159_GANG, kind, s159_remaining(s)};
        s159_pop_claimed_discard(s, tile);
    } else if (p->hand[tile] == 4) {
        if (p->n_melds == S159_MELDS) return S159_CAPACITY;
        p->hand[tile] -= 4;
        kind = S159_AN;
        p->melds[p->n_melds++] = (S159Meld){tile, S159_GANG, kind, s159_remaining(s)};
    } else {
        int found = -1;
        for (int i = 0; i < p->n_melds; i++)
            if (p->melds[i].type == S159_PENG && p->melds[i].tile == tile) { found = i; break; }
        if (p->hand[tile] != 1 || found < 0) return S159_BAD_ACTION;
        p->hand[tile]--;
        kind = S159_BU;
        p->melds[found].type = S159_GANG;
        p->melds[found].kind = kind; /* wr remains the original peng's wr. */
    }
    s->gangs[s->n_gangs++] = (S159Gang){seat, kind, tile,
                                                      kind == S159_MING ? s->last_discarder : -1};
    s159_clear_pending(s);
    s->turn = seat;
    s159_gang_draw(s, seat, f);
    return S159_OK;
}

static void s159_pass(S159State *s, int seat, const S159Functions *f) {
    s->pending[seat] = 0;
    for (int i = 1; i < s->n_pending; i++) s->pending_order[i - 1] = s->pending_order[i];
    s->n_pending--;
    if (s->n_pending == 0) s159_normal_draw(s, f);
}

static int s159_step(S159State *s, const S159Functions *f, const S159Params params[4]) {
    if (s->phase == S159_REACT) {
        if (s->n_pending <= 0) return S159_BAD_STATE;
        int seat = s->pending_order[0], tile = s->last_discard;
        const int8_t *hand = s->players[seat].hand;
        if ((s->pending[seat] & 2) && f->gang(31, hand, tile, S159_MING))
            return s159_gang(s, seat, tile, f);
        if ((s->pending[seat] & 1) && f->peng(31, hand, tile)) return s159_peng(s, seat);
        s159_pass(s, seat, f);
        return S159_OK;
    }
    if (s->phase != S159_DISCARD || s->turn < 0 || s->turn >= 4) return S159_BAD_STATE;
    int seat = s->turn;
    S159Player *player = &s->players[seat];
    /* Game._gang_options sorts the union of concealed and added gangs. */
    for (int tile = 0; tile < 27; tile++) {
        int kind = -1;
        if (player->hand[tile] == 4) kind = S159_AN;
        else if (player->hand[tile] > 0) {
            for (int i = 0; i < player->n_melds; i++)
                if (player->melds[i].type == S159_PENG && player->melds[i].tile == tile) {
                    kind = S159_BU;
                    break;
                }
        }
        if (kind >= 0 && f->gang(31, player->hand, tile, kind))
            return s159_gang(s, seat, tile, f);
    }
    int visible[28];
    int8_t unseen[28];
    uint8_t penged[28] = {0};
    for (int t = 0; t < 28; t++) visible[t] = player->hand[t];
    for (int p = 0; p < 4; p++) {
        const S159Player *q = &s->players[p];
        for (int i = 0; i < q->n_discards; i++) visible[q->discards[i]]++;
        for (int i = 0; i < q->n_melds; i++) {
            const S159Meld *m = &q->melds[i];
            visible[m->tile] += m->type == S159_PENG ? 3 : 4;
            if (p != seat && m->type == S159_PENG) penged[m->tile] = 1;
        }
    }
    for (int t = 0; t < 28; t++) unseen[t] = (int8_t)(visible[t] < 4 ? 4 - visible[t] : 0);
    double eg = (60.0 - s159_remaining(s)) / 60.0;
    if (eg < 0) eg = 0;
    if (eg > 1) eg = 1;
    const S159Params *p = &params[seat];
    int tile = f->choose(player->hand, unseen, penged, eg,
                         p->sw, p->uw, p->cw, p->rw, p->cont_max);
    return s159_discard(s, seat, tile, f);
}

int search159_rollout(S159State *s, const S159Functions *f,
                      const S159Params params[4], int max_steps) {
    if (!s || !f || !params || !f->is_win || !f->choose || !f->peng || !f->gang ||
        max_steps < 0 || s->head < 0 || s->tail < s->head || s->tail > S159_WALL ||
        s->n_gangs < 0 || s->n_gangs > S159_GANGS || s->n_pending < 0 || s->n_pending > 4)
        return S159_BAD_STATE;
    s->steps = 0;
    while (s->phase != S159_OVER && s->steps < max_steps) {
        int rc = s159_step(s, f, params);
        if (rc != S159_OK) return rc;
        s->steps++;
    }
    if (s->phase != S159_OVER) return S159_GUARD;
    s159_expected(s);
    return S159_OK;
}
