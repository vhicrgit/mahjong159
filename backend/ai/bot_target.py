"""Finite-horizon target-probability rule bot."""

import os
from functools import lru_cache

from ..rules.win import is_win, shanten, shanten_cached
from ..rules.ting import discard_options, waiting_tiles

RED = 27
SHORT_HORIZON = 3
LONG_HORIZON = 5
TARGET_STEPS = 3
FRONTIER_CAP = 1
PROB_SCALE = 190.0
LONG_SCALE = 70.0
PROGRESS_SCALE = 14.0
RISK_SCALE = 25.0


def _tuple_add(c: tuple[int, ...], t: int, delta: int) -> tuple[int, ...]:
    a = list(c)
    a[t] += delta
    return tuple(a)


def _shanten(c: tuple[int, ...] | list[int]) -> int:
    return shanten_cached(tuple(c))


@lru_cache(maxsize=300000)
def _wait_count(hand13: tuple[int, ...], unseen: tuple[int, ...]) -> int:
    if _shanten(hand13) != 0:
        return 0
    return sum(unseen[w] for w in waiting_tiles(list(hand13)))


@lru_cache(maxsize=300000)
def _best_next_after_draw(hand14: tuple[int, ...], unseen: tuple[int, ...]):
    best_h, best_key = None, None
    for disc, cnt in enumerate(hand14):
        if cnt <= 0:
            continue
        h = _tuple_add(hand14, disc, -1)
        s = _shanten(h)
        w = _wait_count(h, unseen) if s == 0 else 0
        key = (s, -w, h[disc], disc)
        if best_key is None or key < best_key:
            best_key, best_h = key, h
    return best_h, best_key


def _rank_state(hand13: tuple[int, ...], mass: float,
                unseen: tuple[int, ...]) -> float:
    return mass * (8 - _shanten(hand13)) + 0.01 * _wait_count(hand13, unseen)


@lru_cache(maxsize=120000)
def _target_chain(hand13: tuple[int, ...], unseen: tuple[int, ...],
                  steps: int, frontier_cap: int):
    total = sum(unseen)
    if total <= 0:
        return []
    probs = [(t, n / total) for t, n in enumerate(unseen) if n > 0]
    frontier = {hand13: 1.0}
    chain = []
    for _ in range(steps):
        win_p = 0.0
        progress = {}
        for state, mass in frontier.items():
            base_s = _shanten(state)
            base_w = _wait_count(state, unseen)
            for draw, p_draw in probs:
                p = mass * p_draw
                hand14 = _tuple_add(state, draw, 1)
                if is_win(list(hand14)):
                    win_p += p
                    continue
                h2, key = _best_next_after_draw(hand14, unseen)
                s2 = key[0]
                w2 = -key[1]
                if s2 < base_s or (s2 == base_s and w2 > base_w):
                    progress[h2] = progress.get(h2, 0.0) + p
        prog_p = sum(progress.values())
        chain.append((win_p, prog_p))
        if prog_p <= 1e-12:
            break
        items = sorted(progress.items(),
                       key=lambda kv: _rank_state(kv[0], kv[1], unseen),
                       reverse=True)[:frontier_cap]
        norm = sum(v for _, v in items)
        frontier = {h: v / norm for h, v in items}
    return chain


def _chain_win_probability(chain: list[tuple[float, float]], turns: int) -> float:
    if turns <= 0 or not chain:
        return 0.0
    levels = {0: 1.0}
    win = 0.0
    for _ in range(turns):
        nxt = {}
        for level, mass in levels.items():
            if level >= len(chain):
                nxt[level] = nxt.get(level, 0.0) + mass
                continue
            w, u = chain[level]
            w = max(0.0, min(1.0, w))
            u = max(0.0, min(1.0 - w, u))
            win += mass * w
            nxt[level + 1] = nxt.get(level + 1, 0.0) + mass * u
            nxt[level] = nxt.get(level, 0.0) + mass * max(0.0, 1.0 - w - u)
        levels = nxt
    return max(0.0, min(1.0, win))


class Bot:
    def __init__(self, game, seat: int, short_horizon: int | None = None,
                 long_horizon: int | None = None):
        self.game = game
        self.seat = seat
        self.short_horizon = short_horizon if short_horizon is not None else int(os.environ.get("TARGET_SHORT_H", SHORT_HORIZON))
        self.long_horizon = long_horizon if long_horizon is not None else int(os.environ.get("TARGET_LONG_H", LONG_HORIZON))
        self.target_steps = int(os.environ.get("TARGET_STEPS", TARGET_STEPS))
        self.frontier_cap = int(os.environ.get("TARGET_FRONTIER_CAP", FRONTIER_CAP))
        self.prob_scale = float(os.environ.get("TARGET_PROB_SCALE", PROB_SCALE))
        self.long_scale = float(os.environ.get("TARGET_LONG_SCALE", LONG_SCALE))
        self.progress_scale = float(os.environ.get("TARGET_PROGRESS_SCALE", PROGRESS_SCALE))
        self.risk_scale = float(os.environ.get("TARGET_RISK_SCALE", RISK_SCALE))

    def _visible_counts(self) -> list[int]:
        visible = [0] * 28
        for q in self.game.players:
            for t in q.discards:
                visible[t] += 1
            for m in q.melds:
                visible[m["tile"]] += 3 if m["type"] == "peng" else 4
        p = self.game.players[self.seat]
        for t, n in enumerate(p.hand_counts):
            visible[t] += n
        return visible

    def _unseen_counts(self) -> tuple[int, ...]:
        visible = self._visible_counts()
        return tuple(max(0, 4 - visible[t]) for t in range(28))

    def _penged_by_others(self) -> set[int]:
        out = set()
        for q in self.game.players:
            if q.seat == self.seat:
                continue
            for m in q.melds:
                if m["type"] == "peng":
                    out.add(m["tile"])
        return out

    def _endgame_factor(self) -> float:
        return max(0.0, min(1.0, (60 - self.game.wall_remaining()) / 60.0))

    def choose_discard(self) -> int:
        p = self.game.players[self.seat]
        counts14 = tuple(p.hand_counts)
        opts = discard_options(list(counts14))
        if not opts:
            return p.hand[-1]
        if len(opts) == 1:
            return opts[0]["tile"]

        unseen = self._unseen_counts()
        penged = self._penged_by_others()
        eg = self._endgame_factor()
        risk_w = self.risk_scale * (1.0 + 1.5 * eg)
        sh_w = 100.0 * (1.0 - 0.45 * eg)
        long_h = min(self.long_horizon, max(1, self.game.wall_remaining() // 4 + 1))
        short_h = min(self.short_horizon, long_h)
        min_sh = min(o["shanten"] for o in opts)

        best_t, best_score = None, -1e18
        for o in opts:
            t = o["tile"]
            hand13 = _tuple_add(counts14, t, -1)
            sh = o["shanten"]
            ukeire = _wait_count(hand13, unseen)
            p_short = p_long = progress = 0.0
            if sh == min_sh:
                chain = _target_chain(hand13, unseen, self.target_steps, self.frontier_cap)
                p_short = _chain_win_probability(chain, short_h)
                p_long = _chain_win_probability(chain, long_h)
                progress = sum((i + 1) * u for i, (_, u) in enumerate(chain))
            risk = 0.0
            if t != RED:
                if t in penged:
                    risk = 1.0
                else:
                    risk = {3: 0.4, 2: 0.2, 1: 0.05, 0: 0.0}.get(unseen[t], 0.4)
            score = (-sh_w * sh + self.prob_scale * p_short + self.long_scale * p_long
                     + self.progress_scale * progress + 3.0 * ukeire - risk_w * risk)
            if score > best_score:
                best_score, best_t = score, t
        return best_t

    def decide_peng(self, tile: int) -> bool:
        p = self.game.players[self.seat]
        counts = p.hand_counts
        before = shanten(counts)
        c2 = list(counts)
        c2[tile] -= 2
        after = shanten(c2)
        if after < before:
            return before != 0 or after == 0
        return False

    def decide_gang(self, tile: int, kind: str) -> bool:
        p = self.game.players[self.seat]
        counts = p.hand_counts
        s_before = shanten(counts)
        c2 = list(counts)
        if kind == "ming":
            c2[tile] -= 3
        elif kind == "an":
            c2[tile] -= 4
        else:
            c2[tile] -= 1
        s_after = shanten(c2)
        return not (s_before == 0 and s_after > 0)
