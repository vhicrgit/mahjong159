"""Cheating bots with configurable wall and opponent information."""

from copy import deepcopy
import os

from ..rules.tiles import is_159
from ..rules.win import is_win, shanten
from ..rules.ting import discard_options, waiting_tiles
from .bot_oracle import WIN_DISCOUNT, search_first_discard_detail
from .bot_v1 import Bot as BotV1
from .bot_v2 import Bot as BotV2

RED = 27


class Bot:
    def __init__(self, game, seat: int, wall_lookahead: int = -1,
                 see_opponents: bool = True, beam: int = 18,
                 rollout: bool = False, search_depth: int = 2,
                 search_width: int = 2, root_width: int = 28):
        self.game = game
        self.seat = seat
        self.wall_lookahead = wall_lookahead
        self.see_opponents = see_opponents
        self.beam = beam
        self.rollout = rollout
        self.search_depth = int(os.environ.get("CHEAT_DEPTH", search_depth))
        self.search_width = int(os.environ.get("CHEAT_WIDTH", search_width))
        self.root_width = int(os.environ.get("CHEAT_ROOT_WIDTH", root_width))
        self.fallback = BotV2(game, seat)

    def _known_wall(self, game=None):
        g = game or self.game
        if self.wall_lookahead < 0:
            return g.wall
        return g.wall[:self.wall_lookahead]

    def _future_draws(self, game=None, max_draws: int = 18):
        g = game or self.game
        wall = self._known_wall(g)
        draws = []
        idx = 3
        hard_end = max(0, len(g.wall) - 6)
        visible_end = min(len(wall), hard_end)
        while idx < visible_end and len(draws) < max_draws:
            draws.append(wall[idx])
            idx += 4
        return draws

    def _fan_bonus(self, win_depth: int, game=None) -> float:
        g = game or self.game
        wall = self._known_wall(g)
        draw_idx = 3 + 4 * win_depth
        if draw_idx + 7 > len(wall):
            return 0.0
        n159 = sum(1 for t in wall[draw_idx + 1:draw_idx + 7] if is_159(t))
        return 0.03 * n159

    def _visible_counts(self, game=None):
        g = game or self.game
        visible = [0] * 28
        me = g.players[self.seat]
        for t, n in enumerate(me.hand_counts):
            visible[t] += n
        for q in g.players:
            for t in q.discards:
                visible[t] += 1
            for m in q.melds:
                visible[m["tile"]] += 3 if m["type"] == "peng" else 4
        return visible

    def _unseen_counts(self, game=None):
        visible = self._visible_counts(game)
        return [max(0, 4 - visible[t]) for t in range(28)]

    def _opponent_call_penalty(self, tile: int, game=None) -> float:
        if not self.see_opponents or tile == RED:
            return 0.0
        g = game or self.game
        penalty = 0.0
        for q in g.players:
            if q.seat == self.seat:
                continue
            cnt = q.hand.count(tile)
            if cnt >= 3:
                penalty += 0.28
            elif cnt >= 2:
                before = shanten(q.hand_counts)
                c = list(q.hand_counts)
                c[tile] -= 2
                if shanten(c) < before:
                    penalty += 0.10
        return penalty

    def _heuristic_score(self, counts14, tile, opt, game=None):
        unseen = self._unseen_counts(game)
        hand13 = list(counts14)
        hand13[tile] -= 1
        waits = waiting_tiles(hand13) if opt["shanten"] == 0 else []
        ukeire = sum(unseen[w] for w in waits)
        risk = self._opponent_call_penalty(tile, game)
        return -100.0 * opt["shanten"] + 3.0 * ukeire - 45.0 * risk

    def _ranked_oracle_discards(self, game=None, seat=None):
        g = game or self.game
        s = self.seat if seat is None else seat
        counts14 = list(g.players[s].hand_counts)
        opts = discard_options(counts14)
        if not opts:
            return [g.players[s].hand[-1]]
        future = self._future_draws(g)
        detail = search_first_discard_detail(counts14, future, self.beam) if future else {}
        ranked = []
        for o in opts:
            t = o["tile"]
            wd, sh = detail.get(t, (None, o["shanten"]))
            if wd is None:
                v = WIN_DISCOUNT ** (len(future) + 2 * max(0, sh) + 1)
            else:
                v = 1.0 + WIN_DISCOUNT ** wd + self._fan_bonus(wd, g)
            v += 0.01 * self._heuristic_score(counts14, t, o, g)
            ranked.append((v, t))
        ranked.sort(reverse=True)
        return [t for _, t in ranked]

    def _oracle_choice(self, game=None, seat=None):
        ranked = self._ranked_oracle_discards(game, seat)
        return ranked[0]

    def _choose_rollout_discard(self, game, depth: int) -> int:
        ranked = self._ranked_oracle_discards(game, self.seat)
        if depth <= 0 or len(ranked) == 1:
            return ranked[0]
        best_t, best_v = None, -1e18
        for t in ranked[:self.search_width]:
            game2 = deepcopy(game)
            try:
                game2.action_discard(self.seat, t)
            except Exception:
                continue
            v = self._rollout_to_end(game2, depth - 1)
            if v > best_v:
                best_t, best_v = t, v
        return best_t if best_t is not None else ranked[0]

    def _choose_reaction_rollout(self, game, depth: int) -> str:
        actions = ["pass"]
        pending = game.pending_actions.get(self.seat, {})
        if pending.get("peng"):
            actions.append("peng")
        if pending.get("gang"):
            actions.append("gang")
        if depth <= 0:
            if "gang" in actions and self._ming_gang_good(game):
                return "gang"
            if "peng" in actions and self._peng_good(game):
                return "peng"
            return "pass"
        best_action, best_v = "pass", -1e18
        for action in actions:
            game2 = deepcopy(game)
            try:
                if action == "pass":
                    game2.action_pass(self.seat)
                elif action == "peng":
                    game2.action_peng(self.seat)
                else:
                    game2.action_gang(self.seat)
            except Exception:
                continue
            v = self._rollout_to_end(game2, depth - 1)
            if v > best_v:
                best_action, best_v = action, v
        return best_action

    def _rollout_to_end(self, game, depth: int = 0) -> float:
        bots = {i: BotV1(game, i) for i in range(4) if i != self.seat}
        guard = 0
        while game.phase != "game_over" and guard < 260:
            guard += 1
            if game.phase == "discard_wait":
                if game.turn == self.seat:
                    tile = self._choose_rollout_discard(game, depth)
                else:
                    tile = bots[game.turn].choose_discard()
                game.action_discard(game.turn, tile)
            elif game.phase == "react_wait":
                s = list(game.pending_actions.keys())[0]
                if s == self.seat:
                    action = self._choose_reaction_rollout(game, depth)
                    if action == "gang":
                        game.action_gang(s)
                    elif action == "peng":
                        game.action_peng(s)
                    else:
                        game.action_pass(s)
                else:
                    b = bots[s]
                    if game.pending_actions[s].get("gang") and b.decide_gang(game.last_discard, "ming"):
                        game.action_gang(s)
                    elif game.pending_actions[s].get("peng") and b.decide_peng(game.last_discard):
                        game.action_peng(s)
                    else:
                        game.action_pass(s)
        delta = game.players[self.seat].score_delta
        if game.winner == self.seat:
            return 100.0 + delta
        if game.winner is None:
            return delta
        return -100.0 + delta

    def _root_rollout_score(self, tile: int) -> float:
        game = deepcopy(self.game)
        try:
            game.action_discard(self.seat, tile)
        except Exception:
            return -1e18
        return self._rollout_to_end(game, self.search_depth - 1)

    def _reaction_rollout_score(self, action: str) -> float:
        game = deepcopy(self.game)
        try:
            if action == "pass":
                game.action_pass(self.seat)
            elif action == "peng":
                game.action_peng(self.seat)
            elif action == "gang":
                game.action_gang(self.seat)
            else:
                return -1e18
        except Exception:
            return -1e18
        return self._rollout_to_end(game, self.search_depth - 1)

    def choose_discard(self) -> int:
        p = self.game.players[self.seat]
        opts = discard_options(list(p.hand_counts))
        if not opts:
            return p.hand[-1]
        if self.rollout and self.wall_lookahead < 0 and self.see_opponents:
            best_t, best_v = None, -1e18
            for t in self._ranked_oracle_discards()[:self.root_width]:
                v = self._root_rollout_score(t)
                if v > best_v:
                    best_t, best_v = t, v
            return best_t if best_t is not None else self._oracle_choice()
        return self._oracle_choice()

    def _ming_gang_good(self, game=None) -> bool:
        g = game or self.game
        tile = g.last_discard
        p = g.players[self.seat]
        c = list(p.hand_counts)
        c[tile] -= 3
        tail = g.wall[-1] if self.wall_lookahead < 0 and g.wall else None
        if tail is not None:
            c[tail] += 1
            if is_win(c):
                return True
            c[tail] -= 1
        before = shanten(p.hand_counts)
        after = shanten(c)
        return after <= before

    def _peng_good(self, game=None) -> bool:
        g = game or self.game
        tile = g.last_discard
        p = g.players[self.seat]
        before = shanten(p.hand_counts)
        c = list(p.hand_counts)
        c[tile] -= 2
        after = shanten(c)
        return after < before or (before == 0 and after == 0)

    def decide_peng(self, tile: int) -> bool:
        if self.rollout and self.wall_lookahead < 0 and self.see_opponents:
            return self._reaction_rollout_score("peng") > self._reaction_rollout_score("pass")
        return self._peng_good()

    def decide_gang(self, tile: int, kind: str) -> bool:
        if kind == "ming":
            if self.rollout and self.wall_lookahead < 0 and self.see_opponents:
                return self._reaction_rollout_score("gang") > self._reaction_rollout_score("pass")
            return self._ming_gang_good()
        if self.rollout and self.wall_lookahead < 0 and self.see_opponents:
            gang_game = deepcopy(self.game)
            try:
                gang_game.action_gang(self.seat, tile)
            except Exception:
                return False
            gang_v = self._rollout_to_end(gang_game)
            no_gang_v = max(self._root_rollout_score(o["tile"])
                            for o in discard_options(list(self.game.players[self.seat].hand_counts)))
            return gang_v > no_gang_v
        wall = self._known_wall()
        p = self.game.players[self.seat]
        c = list(p.hand_counts)
        if kind == "an":
            c[tile] -= 4
        else:
            c[tile] -= 1
        if self.wall_lookahead < 0 and wall:
            c[wall[-1]] += 1
            if is_win(c):
                return True
            c[wall[-1]] -= 1
        return self.fallback.decide_gang(tile, kind)
