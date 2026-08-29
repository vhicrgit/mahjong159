"""原生(C)加速版 cheat_full —— 与 backend.ai.bot_cheat.Bot 逐位同口径。

profile 结论(tools/perf/profile_cheat_full.py): 一次 cheat_full 决策 6.64s,
其中 96% 花在 bot_oracle.search_first_discard_detail 的 beam search 上,
引擎/deepcopy/BotV1 只占 4%。所以这里只替掉叶子级的纯函数:

  search_first_discard_detail -> native.beam_detail      (546x, 0 分歧)
  discard_options 的 shanten  -> native.discard_shanten  (省掉用不到的 waits)
  _heuristic_score 的 ukeire  -> native.waits_ukeire
  shanten / is_win            -> native.shanten / is_win
  嵌套 rollout 里的 BotV1     -> NativeV1

搜索结构(root_width / search_depth / search_width / beam)一点没动, 所以
出牌结果与 bot_cheat 完全一致, 由 backend/ai/test_parity_cheat.py 把关。
"""

from copy import deepcopy

from ..native import native
from .bot_cheat import Bot as PyCheat
from .bot_native import NativeV1
from .bot_oracle import WIN_DISCOUNT

RED = 27


class NativeCheat(PyCheat):
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
                counts = q.hand_counts
                before = native.shanten(counts)
                c = list(counts)
                c[tile] -= 2
                if native.shanten(c) < before:
                    penalty += 0.10
        return penalty

    def _heuristic_score(self, counts14, tile, opt, game=None):
        unseen = self._unseen_counts(game)
        hand13 = list(counts14)
        hand13[tile] -= 1
        ukeire = native.waits_ukeire(hand13, unseen) \
            if opt["shanten"] == 0 else 0
        risk = self._opponent_call_penalty(tile, game)
        return -100.0 * opt["shanten"] + 3.0 * ukeire - 45.0 * risk

    def _ranked_oracle_discards(self, game=None, seat=None):
        g = game or self.game
        s = self.seat if seat is None else seat
        counts14 = list(g.players[s].hand_counts)
        opts = native.discard_shanten(counts14)
        if not opts:
            return [g.players[s].hand[-1]]
        future = self._future_draws(g)
        detail = native.beam_detail(counts14, future, self.beam) if future \
            else {}
        ranked = []
        for tile, sh0 in opts:
            wd, sh = detail.get(tile, (None, sh0))
            if wd is None:
                v = WIN_DISCOUNT ** (len(future) + 2 * max(0, sh) + 1)
            else:
                v = 1.0 + WIN_DISCOUNT ** wd + self._fan_bonus(wd, g)
            v += 0.01 * self._heuristic_score(counts14, tile,
                                              {"shanten": sh0}, g)
            ranked.append((v, tile))
        ranked.sort(reverse=True)
        return [t for _, t in ranked]

    def _rollout_to_end(self, game, depth: int = 0) -> float:
        bots = {i: NativeV1(game, i) for i in range(4) if i != self.seat}
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
                    if game.pending_actions[s].get("gang") and \
                            b.decide_gang(game.last_discard, "ming"):
                        game.action_gang(s)
                    elif game.pending_actions[s].get("peng") and \
                            b.decide_peng(game.last_discard):
                        game.action_peng(s)
                    else:
                        game.action_pass(s)
        delta = game.players[self.seat].score_delta
        if game.winner == self.seat:
            return 100.0 + delta
        if game.winner is None:
            return delta
        return -100.0 + delta

    def choose_discard(self) -> int:
        p = self.game.players[self.seat]
        if not any(p.hand_counts):
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
        counts = p.hand_counts
        c = list(counts)
        c[tile] -= 3
        tail = g.wall[-1] if self.wall_lookahead < 0 and g.wall else None
        if tail is not None:
            c[tail] += 1
            if native.is_win(c):
                return True
            c[tail] -= 1
        return native.shanten(c) <= native.shanten(counts)

    def _peng_good(self, game=None) -> bool:
        g = game or self.game
        tile = g.last_discard
        counts = g.players[self.seat].hand_counts
        before = native.shanten(counts)
        c = list(counts)
        c[tile] -= 2
        after = native.shanten(c)
        return after < before or (before == 0 and after == 0)


class NativeCheatFull(NativeCheat):
    """cheat_full 的原生版: 全信息 + rollout 搜索。"""

    def __init__(self, game, seat: int, **kw):
        kw.setdefault("wall_lookahead", -1)
        kw.setdefault("see_opponents", True)
        kw.setdefault("rollout", True)
        super().__init__(game, seat, **kw)
