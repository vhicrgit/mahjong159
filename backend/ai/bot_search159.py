"""Experimental 159 search Bot with the existing live game's Bot API.

Use ``choose_action`` for unified discard/call/kong decisions. Legacy adapters
share a cached plan, so the server's repeated kong/peng/discard questions do
not re-search one decision or accidentally choose incompatible actions.
"""
from __future__ import annotations

from dataclasses import asdict
import os
import time

from ..native import native
from .search159 import simulator as sim
from .search159.planner import Planner, SearchConfig, unseen_counts


class Bot:
    def __init__(self, game, seat, *, simulations=None, depth=2,
                 candidate_limit=4, seed=0, mode="search", confirmation=None,
                 confidence_z=None, min_gain=0.0, base_policy="v31",
                 opponent_policy="v31", finite_horizon=2,
                 finite_discount=1.0, finite_max_nodes=100_000,
                 max_search_shanten=None, fast_rollout=True,
                 tree_key="history", paired_future=False,
                 future_candidate_limit=2):
        self.game, self.seat, self.mode = game, seat, mode
        if mode not in ("search", "tree", "root", "finite"):
            raise ValueError(f"unknown search mode {mode!r}")
        self.config = SearchConfig(
            simulations=int(simulations if simulations is not None else os.environ.get("SEARCH159_SIMS", 32)),
            depth=1 if mode == "root" else depth,
            candidate_limit=candidate_limit, confirmation=confirmation,
            confidence_z=float(confidence_z if confidence_z is not None else os.environ.get("SEARCH159_Z", 0)),
            min_gain=min_gain, base_policy=base_policy, opponent_policy=opponent_policy,
            finite_horizon=finite_horizon, finite_discount=finite_discount,
            finite_max_nodes=finite_max_nodes,
            max_search_shanten=max_search_shanten, fast_rollout=fast_rollout,
            tree_key=tree_key, paired_future=paired_future,
            future_candidate_limit=future_candidate_limit)
        self.planner = Planner(self.config, seed)
        self.stats = {}
        self.last_analysis = {}
        self._cached_observation = None
        self._cached_action = None

    def _finite(self, obs):
        from ..analysis.finite_horizon import rank_discards
        base = sim.base_action(self.game, self.seat, self.config.base_policy)
        if self.game.phase != "discard_wait" or base.kind != "discard":
            return base, {"reason": "baseline_call", "actions": []}
        k = min(self.config.finite_horizon, max(0, (obs.wall_length - 6) // 4))
        if k == 0:
            return base, {"reason": "no_nominal_draws", "actions": []}
        rows = rank_discards(obs.hand_counts, unseen_counts(obs), k,
                             max_nodes=self.config.finite_max_nodes,
                             discount=self.config.finite_discount)
        exact = [r for r in rows if r["exact"]]
        best = max(exact, key=lambda r: (r["value"], r["tile"] == base.tile)) if exact else None
        # Never rank overlapping budget-truncated bounds as exact probabilities.
        if best is None or any(r["upper_bound"] > best["value"] + 1e-12 for r in rows):
            return base, {"reason": "unresolved_finite_bounds", "actions": rows,
                          "model": "solitary_without_replacement"}
        return sim.Action("discard", best["tile"]), {
            "reason": "finite_completion", "horizon": k, "actions": rows,
            "discount": self.config.finite_discount,
            "model": "solitary_without_replacement"}

    def choose_action(self):
        obs = sim.observe(self.game, self.seat)
        if obs == self._cached_observation:
            return self._cached_action
        started = time.perf_counter()
        if self.mode == "finite":
            action, analysis = self._finite(obs)
            stats = {"finite_decisions": 1, "seconds": time.perf_counter() - started}
        elif (self.config.max_search_shanten is not None
              and self.game.phase == "discard_wait"
              and min(s for _, s in native.discard_shanten(obs.hand_counts)) > self.config.max_search_shanten):
            action = sim.base_action(self.game, self.seat, self.config.base_policy)
            analysis = {"reason": "shanten_budget_gate", "actions": []}
            stats = {"fallbacks": 1, "seconds": time.perf_counter() - started}
        else:
            action, analysis = self.planner.plan(obs)
            stats = self.planner.stats
        if action not in obs.own_legal_actions:
            raise RuntimeError(f"search produced an illegal action: {action}")
        for name, value in stats.items():
            self.stats[name] = self.stats.get(name, 0) + value
        self.last_analysis = analysis
        self._cached_observation, self._cached_action = obs, action
        return action

    def choose_discard(self):
        action = self.choose_action()
        if action.kind != "discard":
            raise RuntimeError("execute the selected kong before requesting a discard")
        return action.tile

    def decide_peng(self, tile):
        action = self.choose_action()
        return action.kind == "peng" and action.tile == tile

    def decide_gang(self, tile, kind):
        action = self.choose_action()
        return action.kind == "gang" and action.tile == tile

    def explain(self):
        """JSON-serializable diagnostics from the same decision used by the Bot."""
        self.choose_action()
        def convert(value):
            if isinstance(value, sim.Action):
                return asdict(value)
            if isinstance(value, dict):
                return {str(k): convert(v) for k, v in value.items()}
            if isinstance(value, (list, tuple)):
                return [convert(v) for v in value]
            return value
        return convert(self.last_analysis)
