"""Observation-history Monte Carlo planning for the complete 159 game.

Hidden states belong to the simulator, never to tree keys or action selection.
Root candidates share worlds. By default, subsequent hero decisions share
statistics only when public actions AND hero observations match. An optional
public-hand abstraction drops some observable history to increase reuse; it
never includes private information. A separate set of worlds evaluates the
frozen learned continuations;
reported root standard errors come from that fixed-policy confirmation set,
not the adaptively changing tree's training returns.

The current belief is a uniform, jointly feasible hidden-state distribution.
It conditions on public counts and the current decision opportunity, not on
opponent action likelihoods from the entire preceding game. This approximation
is deliberately exposed in diagnostics rather than called a full posterior.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
import math
from numbers import Integral, Real
import random
import time

from ...native import native
from . import simulator as sim


@dataclass(frozen=True)
class SearchConfig:
    simulations: int = 32
    depth: int = 2
    candidate_limit: int = 4
    confirmation: int | None = None
    exploration: float = 3.0
    min_node_visits: int = 2
    confidence_z: float = 0.0
    min_gain: float = 0.0
    base_policy: str = "v31"
    opponent_policy: str = "v31"
    finite_horizon: int = 2
    finite_discount: float = 1.0
    finite_max_nodes: int | None = 100_000
    max_search_shanten: int | None = None
    fast_rollout: bool = True
    tree_key: str = "history"
    paired_future: bool = False
    future_candidate_limit: int = 2

    def __post_init__(self):
        def integer(value, minimum=1):
            return isinstance(value, Integral) and not isinstance(value, bool) and value >= minimum

        def finite_real(value):
            if not isinstance(value, Real) or isinstance(value, bool):
                return False
            try:
                return math.isfinite(value)
            except OverflowError:
                return False

        for name in ("simulations", "depth", "candidate_limit", "min_node_visits",
                     "finite_horizon", "future_candidate_limit"):
            if not integer(getattr(self, name)):
                raise ValueError(f"{name} must be a non-boolean positive integer")
        if self.confirmation is not None and not integer(self.confirmation, 2):
            raise ValueError("confirmation must be a non-boolean integer of at least 2")
        for name in ("exploration", "confidence_z", "min_gain"):
            value = getattr(self, name)
            if not finite_real(value) or value < 0:
                raise ValueError(f"{name} must be a finite nonnegative real number")
        if self.finite_max_nodes is not None and (
                not integer(self.finite_max_nodes) or self.finite_max_nodes > (1 << 64) - 1):
            raise ValueError("finite_max_nodes must be None or a non-boolean positive uint64")
        if not finite_real(self.finite_discount) or not 0 < self.finite_discount <= 1:
            raise ValueError("finite_discount must be a finite non-boolean real number in (0, 1]")
        if self.max_search_shanten is not None and (
                not isinstance(self.max_search_shanten, Integral) or isinstance(self.max_search_shanten, bool)):
            raise ValueError("max_search_shanten must be None or a non-boolean integer")
        if self.tree_key not in {"history", "public_hand"}:
            raise ValueError("tree_key must be 'history' or 'public_hand'")
        if not isinstance(self.paired_future, bool):
            raise ValueError("paired_future must be a boolean")

    @property
    def effective_future_candidate_limit(self):
        # Preserve the pre-abstraction default's candidate set. The separate
        # future budget becomes active with either experimental feature.
        if self.tree_key == "history" and not self.paired_future:
            return self.candidate_limit
        return self.future_candidate_limit


def unseen_counts(obs: sim.Observation) -> tuple[int, ...]:
    visible = list(obs.hand_counts)
    for discards in obs.discards:
        for tile in discards:
            visible[tile] += 1
    for melds in obs.melds:
        for meld in melds:
            visible[meld.tile] += 3 if meld.type == "peng" else 4
    result = tuple(4 - n for n in visible)
    if min(result) < 0:
        raise ValueError("public information exceeds four copies of a tile")
    return result


def observation_seed(obs, seed: int, stream: str) -> int:
    """Stable across processes; neither the Game RNG nor actual wall is used."""
    payload = f"{seed}:{stream}:{obs!r}".encode()
    return int.from_bytes(hashlib.blake2b(payload, digest_size=16).digest(), "big")


def public_hand_key(obs: sim.Observation, actions: tuple[sim.Action, ...]):
    """An information-limited policy abstraction, not a complete information set.

    It deliberately omits the discard history/count vectors, intervening trace
    and own-draw provenance. Consequently it may pool states with different
    public tile availability. Candidate ordering is computed from public data
    and retained so a shared node always has the same baseline and actions.
    Hidden hands, wall order and opponent action availability never enter it.
    """
    return ("public_hand", obs.hero, obs.hand_counts, obs.melds,
            obs.gang_records, obs.own_legal_actions, obs.phase, obs.turn,
            obs.last_discard, obs.last_discarder, obs.wall_length, tuple(actions))


def candidates(game, hero, config):
    """All calls plus ranked discards; baseline is always retained.

    The limit applies to discards, not calls. This is heuristic pruning, not
    a proof that omitted actions are dominated. Set the root and effective
    future limits to 28 to retain every discard.
    """
    legal = sim.legal_actions(game, hero)
    base = sim.base_action(game, hero, config.base_policy)
    if not legal or base not in legal:
        raise ValueError("planner must run at a legal hero decision")
    if game.phase == "react_wait":
        return (base,) + tuple(a for a in legal if a != base), base
    obs = sim.observe(game, hero)
    u = unseen_counts(obs)
    rows = native.score_discards_v10(obs.hand_counts, u, [0] * 28, 0.0)
    rows.sort(key=lambda r: (-r["score"], r["tile"]))
    selected = [sim.Action("discard", r["tile"])
                for r in rows[:config.candidate_limit]]
    if base.kind == "discard" and base not in selected:
        selected[-1:] = [base]
    all_calls = [a for a in legal if a.kind != "discard"]
    result = [base]
    for action in all_calls + selected:
        if action not in result:
            result.append(action)
    return tuple(result), base


@dataclass
class Edge:
    visits: int = 0
    total: float = 0.0
    children: dict = field(default_factory=dict)

    @property
    def mean(self):
        return self.total / self.visits if self.visits else 0.0


@dataclass
class Node:
    actions: tuple
    baseline: sim.Action
    edges: dict = field(init=False)
    visits: int = 0

    def __post_init__(self):
        self.edges = {a: Edge() for a in self.actions}

    def select(self, config, *, frozen=False):
        if frozen:
            supported = [a for a in self.actions
                         if self.edges[a].visits >= config.min_node_visits]
            if not supported:
                return self.baseline
            return max(supported, key=lambda a: (self.edges[a].mean,
                                                a == self.baseline))
        for action in self.actions:
            if not self.edges[action].visits:
                return action
        return max(self.actions, key=lambda a: (
            self.edges[a].mean + config.exploration * math.sqrt(
                math.log(self.visits + 1) / self.edges[a].visits),
            a == self.baseline))


def _mean(values):
    return sum(values) / len(values)


def _se(values):
    if len(values) < 2:
        return None
    mean = _mean(values)
    return math.sqrt(sum((x - mean) ** 2 for x in values)
                     / ((len(values) - 1) * len(values)))


class Planner:
    def __init__(self, config=None, seed=0):
        self.config = config or SearchConfig()
        self.future_config = replace(
            self.config, candidate_limit=self.config.effective_future_candidate_limit)
        self.seed = int(seed)
        self.stats = {}
        self.last_tree = None

    def _node(self, game, hero, *, future=False, prepared=None):
        actions, base = (prepared if prepared is not None else
                         candidates(game, hero, self.future_config if future else self.config))
        self.stats["tree_nodes"] += 1
        return Node(actions, base)

    def _future_key(self, game, hero, trace):
        obs = sim.observe(game, hero)
        if self.config.tree_key == "history":
            return (trace, obs), None
        prepared = candidates(game, hero, self.future_config)
        return public_hand_key(obs, prepared[0]), prepared

    def _tree_settings(self):
        return {"tree_key": self.config.tree_key,
                "paired_future": self.config.paired_future,
                "future_candidate_limit": self.config.future_candidate_limit,
                "effective_future_candidate_limit": self.config.effective_future_candidate_limit}

    def _policies(self, hero):
        return tuple(self.config.base_policy if s == hero
                     else self.config.opponent_policy for s in range(4))

    def _rollout(self, game, hero):
        policies = self._policies(hero)
        # Keep the Python production-engine path as a reference and for other
        # continuation policies. The native path uses the same rule primitives
        # and has complete-state parity tests, including all kinds of kong.
        from ...game.engine import Game
        if self.config.fast_rollout and isinstance(game, Game):
            from .fast_rollout import supports, rollout_fast
            if supports(policies):
                result = rollout_fast(game, hero, policies, mutate=True)
            else:
                result = sim.rollout(game, hero, policies)
        else:
            result = sim.rollout(game, hero, policies)
        self.stats["rollout_steps"] += result.steps
        self.stats["leaf_rollouts"] += 1
        return result.hero_score

    def _after(self, node, action, game, hero, remaining, *, frozen):
        sim.apply_action(game, action, hero)
        trace = sim.advance(game, hero, self._policies(hero))
        if game.phase == "game_over":
            value = sim.expected_scores(game)[hero]
            self.stats["leaf_rollouts"] += 1
        elif remaining <= 1:
            value = self._rollout(game, hero)
        else:
            key, prepared = self._future_key(game, hero, trace)
            edge = node.edges[action]
            child = edge.children.get(key)
            if child is None:
                if frozen:
                    self.stats["frozen_tree_misses"] += 1
                    return self._rollout(game, hero)
                child = self._node(game, hero, future=True, prepared=prepared)
                edge.children[key] = child
            elif frozen:
                self.stats["frozen_tree_hits"] += 1
            else:
                self.stats["revisited_nodes"] += 1
            chosen = child.select(self.config, frozen=frozen)
            if frozen and chosen != child.baseline:
                self.stats["future_overrides"] += 1
            if self.config.paired_future and not frozen:
                # Select BEFORE observing this world's action returns. Evaluate
                # alternatives from the same pre-action world, but return only
                # the chosen action's value to the parent. A per-world max would
                # give the policy access to hidden information (strategy fusion).
                snapshot = sim.clone(game)
                self.stats["paired_evaluations"] += len(child.actions)
                value = self._after(child, chosen, game, hero, remaining - 1,
                                    frozen=False)
                for alternative in child.actions:
                    if alternative != chosen:
                        self._after(child, alternative, sim.clone(snapshot), hero,
                                    remaining - 1, frozen=False)
            else:
                value = self._after(child, chosen, game, hero, remaining - 1,
                                    frozen=frozen)
        if not frozen:
            edge = node.edges[action]
            edge.visits += 1
            edge.total += value
            node.visits += 1
        return value

    def plan(self, observation):
        started = time.perf_counter()
        self.stats = {"searches": 1, "simulations": 0, "tree_nodes": 0,
                      "rollout_steps": 0, "root_candidates": 0,
                      "fallbacks": 0, "overrides": 0, "seconds": 0.0,
                      "frozen_tree_hits": 0, "frozen_tree_misses": 0,
                      "revisited_nodes": 0, "future_overrides": 0,
                      # Actual terminal trajectory evaluations, including an
                      # advance that already ends the hand. Unlike simulations,
                      # this counts all extra paired-future branches.
                      "leaf_rollouts": 0,
                      # All forced candidate evaluations in paired future
                      # groups, including the preselected candidate.
                      "paired_evaluations": 0}
        rng = random.Random(observation_seed(observation, self.seed, "train"))
        # Even candidate generation receives a sampled Game, never the actual
        # hidden state. It only uses public fields and the hero's own hand.
        world = sim.sample_world(observation, rng)
        hero = observation.hero
        root = self._node(world, hero)
        self.last_tree = root
        self.stats["root_candidates"] = len(root.actions)
        if len(root.actions) == 1:
            self.stats["fallbacks"] = 1
            self.stats["seconds"] = time.perf_counter() - started
            return root.baseline, {"selected": root.baseline, "baseline": root.baseline,
                                   "actions": [], "reason": "only_one_candidate",
                                   "belief": "uniform_joint_counts",
                                   **self._tree_settings()}
        # A depth-one planner has no continuation tree to learn. Spend its
        # entire budget evaluating the fixed base policy instead of discarding
        # a training batch that cannot influence the selected action.
        train_rounds = self.config.simulations if self.config.depth > 1 else 0
        for i in range(train_rounds):
            world = sim.sample_world(observation, rng)
            # Rotate processing order to avoid systematic order effects in any
            # future shared infrastructure. Every candidate gets every world.
            actions = root.actions[i % len(root.actions):] + root.actions[:i % len(root.actions)]
            for action in actions:
                self._after(root, action, sim.clone(world), hero,
                            self.config.depth, frozen=False)
                self.stats["simulations"] += 1

        count = self.config.confirmation or max(16, self.config.simulations // 2)
        if self.config.depth == 1:
            count += self.config.simulations
        rng = random.Random(observation_seed(observation, self.seed, "confirm"))
        returns = {a: [] for a in root.actions}
        wins = {a: 0 for a in root.actions}
        draws = {a: 0 for a in root.actions}
        for _ in range(count):
            world = sim.sample_world(observation, rng)
            for action in root.actions:
                g = sim.clone(world)
                value = self._after(root, action, g, hero,
                                    self.config.depth, frozen=True)
                returns[action].append(value)
                wins[action] += g.winner == hero
                draws[action] += g.huangzhuang
                self.stats["simulations"] += 1

        records = []
        baseline_returns = returns[root.baseline]
        for action in root.actions:
            values = returns[action]
            delta = [a - b for a, b in zip(values, baseline_returns)]
            se = _se(delta)
            records.append({"action": action, "expected_score": _mean(values),
                            "delta_vs_baseline": _mean(delta), "paired_se": se,
                            "win_probability": wins[action] / count,
                            "draw_probability": draws[action] / count,
                            "samples": count, "train_visits": root.edges[action].visits,
                            "train_mean": root.edges[action].mean})
        best = max(records, key=lambda r: (r["delta_vs_baseline"]
                    - self.config.confidence_z * (r["paired_se"] or 0.0),
                    r["action"] == root.baseline))
        selected = root.baseline
        margin = best["delta_vs_baseline"] - self.config.confidence_z * (best["paired_se"] or 0.0)
        if margin > self.config.min_gain:
            selected = best["action"]
        self.stats["overrides"] = int(selected != root.baseline)
        self.stats["seconds"] = time.perf_counter() - started
        records.sort(key=lambda r: -r["expected_score"])
        return selected, {"selected": selected, "baseline": root.baseline,
                          "actions": records, "belief": "uniform_joint_counts",
                          "reason": "confirmed_values", "depth": self.config.depth,
                          **self._tree_settings(),
                          "candidate_pruning": (self.config.candidate_limit < 28 or
                                                self.config.effective_future_candidate_limit < 28),
                          "uncertainty": "Fixed-continuation sampling SE; excludes belief/model error. "
                                         "Not a simultaneous or post-selection confidence guarantee."}
