"""Behavioral information-boundary and held-out-value checks for search159.

The tiny hidden-world game has a known attainable value: without a public
signal a fixed guess earns zero, while observing the hidden bit permits +1.
It detects clairvoyant per-world continuation choices without reproducing the
planner's search equations. Real-game tests exercise the public Bot boundary.
"""
from contextlib import contextmanager
import copy
from dataclasses import dataclass, replace
import math
import random
import unittest
from unittest.mock import patch

from backend.ai.bot_search159 import Bot
from backend.ai.search159 import planner as planning
from backend.ai.search159 import simulator as sim
from backend.game.engine import Game


@dataclass(frozen=True)
class TinyObservation:
    hero: int = 0
    stage: int = 0
    hand_counts: tuple = (1, 1) + (0,) * 26
    melds: tuple = ((), (), (), ())
    gang_records: tuple = ()
    own_legal_actions: tuple = (sim.Action("discard", 0), sim.Action("discard", 1))
    phase: str = "discard_wait"
    turn: int = 0
    last_discard: int | None = None
    last_discarder: int | None = None
    wall_length: int = 20


@dataclass
class TinyWorld:
    hidden: int
    stage: int = 0
    phase: str = "discard_wait"
    winner: int | None = None
    huangzhuang: bool = False
    score: float = 0.0


@contextmanager
def hidden_guess_game(*, public_signal=False):
    """Root action is irrelevant; next action must guess an unobserved bit."""
    counter = [0]
    actions = (sim.Action("discard", 0), sim.Action("discard", 1))

    def sample(_observation, _rng):
        counter[0] += 1
        return TinyWorld(hidden=counter[0] % 2)

    def candidate_actions(_game, _hero, _config):
        return actions, actions[0]

    def apply(game, action, _hero):
        if game.stage == 0:
            game.stage = 1
        else:
            game.stage = 2
            game.score = 1.0 if action.tile == game.hidden else -1.0
            game.winner = 0 if game.score > 0 else 1
            game.phase = "game_over"

    def advance(game, _hero, _policies):
        if game.phase == "game_over":
            return ()
        # Final observation is identical; only the intervening PUBLIC trace
        # differs in the signalled version of the game.
        return ((1, "discard", 7 + game.hidden if public_signal else 7),)

    def scores(game):
        return (game.score, -game.score, 0.0, 0.0)

    def rollout(game, hero, _policies):
        apply(game, actions[0], hero)
        return sim.RolloutResult(scores(game), game.score, game.winner, False, 1)

    with patch.object(planning, "candidates", candidate_actions), patch.multiple(
            sim, sample_world=sample, clone=copy.deepcopy, apply_action=apply,
            advance=advance, observe=lambda game, _hero: TinyObservation(stage=game.stage),
            expected_scores=scores, rollout=rollout):
        yield


class SearchPlannerTests(unittest.TestCase):
    def test_identical_observations_share_continuation_without_clairvoyance(self):
        planner = planning.Planner(planning.SearchConfig(
            simulations=64, confirmation=32, depth=2, candidate_limit=2), seed=7)
        with hidden_guess_game(public_signal=False):
            _selected, report = planner.plan(TinyObservation())
        for record in report["actions"]:
            # Both equally likely hidden worlds enter one information set.
            # No frozen policy at that information set can exceed value zero.
            self.assertEqual(record["expected_score"], 0.0)
        for edge in planner.last_tree.edges.values():
            self.assertEqual(len(edge.children), 1)
            child = next(iter(edge.children.values()))
            self.assertEqual(edge.visits, 64)
            self.assertEqual(child.visits, 64)
            self.assertEqual(sum(e.visits for e in child.edges.values()), 64)
            # Confirmation must not train the tree or alter its continuation.
            self.assertEqual(sum(e.visits for e in child.edges.values()), edge.visits)

    def test_public_history_can_legitimately_separate_identical_final_observations(self):
        planner = planning.Planner(planning.SearchConfig(
            simulations=64, confirmation=32, depth=2, candidate_limit=2), seed=7)
        with hidden_guess_game(public_signal=True):
            _selected, report = planner.plan(TinyObservation())
        for record in report["actions"]:
            self.assertEqual(record["expected_score"], 1.0)
        for edge in planner.last_tree.edges.values():
            self.assertEqual(len(edge.children), 2)

    def test_public_hand_abstraction_cannot_exploit_discarded_history_signal(self):
        planner = planning.Planner(planning.SearchConfig(
            simulations=64, confirmation=32, depth=2, candidate_limit=2,
            tree_key="public_hand", paired_future=True), seed=7)
        with hidden_guess_game(public_signal=True):
            _selected, report = planner.plan(TinyObservation())
        # History mode can use this public signal (the test above earns +1).
        # The abstraction deliberately forgets it and must choose consistently
        # across both worlds, even though both candidate payoffs are evaluated.
        for record in report["actions"]:
            self.assertEqual(record["expected_score"], 0.0)
        for edge in planner.last_tree.edges.values():
            self.assertEqual(len(edge.children), 1)
            child = next(iter(edge.children.values()))
            self.assertEqual(edge.visits, 64)
            self.assertEqual([e.visits for e in child.edges.values()], [64, 64])
        self.assertEqual(report["tree_key"], "public_hand")
        self.assertTrue(report["paired_future"])

    def test_public_hand_key_retains_actions_and_public_turn_payoff_context(self):
        obs = TinyObservation()
        actions = obs.own_legal_actions
        key = planning.public_hand_key(obs, actions)
        self.assertNotEqual(key, planning.public_hand_key(obs, actions[::-1]))
        for change in (
            {"hero": 1}, {"hand_counts": (0, 2) + (0,) * 26},
            {"melds": ((sim.Meld("peng", 2),), (), (), ())},
            {"gang_records": (sim.GangRecord(0, "ming", 2, 1),)},
            {"own_legal_actions": (actions[0],)}, {"phase": "react_wait"},
            {"turn": 1}, {"last_discard": 3}, {"last_discarder": 2},
            {"wall_length": 19},
        ):
            with self.subTest(change=change):
                self.assertNotEqual(key, planning.public_hand_key(replace(obs, **change), actions))

    def test_paired_training_returns_preselected_action_not_world_maximum(self):
        planner = planning.Planner(planning.SearchConfig(
            simulations=1, confirmation=2, depth=2, candidate_limit=2,
            tree_key="public_hand", paired_future=True, min_node_visits=1), seed=7)
        with hidden_guess_game(public_signal=False):
            with patch.object(sim, "sample_world", side_effect=lambda *_: TinyWorld(hidden=1)):
                _selected, report = planner.plan(TinyObservation())
        for record in report["actions"]:
            # Before seeing the first world's payoffs, the policy chooses 0
            # and loses. The unchosen guess 1 wins but must not replace that
            # root training return. It may legitimately improve the later,
            # frozen continuation in this constant-hidden-bit distribution.
            self.assertEqual(record["train_mean"], -1.0)
            self.assertEqual(record["expected_score"], 1.0)
        for edge in planner.last_tree.edges.values():
            child = next(iter(edge.children.values()))
            self.assertEqual([e.visits for e in child.edges.values()], [1, 1])
            self.assertEqual([e.mean for e in child.edges.values()], [-1.0, 1.0])
        # Two roots × two training guesses, then two confirmation worlds ×
        # two roots × one frozen guess. Confirmation must not pair or train.
        self.assertEqual(planner.stats["leaf_rollouts"], 8)
        self.assertEqual(planner.stats["paired_evaluations"], 4)
        self.assertEqual(planner.stats["simulations"], 6)

    def test_default_candidate_budget_is_unchanged_until_feature_enabled(self):
        default = planning.SearchConfig(candidate_limit=4)
        self.assertEqual(default.effective_future_candidate_limit, 4)
        self.assertEqual(replace(default, tree_key="public_hand").effective_future_candidate_limit, 2)
        self.assertEqual(replace(default, paired_future=True).effective_future_candidate_limit, 2)

    def test_invalid_config_fails_before_search_or_finite_solver_runs(self):
        invalid = {
            "finite_horizon": (-1, 0, 2.5, True, float("nan")),
            "finite_max_nodes": (-1, 0, 2.5, True, 1 << 64),
            "finite_discount": (True, 0.0, 1.1, float("nan"), float("inf")),
            "simulations": (True, 0, 1.5, float("inf")),
            "depth": (False, float("nan")),
            "candidate_limit": (1.5, float("inf")),
            "future_candidate_limit": (0, True),
            "confirmation": (1, 2.5, True),
            "min_node_visits": (True, float("nan")),
            "exploration": (-1.0, True, float("nan"), float("inf")),
            "confidence_z": (-1.0, True, float("nan"), float("inf")),
            "min_gain": (-1.0, True, float("nan"), float("inf")),
            "max_search_shanten": (True, float("nan"), 1.5),
        }
        for name, values in invalid.items():
            for value in values:
                with self.subTest(parameter=name, value=value), self.assertRaises(ValueError):
                    planning.SearchConfig(**{name: value})
        for budget in (None, 1, (1 << 64) - 1):
            self.assertEqual(planning.SearchConfig(finite_max_nodes=budget).finite_max_nodes, budget)

    def test_held_out_values_and_paired_standard_error(self):
        config = planning.SearchConfig(simulations=4, confirmation=4, depth=2,
                                       candidate_limit=2)
        planner = planning.Planner(config, seed=7)
        actions = (sim.Action("discard", 0), sim.Action("discard", 1))
        samples = [0]

        def sample(_obs, _rng):
            index = samples[0]
            samples[0] += 1
            # Candidate construction + training see a misleading distribution.
            hidden = 2 if index <= config.simulations else (index - 5) % 2
            return TinyWorld(hidden)

        def apply(game, action, _hero):
            payoff = {2: (10.0, -10.0), 0: (2.0, 3.0), 1: (-2.0, 2.0)}
            game.score = payoff[game.hidden][action.tile]
            game.phase = "game_over"
            game.winner = 0 if game.score > 0 else 1

        with patch.object(planning, "candidates", lambda *_: (actions, actions[0])), patch.multiple(
                sim, sample_world=sample, clone=copy.deepcopy, apply_action=apply,
                advance=lambda *_: (),
                expected_scores=lambda g: (g.score, -g.score, 0.0, 0.0)):
            selected, report = planner.plan(TinyObservation())
        records = {r["action"]: r for r in report["actions"]}
        self.assertGreater(records[actions[0]]["train_mean"], records[actions[1]]["train_mean"])
        self.assertEqual(selected, actions[1])
        self.assertEqual(records[actions[0]]["expected_score"], 0.0)
        self.assertEqual(records[actions[1]]["expected_score"], 2.5)
        self.assertEqual(records[actions[1]]["delta_vs_baseline"], 2.5)
        # Paired differences [1,4,1,4] have mean 2.5 and SE sqrt(3/4).
        self.assertAlmostEqual(records[actions[1]]["paired_se"], math.sqrt(0.75))

    def test_real_hidden_reallocation_does_not_change_plan(self):
        source = Game(seed=1592026, human_seat=-1)
        self.assertEqual(source.phase, "discard_wait")
        other = sim.clone(source)
        hidden = list(other.wall)
        for player in other.players[1:]:
            hidden.extend(player.hand)
        random.Random(9381).shuffle(hidden)
        position = 0
        for player in other.players[1:]:
            count = len(player.hand)
            player.hand = sorted(hidden[position:position + count])
            position += count
        other.wall = hidden[position:]
        self.assertEqual(sim.observe(source, 0), sim.observe(other, 0))
        bots = [Bot(g, 0, mode="root", simulations=1, confirmation=2,
                    candidate_limit=2, seed=99) for g in (source, other)]
        self.assertEqual(bots[0].choose_action(), bots[1].choose_action())
        self.assertEqual(bots[0].last_analysis, bots[1].last_analysis)
        config = planning.SearchConfig(simulations=2, confirmation=2, depth=2,
                                        candidate_limit=2, tree_key="public_hand",
                                        paired_future=True)
        plans = [planning.Planner(config, seed=99).plan(sim.observe(g, 0))
                 for g in (source, other)]
        self.assertEqual(plans[0], plans[1])

    def test_finite_mode_rejects_unresolved_bounds_and_caches_same_observation(self):
        game = Game(seed=1592026, human_seat=-1)
        bot = Bot(game, 0, mode="finite", finite_horizon=2)
        baseline = sim.base_action(game, 0)
        self.assertEqual(baseline.kind, "discard")
        alternative = next(a for a in sim.legal_actions(game, 0)
                           if a.kind == "discard" and a != baseline)
        rows = [
            {"tile": baseline.tile, "probability": 0.4, "value": 0.4, "lower_bound": 0.4,
             "upper_bound": 0.4, "exact": True, "nodes": 1},
            {"tile": alternative.tile, "probability": 0.3, "value": 0.3, "lower_bound": 0.3,
             "upper_bound": 0.9, "exact": False, "nodes": 1},
        ]
        with patch("backend.analysis.finite_horizon.rank_discards", return_value=rows) as rank:
            self.assertEqual(bot.choose_action(), baseline)
            self.assertEqual(bot.choose_action(), baseline)
            self.assertEqual(bot.choose_discard(), baseline.tile)
            self.assertEqual(rank.call_count, 1)
        self.assertEqual(bot.last_analysis["reason"], "unresolved_finite_bounds")


if __name__ == "__main__":
    unittest.main()
