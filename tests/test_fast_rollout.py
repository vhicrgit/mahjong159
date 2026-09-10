"""Complete-state parity and fresh-world timing for the C v31 continuation."""

import copy
import os
import random
import time
import unittest
from unittest.mock import patch

from backend.ai.search159.fast_rollout import rollout_fast, supports, warmup
from backend.ai.search159.simulator import (
    acting_seat, apply_action, base_action, clone, observe, rollout, sample_world,
)
from backend.game.engine import Game

try:  # unittest discovery and module execution both work without a tests package.
    from test_search_simulator import fixture, tile_counts
except ImportError:
    from tests.test_search_simulator import fixture, tile_counts


def rule_state(game):
    return {
        "players": [
            {"seat": p.seat, "hand": p.hand, "melds": p.melds,
             "discards": p.discards, "score_delta": p.score_delta}
            for p in game.players
        ],
        **{name: getattr(game, name) for name in (
            "wall", "turn", "phase", "last_discard", "last_discarder",
            "pending_actions", "gang_records", "winner", "win_tile", "win_kind",
            "huangzhuang", "n_159", "fan_159", "last_drawn", "last_action",
        )},
    }


def collect_snapshots(n=224, seed0=951000):
    """Actual middle-game states; each contributing baseline game is finished."""
    result = []
    seed = seed0
    while len(result) < n:
        game = Game(seed=seed, human_seat=-1)
        step = 0
        while game.phase != "game_over":
            seat = acting_seat(game)
            if step > 0 and step % 3 == 1 and len(result) < n:
                result.append(clone(game, keep_log=False))
            apply_action(game, base_action(game, seat), seat)
            step += 1
            if step >= 500:
                raise RuntimeError("baseline collection did not finish")
        seed += 1
    return result


class FastRolloutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        warmup()
        cls.snapshots = collect_snapshots()

    def assert_parity(self, snapshot, hero=None):
        hero = acting_seat(snapshot) if hero is None else hero
        if hero is None:
            hero = 0
        python_game = clone(snapshot, keep_log=False)
        fast_game = clone(snapshot, keep_log=False)
        expected = rollout(python_game, hero)
        actual = rollout_fast(fast_game, hero, mutate=True)
        self.assertEqual(actual.winner, expected.winner)
        self.assertEqual(actual.draw, expected.draw)
        self.assertEqual(actual.steps, expected.steps)
        self.assertEqual(actual.actual_scores, tuple(p.score_delta for p in python_game.players))
        for a, b in zip(actual.scores, expected.scores):
            self.assertAlmostEqual(a, b, places=12)
        self.assertEqual(actual.hero_score, actual.scores[hero])
        self.assertEqual(actual.wall_remaining, len(python_game.wall))
        self.assertEqual(actual.final_state, rule_state(python_game))
        self.assertEqual(rule_state(fast_game), rule_state(python_game))
        self.assertEqual(tile_counts(fast_game), [4] * 28)
        return actual

    def test_224_real_middle_game_states_all_seats_complete_state_parity(self):
        self.assertEqual(len(self.snapshots), 224)
        self.assertEqual({acting_seat(g) for g in self.snapshots}, {0, 1, 2, 3})
        self.assertEqual({g.phase for g in self.snapshots}, {"discard_wait", "react_wait"})
        for index, snapshot in enumerate(self.snapshots):
            with self.subTest(index=index, seat=acting_seat(snapshot), phase=snapshot.phase):
                self.assert_parity(snapshot)

    def test_new_uniform_hidden_worlds_match_python(self):
        rng = random.Random(78300)
        for index, snapshot in enumerate(self.snapshots[::7]):
            hero = acting_seat(snapshot)
            world = sample_world(observe(snapshot, hero), rng)
            with self.subTest(index=index):
                self.assert_parity(world, hero)

    def test_ming_an_bu_gang_and_tail_win_at_all_wall_boundaries(self):
        for wall_size in (0, 1, 6, 7, None):
            for kind in ("ming", "an", "bu"):
                kwargs = {"wall_size": wall_size}
                if wall_size != 0:
                    kwargs["tail"] = 20
                if kind == "ming":
                    game = fixture([0, 0, 0, 3, 4, 5, 9, 10, 11, 18, 19, 22, 22],
                                   claim=(1, 0), **kwargs)
                elif kind == "an":
                    game = fixture([0, 0, 0, 0, 3, 4, 5, 9, 10, 11, 18, 19, 22, 22], **kwargs)
                else:
                    game = fixture([0, 3, 4, 5, 9, 10, 11, 18, 19, 22, 22],
                                   melds=[{"type": "peng", "tile": 0, "wr": 50}], **kwargs)
                with self.subTest(kind=kind, wall_size=wall_size):
                    result = self.assert_parity(game, 0)
                    self.assertEqual(result.final_state["gang_records"][0]["kind"], kind)
                    if wall_size == 0:
                        self.assertTrue(result.draw)
                        self.assertEqual(result.scores, (0.0,) * 4)
                    else:
                        self.assertEqual(result.winner, 0)
                        self.assertEqual(result.final_state["win_kind"], "gangshang")
                        self.assertEqual(result.final_state["win_tile"], 20)
                        if wall_size is not None and wall_size <= 6:
                            self.assertEqual(result.final_state["n_159"], 0)
                            self.assertEqual(result.actual_scores, tuple(int(x) for x in result.scores))

    def test_exposed_fourth_meld_single_wait_and_red_wait(self):
        melds = [{"type": "peng", "tile": t, "wr": 45} for t in (9, 13, 18, 22)]
        for hand in ([0], [27]):
            game = fixture(hand, hero=2, melds=melds, claim=(1, 4), wall_size=10)
            # This is a discard by seat 1 with no focal response; finish the
            # existing response or normal transition as in the actual engine.
            if not game.pending_actions:
                game._next_draw()
            with self.subTest(hand=hand):
                self.assert_parity(game, 2)

    def test_live_input_unchanged_without_mutate_and_terminal_input_noop(self):
        game = clone(self.snapshots[8])
        before = copy.deepcopy(rule_state(game))
        result = rollout_fast(game, 3)
        self.assertEqual(rule_state(game), before)
        ended = clone(game)
        rollout_fast(ended, 3, mutate=True)
        again = rollout_fast(ended, 3, mutate=True)
        self.assertEqual(again.steps, 0)
        self.assertEqual(again.final_state, result.final_state)

    def test_environment_v31_weights_are_not_silently_hardcoded(self):
        overrides = {
            "V10_SHANTEN_W": "85.5", "V10_UKEIRE_W": "1.3", "V10_CONT_W": "0.17",
            "V10_RISK_W": "7.5", "V10_CONT_MAX_SH": "1",
        }
        with patch.dict(os.environ, overrides):
            for snapshot in self.snapshots[::14]:
                self.assert_parity(snapshot)
        with patch.dict(os.environ, {"V10_CONT_MAX_SH": "-1", "V10_CONT_W": "0.0"}):
            for snapshot in self.snapshots[::56]:
                self.assert_parity(snapshot)

    def test_policy_support_guard_and_invalid_physical_inputs(self):
        self.assertTrue(supports("v31"))
        self.assertTrue(supports(["v31", "v31n", "v31", "v31n"]))
        self.assertTrue(supports({2: "v31n"}))
        self.assertFalse(supports("hv"))
        self.assertFalse(supports(["v31", "hv", "v31", "v31"]))
        with self.assertRaises(ValueError):
            rollout_fast(self.snapshots[0], 0, policies="hv")
        with self.assertRaises(RuntimeError):
            rollout_fast(self.snapshots[0], 0, max_steps=0)
        bad = clone(self.snapshots[0])
        bad.wall.pop()
        with self.assertRaises(ValueError):
            rollout_fast(bad, 0)


def benchmark_fresh_worlds(n=128):
    """Balanced call order prevents giving one backend every warm-cache trial."""
    warmup()
    snapshots = collect_snapshots(32, seed0=968000)
    rng = random.Random(892834)
    py_cpu = fast_cpu = py_wall = fast_wall = 0.0
    n_steps = 0
    for i in range(n):
        snapshot = snapshots[i % len(snapshots)]
        hero = acting_seat(snapshot)
        world = sample_world(observe(snapshot, hero), rng)
        games = {"python": clone(world, keep_log=False), "c": clone(world, keep_log=False)}
        order = ("python", "c") if i % 2 else ("c", "python")
        results = {}
        for kind in order:
            c0, t0 = time.process_time(), time.perf_counter()
            if kind == "python":
                results[kind] = rollout(games[kind], hero)
                py_cpu += time.process_time() - c0
                py_wall += time.perf_counter() - t0
            else:
                results[kind] = rollout_fast(games[kind], hero, mutate=True)
                fast_cpu += time.process_time() - c0
                fast_wall += time.perf_counter() - t0
        if rule_state(games["python"]) != rule_state(games["c"]):
            raise AssertionError(f"benchmark parity failed for fresh world {i}")
        n_steps += results["python"].steps
    return {
        "worlds": n, "average_remaining_actions": n_steps / n,
        "python_cpu_seconds": py_cpu, "c_cpu_seconds": fast_cpu,
        "python_wall_seconds": py_wall, "c_wall_seconds": fast_wall,
        "cpu_speedup": py_cpu / fast_cpu, "wall_speedup": py_wall / fast_wall,
        "c_ms_per_world": 1000 * fast_wall / n,
        "note": "New shared hidden worlds; balanced backend order; clone/sample/compile outside timing, C serialization and writeback included.",
    }


if __name__ == "__main__":
    unittest.main()
