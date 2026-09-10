"""Evaluation pipeline checks: settlement, pairing, reproducibility and API use."""

from dataclasses import replace
import unittest
from unittest.mock import patch

from tools.eval_search159 import EvalConfig, _choose_action, evaluate, play_game


def outcomes(report):
    return [(row["seed"], [(s["hero"], [(s[arm]["scores"], s[arm]["winner"],
             s[arm]["all_actions"], s[arm]["hero_actions"], s[arm]["search_seed"])
             for arm in ("a", "b")]) for s in row["seats"]]) for row in report["results"]]


class EvaluationPipelineTests(unittest.TestCase):
    def test_identical_arms_pair_all_seats_and_settle_real_zero_sum_scores(self):
        report = evaluate(EvalConfig(seeds=2, seed0=5100000))
        self.assertEqual(report["executed_games"], 16)
        self.assertEqual(report["paired_games"], 8)
        self.assertEqual(report["b_minus_a"]["score"]["ci95"], [0.0, 0.0])
        for row in report["results"]:
            self.assertEqual([s["hero"] for s in row["seats"]], list(range(4)))
            for seat in row["seats"]:
                a, b = seat["a"], seat["b"]
                self.assertEqual(a["scores"], b["scores"])
                self.assertEqual(a["all_actions"], b["all_actions"])
                self.assertEqual(sum(a["scores"]), 0)
                self.assertEqual(a["score"], a["scores"][seat["hero"]])
                self.assertEqual(a["search_seed"], b["search_seed"])

    def test_worker_count_does_not_change_results(self):
        config = EvalConfig(seeds=2, seed0=5100007)
        self.assertEqual(outcomes(evaluate(config)), outcomes(evaluate(replace(config, workers=2))))

    def test_unified_action_called_once_per_event(self):
        from backend.ai.search159.simulator import base_action
        calls = []

        class UnifiedBot:
            def __init__(self, game, seat):
                self.game, self.seat = game, seat

            def choose_action(self):
                calls.append(self.seat)
                return base_action(self.game, self.seat, policy="v31")

            def choose_discard(self):
                raise AssertionError("Unified policy must not use legacy discard wrapper")

            def decide_gang(self, *_):
                raise AssertionError("Unified policy must not use legacy gang wrapper")

        with patch("tools.eval_search159._make_bot", side_effect=lambda kind, g, s, *args: UnifiedBot(g, s)):
            result = play_game(5100011, 2, "v31", EvalConfig())
        self.assertEqual(len(calls), result["steps"])
        self.assertEqual(calls.count(2), result["hero_decisions"])

    def test_native_scholar_matches_python_on_real_closed_open_and_call_states(self):
        from backend.ai.bot_hv import Bot as PythonScholar
        from backend.game.engine import Game
        from backend.ai.search159.simulator import acting_seat, apply_action, base_action, clone

        cases = {}
        for seed in range(5100000, 5100010):
            game = Game(seed=seed, human_seat=-1)
            while game.phase != "game_over":
                seat = acting_seat(game)
                label = ("reaction" if game.phase == "react_wait" else
                         "self_gang" if game._gang_options(seat) else
                         "open_discard" if game.players[seat].melds else "closed_discard")
                if label not in cases:
                    cases[label] = (clone(game), seat)
                apply_action(game, base_action(game, seat, "v31"))
            if len(cases) == 4:
                break
        self.assertEqual(set(cases), {"reaction", "self_gang", "open_discard", "closed_discard"})
        for label, (game, seat) in cases.items():
            with self.subTest(phase=label):
                expected = _choose_action(game, seat, PythonScholar(game, seat))
                self.assertEqual(base_action(game, seat, "hv"), expected)


class EvaluationMetadataTests(unittest.TestCase):
    @staticmethod
    def fingerprint(head="a" * 40):
        return {"git_head": head, "source_sha256": {"planner.py": "a" * 64},
                "policy_environment": {"V10_CONT_W": None},
                "git_revision_check": {"status": "verified" if head else "unavailable",
                                       "error": None}}

    def metadata_report(self, before, after):
        # Exercise evaluate's metadata path without running or modifying games.
        with patch("tools.eval_search159.source_fingerprint", side_effect=[before, after]), \
                patch("tools.eval_search159.play_seed", return_value={}), \
                patch("tools.eval_search159.summarize", return_value={}):
            return evaluate(EvalConfig(seeds=1))

    def test_missing_git_is_incomplete_not_a_source_change(self):
        for old, new in (("a" * 40, None), (None, "a" * 40), (None, None)):
            with self.subTest(before=old, after=new):
                before, after = self.fingerprint(old), self.fingerprint(new)
                report = self.metadata_report(before, after)
                self.assertFalse(report["source_changed_during_run"])
                self.assertFalse(report["git_revision_verification_complete"])
                self.assertIsNone(report["source_change_details"]["git_revision_changed"])
                self.assertEqual(report["source_version_after_run"], after)

    def test_actual_source_hash_change_is_reported_even_if_git_is_missing(self):
        before, after = self.fingerprint(), self.fingerprint(None)
        after["source_sha256"]["planner.py"] = "b" * 64
        report = self.metadata_report(before, after)
        self.assertTrue(report["source_changed_during_run"])
        self.assertTrue(report["source_change_details"]["source_sha256_changed"])
        self.assertFalse(report["git_revision_verification_complete"])

    def test_policy_environment_change_is_reported(self):
        before, after = self.fingerprint(), self.fingerprint()
        after["policy_environment"]["V10_CONT_W"] = "0.75"
        report = self.metadata_report(before, after)
        self.assertTrue(report["source_changed_during_run"])
        self.assertTrue(report["source_change_details"]["policy_environment_changed"])
        self.assertTrue(report["git_revision_verification_complete"])

    def test_two_different_verified_revisions_are_a_change(self):
        before, after = self.fingerprint("a" * 40), self.fingerprint("b" * 40)
        report = self.metadata_report(before, after)
        self.assertTrue(report["source_changed_during_run"])
        self.assertTrue(report["git_revision_verification_complete"])
        self.assertTrue(report["source_change_details"]["git_revision_changed"])

    def test_unchanged_verified_revision_remains_verified(self):
        before, after = self.fingerprint("a" * 64), self.fingerprint("a" * 64)
        report = self.metadata_report(before, after)
        self.assertFalse(report["source_changed_during_run"])
        self.assertTrue(report["git_revision_verification_complete"])
        self.assertFalse(report["source_change_details"]["git_revision_changed"])
        self.assertEqual(report["source_version_after_run"], after)

    def test_git_timeout_keeps_an_explicit_diagnostic(self):
        import subprocess
        from tools.eval_search159 import source_fingerprint
        failure = subprocess.TimeoutExpired(["git", "rev-parse", "--verify", "HEAD"], 5)
        with patch("tools.eval_search159.subprocess.run", side_effect=failure):
            result = source_fingerprint()
        self.assertIsNone(result["git_head"])
        self.assertEqual(result["git_revision_check"]["status"], "unavailable")
        self.assertEqual(result["git_revision_check"]["error"]["type"], "TimeoutExpired")
        self.assertEqual(result["git_revision_check"]["error"]["timeout_seconds"], 5)


if __name__ == "__main__":
    unittest.main()
