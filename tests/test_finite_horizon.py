"""Independent small-pool checks for the solitary finite-horizon solver.

The reference uses exact Fractions, Python win rules, no shanten pruning,
and no solver code. Permutation evaluation fixes a non-clairvoyant policy:
actions see only hand, remaining pool counts and remaining horizon, never the
unrevealed suffix of the enumerated permutation.

Run: .venv/bin/python -m unittest discover -s tests -p test_finite_horizon.py
"""
from fractions import Fraction
from functools import lru_cache
import itertools
import math
import unittest

from backend.analysis.finite_horizon import rank_discards, solve
from backend.rules.win import is_win


def counts(tiles):
    result = [0] * 28
    for tile in tiles:
        result[tile] += 1
    return tuple(result)


def adjust(hand, tile, delta):
    result = list(hand)
    result[tile] += delta
    return tuple(result)


@lru_cache(maxsize=None)
def reference_value(hand, unseen, k, discount=Fraction(1)):
    if k <= 0 or not sum(unseen):
        return Fraction(0)
    result = Fraction(0)
    for tile, weight in enumerate(unseen):
        if not weight:
            continue
        h14 = adjust(hand, tile, 1)
        next_unseen = adjust(unseen, tile, -1)
        if is_win(list(h14)):
            value = Fraction(1)
        else:
            value = discount * max(
                reference_value(adjust(h14, d, -1), next_unseen, k - 1, discount)
                for d, n in enumerate(h14) if n)
        result += Fraction(weight, sum(unseen)) * value
    return result


def reference_discard(hand, unseen, k):
    return max((d for d, n in enumerate(hand) if n),
               key=lambda d: (reference_value(adjust(hand, d, -1), unseen, k), -d))


def permutation_probability(hand, unseen, k):
    """Evaluate the independently derived policy over equally likely walls."""
    pool = [t for t, n in enumerate(unseen) for _ in range(n)]
    wins = total = 0
    for wall in itertools.permutations(pool):
        total += 1
        h, u = hand, unseen
        for step, tile in enumerate(wall[:k]):
            h = adjust(h, tile, 1)
            u = adjust(u, tile, -1)
            if is_win(list(h)):
                wins += 1
                break
            discard = reference_discard(h, u, k - step - 1)
            h = adjust(h, discard, -1)
    return Fraction(wins, total)


class FiniteHorizonTests(unittest.TestCase):
    def assert_exact(self, hand, unseen, k, expected):
        for backend in ("c", "python"):
            with self.subTest(backend=backend, k=k):
                result = solve(hand, unseen, k, backend=backend, cache_bits=10)
                self.assertTrue(result["exact"])
                self.assertEqual(result["cutoffs"], 0)
                self.assertEqual(result["lower_bound"], result["upper_bound"])
                self.assertAlmostEqual(result["probability"], float(expected), places=13)

    def test_zero_horizon_empty_pool_and_exhausted_pool(self):
        h = counts([0])
        self.assert_exact(h, counts([0, 8]), 0, 0)
        self.assert_exact(h, counts([]), 3, 0)
        self.assert_exact(h, counts([8]), 100, 0)
        self.assert_exact(h, counts([0]), 100, 1)

    def test_without_replacement_does_not_recycle_misses(self):
        # Two distinct targets among six cards are both required. Exact CDF is
        # choose(k, 2) / choose(6, 2), not a with-replacement approximation.
        hand = counts([0, 1, 8, 12, 13, 26, 26])
        unseen = counts([2, 14, 18, 19, 22, 23])
        for k in (1, 2, 3):
            self.assert_exact(hand, unseen, k, Fraction(math.comb(k, 2), 15))

    def test_optimizes_nonprogress_draw_in_four_meld_hand(self):
        # Starting with a 1条 singleton, draw 2条 from a pool of three 2条.
        # Keeping 2条 instead of treating this as a useless stay guarantees
        # completion on the second draw.
        self.assert_exact(counts([0]), counts([1, 1, 1]), 1, 0)
        self.assert_exact(counts([0]), counts([1, 1, 1]), 2, 1)

    def test_reference_recursion_and_all_pool_permutations(self):
        cases = [
            (counts([0, 1, 9, 9]), counts([2, 8, 8, 27]), 3),
            (counts([0, 2, 9, 18]), counts([1, 9, 18, 27]), 3),
            (counts([0]), counts([1, 1, 8, 8]), 3),
        ]
        for hand, unseen, k in cases:
            with self.subTest(hand=hand):
                expected = reference_value(hand, unseen, k)
                self.assertEqual(permutation_probability(hand, unseen, k), expected)
                self.assert_exact(hand, unseen, k, expected)

    def test_joker_held_and_joker_drawn(self):
        self.assert_exact(counts([27]), counts([0, 8, 18]), 1, 1)
        self.assert_exact(counts([0, 1, 9, 9]), counts([2, 27, 18]), 1, Fraction(2, 3))
        hand, unseen = counts([0, 1, 9, 27]), counts([2, 8, 18, 27])
        self.assert_exact(hand, unseen, 3, reference_value(hand, unseen, 3))

    def test_all_concealed_sizes_after_melds(self):
        blocks = ([9, 10, 11], [12, 13, 14], [18, 19, 20], [21, 22, 23])
        tiles = [0]
        for n in range(5):
            if n:
                tiles += blocks[n - 1]
            self.assert_exact(counts(tiles), counts([0, 8]), 1, Fraction(1, 2))

    def test_discard_ranking_matches_independent_values(self):
        hand = counts(list(range(9)) + [9, 10, 11, 18, 26])
        unseen = counts([18, 18, 26])
        for backend in ("c", "python"):
            rows = rank_discards(hand, unseen, 1, backend=backend, cache_bits=10)
            self.assertEqual(rows[0]["tile"], 26)
            self.assertAlmostEqual(rows[0]["probability"], 2 / 3)
            for row in rows:
                expected = reference_value(adjust(hand, row["tile"], -1), unseen, 1)
                self.assertTrue(row["exact"])
                self.assertAlmostEqual(row["probability"], float(expected))
        # Four exposed melds still require choosing which singleton to retain.
        rows = rank_discards(counts([0, 8]), counts([0, 8, 8]), 1, backend="c")
        self.assertEqual(rows[0]["tile"], 0)
        self.assertAlmostEqual(rows[0]["probability"], 2 / 3)

    def test_budget_returns_honest_bounds(self):
        hand = counts([0, 1, 8, 12, 13, 26, 26])
        unseen = counts([2, 14, 18, 19, 22, 23])
        expected = 0.2
        for backend in ("c", "python"):
            result = solve(hand, unseen, 3, max_nodes=1, backend=backend)
            self.assertFalse(result["exact"])
            self.assertLessEqual(result["nodes"], 1)
            self.assertGreater(result["cutoffs"], 0)
            self.assertLessEqual(result["lower_bound"], expected)
            self.assertGreaterEqual(result["upper_bound"], expected)
            self.assertEqual(result["probability"], result["lower_bound"])
        # Pruning an impossible deadline is exact even with a tiny budget.
        self.assertTrue(solve(hand, unseen, 1, max_nodes=1, backend="c")["exact"])

    def test_hash_collisions_do_not_change_values(self):
        hand = counts([0, 2, 9, 18])
        unseen = counts([1, 9, 18, 27])
        expected = reference_value(hand, unseen, 3)
        result = solve(hand, unseen, 3, backend="c", cache_bits=8)
        self.assertTrue(result["exact"])
        self.assertAlmostEqual(result["probability"], float(expected))

    def test_discount_one_preserves_probability_and_all_old_values(self):
        hand, unseen = counts([0, 2, 9, 18]), counts([1, 9, 18, 27])
        for backend in ("c", "python"):
            original = solve(hand, unseen, 3, backend=backend)
            explicit = solve(hand, unseen, 3, backend=backend, discount=1)
            self.assertEqual(original, explicit)
            self.assertEqual(original["value"], original["probability"])
            self.assertEqual(original["objective"], "completion_probability")

    def test_discounted_completion_time_matches_exact_distribution(self):
        # With the two-required-target pool, P(T=2)=1/15, P(T=3)=2/15.
        hand = counts([0, 1, 8, 12, 13, 26, 26])
        unseen = counts([2, 14, 18, 19, 22, 23])
        discount = Fraction(3, 4)
        expected = discount / 15 + 2 * discount**2 / 15
        for backend in ("c", "python"):
            result = solve(hand, unseen, 3, discount=discount, backend=backend)
            self.assertTrue(result["exact"])
            self.assertIsNone(result["probability"])
            self.assertEqual(result["objective"], "discounted_completion")
            self.assertAlmostEqual(result["value"], float(expected))
            self.assertEqual(result["value"], result["discounted_completion_value"])
            # Immediate completion is never discounted; forced second-draw
            # completion receives exactly one survival factor.
            first = solve(counts([27]), counts([0, 8]), 1,
                          discount=discount, backend=backend)
            second = solve(counts([0]), counts([1, 1, 1]), 2,
                           discount=discount, backend=backend)
            self.assertEqual(first["value"], 1.0)
            self.assertEqual(second["value"], float(discount))

    def test_discounted_bellman_and_rank_match_fraction_reference(self):
        hand = counts([0, 2, 9, 18, 27])
        unseen = counts([1, 9, 18, 27])
        for discount in (Fraction(1, 4), Fraction(3, 4)):
            expected = {
                d: reference_value(adjust(hand, d, -1), unseen, 3, discount)
                for d, n in enumerate(hand) if n}
            for backend in ("c", "python"):
                rows = rank_discards(hand, unseen, 3, discount=discount, backend=backend)
                self.assertEqual([r["tile"] for r in rows],
                                 sorted(expected, key=lambda d: (-expected[d], d)))
                for row in rows:
                    self.assertTrue(row["exact"])
                    self.assertIsNone(row["probability"])
                    self.assertAlmostEqual(row["value"], float(expected[row["tile"]]), places=13)

    def test_discounted_budget_bounds_are_not_reported_as_probabilities(self):
        hand = counts([0, 1, 8, 12, 13, 26, 26])
        unseen = counts([2, 14, 18, 19, 22, 23])
        discount = Fraction(1, 2)
        expected = discount / 15 + 2 * discount**2 / 15
        for backend in ("c", "python"):
            result = solve(hand, unseen, 3, discount=discount, max_nodes=1, backend=backend)
            self.assertFalse(result["exact"])
            self.assertIsNone(result["probability"])
            self.assertLessEqual(result["lower_bound"], float(expected))
            self.assertGreaterEqual(result["upper_bound"], float(expected))
            self.assertLessEqual(result["upper_bound"], float(discount))

    def test_invalid_discount(self):
        for discount in (0, -0.1, 1.01, float("nan"), float("inf"), True, None):
            with self.subTest(discount=discount), self.assertRaises(ValueError):
                solve(counts([0]), counts([0]), 1, discount=discount)

    def test_invalid_counts_are_rejected_instead_of_truncated(self):
        h, u = list(counts([0])), list(counts([0]))
        bad = u.copy(); bad[1] = 0.5
        with self.assertRaises(ValueError):
            solve(h, bad, 1)
        bad = u.copy(); bad[0] = 4
        with self.assertRaises(ValueError):
            solve(h, bad, 1)
        with self.assertRaises(ValueError):
            solve(h, u, -1)
        with self.assertRaises(ValueError):
            solve(h, u, 1.5)
        with self.assertRaises(ValueError):
            solve(h, u, 1, max_nodes=0)
        with self.assertRaises(ValueError):
            solve([0] * 28, u, 1)
        with self.assertRaises(ValueError):
            solve(h[:-1], u, 1)
        with self.assertRaises(ValueError):
            rank_discards(h, u, 1)


if __name__ == "__main__":
    unittest.main()
