"""Production-engine, information-boundary and conservation tests (unittest)."""

import copy
import random
import unittest
from dataclasses import FrozenInstanceError, replace

from backend.ai.search159.simulator import (
    Action, acting_seat, advance, apply_action, base_action, clone,
    expected_scores, legal_actions, observe, rollout, sample_world,
)
from backend.game.engine import Game
from backend.rules.tiles import build_wall


def state(game):
    """All rule-relevant engine state, including actual scoring and draw metadata."""
    return {
        "wall": game.wall,
        "players": [(p.hand, p.melds, p.discards, p.score_delta) for p in game.players],
        **{name: getattr(game, name) for name in (
            "turn", "phase", "last_discard", "last_discarder", "pending_actions",
            "winner", "win_tile", "win_kind", "fan_159", "n_159", "huangzhuang",
            "gang_records", "last_drawn", "last_action", "log",
        )},
    }


def tile_counts(game):
    counts = [0] * 28
    for tile in game.wall:
        counts[tile] += 1
    for player in game.players:
        for tile in player.hand + player.discards:
            counts[tile] += 1
        for meld in player.melds:
            counts[meld["tile"]] += 3 if meld["type"] == "peng" else 4
    return counts


def fixture(hand, *, hero=0, melds=(), claim=None, wall_size=None, tail=None):
    """Make a tile-conserving decision state with a chosen focal hand.

    Unspecified tiles are allocated once to public discards, opponent hands and
    wall, so these fixtures also exercise the sampler's physical constraints.
    """
    game = Game(seed=90210, human_seat=-1)
    for player in game.players:
        player.hand = []
        player.discards = []
        player.melds = []
        player.score_delta = 0
    game.players[hero].hand = sorted(hand)
    game.players[hero].melds = copy.deepcopy(list(melds))
    pool = build_wall()
    for tile in hand:
        pool.remove(tile)
    for meld in melds:
        for _ in range(3 if meld["type"] == "peng" else 4):
            pool.remove(meld["tile"])
    if claim is not None:
        discarder, tile = claim
        pool.remove(tile)
        game.players[discarder].discards.append(tile)
    if tail is not None:
        pool.remove(tail)
    random.Random(555).shuffle(pool)
    if wall_size is not None:
        excess = len(pool) + (tail is not None) - 39 - wall_size
        if excess < 0:
            raise ValueError("fixture has too few tiles for its requested wall")
        discards = pool[:excess]
        del pool[:excess]
        for i, tile in enumerate(discards):
            game.players[i % 4].discards.insert(0, tile)
    for seat in range(4):
        if seat != hero:
            game.players[seat].hand = sorted(pool[:13])
            del pool[:13]
    game.wall = pool + ([] if tail is None else [tail])
    game.turn = hero
    game.phase = "discard_wait"
    game.last_discard = None
    game.last_discarder = None
    game.pending_actions = {}
    game.winner = None
    game.win_tile = None
    game.win_kind = None
    game.fan_159 = []
    game.n_159 = 0
    game.huangzhuang = False
    game.gang_records = []
    game.log = []
    game.last_action = ""
    game.last_drawn = None
    if claim is not None:
        discarder, tile = claim
        game.turn = discarder
        game.phase = "react_wait"
        game.last_discard = tile
        game.last_discarder = discarder
        for seat, player in enumerate(game.players):
            count = player.hand.count(tile)
            if seat != discarder and count >= 2:
                game.pending_actions[seat] = {"peng": True, "gang": count >= 3}
    assert tile_counts(game) == [4] * 28
    return game


class SimplePolicy:
    """Deterministic inexpensive policy for rules-only tests, not strength tests."""

    def __init__(self, game, seat):
        self.game, self.seat = game, seat

    def choose_discard(self):
        return self.game.players[self.seat].hand[0]

    def decide_peng(self, _tile):
        return True

    def decide_gang(self, _tile, _kind):
        return True


class NoCalls(SimplePolicy):
    def decide_peng(self, _tile):
        return False

    def decide_gang(self, _tile, _kind):
        return False


class SearchSimulatorTests(unittest.TestCase):
    def test_observation_is_immutable_and_contains_no_other_hidden_state(self):
        source = Game(seed=1001, human_seat=-1)
        # An opponent has just drawn; its tile is not part of our information.
        source.last_drawn = {"seat": 1, "tile": source.players[1].hand[-1]}
        hidden = clone(source)
        hidden.wall.reverse()
        hidden.players[1].hand, hidden.players[2].hand = hidden.players[2].hand, hidden.players[1].hand
        hidden.last_drawn = {"seat": 1, "tile": (source.last_drawn["tile"] + 1) % 28}
        hidden.log = ["secret hands and future wall"]
        hidden.last_action = "secret drawn tile"
        a, b = observe(source, 0), observe(hidden, 0)
        self.assertEqual(a, b)
        self.assertEqual(a.key(), b.key())
        self.assertEqual(hash(a), hash(b))
        self.assertIsNone(a.own_drawn_tile)
        with self.assertRaises(FrozenInstanceError):
            a.turn = 1
        wa = sample_world(a, random.Random(222))
        wb = sample_world(b, random.Random(222))
        self.assertEqual(state(wa), state(wb))
        self.assertEqual(wa.log, [])
        self.assertEqual(wa.last_action, "")
        self.assertIsNone(wa.last_drawn)

    def test_observation_ignores_other_pending_legality_and_order(self):
        hand = [0, 0, 3, 4, 5, 9, 10, 11, 18, 19, 22, 22, 26]
        source = fixture(hand, hero=2, claim=(3, 0))
        altered = clone(source)
        # Poisoning a private field must not become a search input. Multiple
        # claimants are impossible in a legal four-copy game, but the boundary
        # should not accidentally expose them even under this adversarial test.
        altered.pending_actions = {0: {"peng": True, "gang": True},
                                   2: {"peng": True, "gang": False}}
        self.assertEqual(observe(source, 2), observe(altered, 2))
        sampled = sample_world(observe(altered, 2), random.Random(7))
        self.assertEqual(acting_seat(sampled), 2)
        self.assertEqual(legal_actions(sampled, 2), observe(source, 2).own_legal_actions)

    def test_sample_world_conserves_all_tiles_and_hand_sizes_for_each_seat(self):
        game = Game(seed=730, human_seat=-1)
        for hero in range(4):
            observation = observe(game, hero)
            for seed in range(10):
                with self.subTest(hero=hero, seed=seed):
                    sampled = sample_world(observation, random.Random(seed))
                    self.assertEqual(tile_counts(sampled), [4] * 28)
                    self.assertEqual(tuple(len(p.hand) for p in sampled.players), observation.hand_sizes)
                    self.assertEqual(sampled.players[hero].hand_counts, list(observation.hand_counts))
                    self.assertEqual(len(sampled.wall), observation.wall_length)
                    self.assertEqual(observe(sampled, hero), observation)

    def test_sample_world_handles_exposed_gang_and_reaction(self):
        hand = [0, 0, 3, 4, 5, 9, 10, 11, 22, 22]
        game = fixture(hand, hero=1,
                       melds=[{"type": "gang", "tile": 26, "kind": "an", "wr": 48}],
                       claim=(3, 0))
        observation = observe(game, 1, passed_seats=(0,))
        for seed in range(20):
            sampled = sample_world(observation, random.Random(seed))
            self.assertEqual(tile_counts(sampled), [4] * 28)
            self.assertNotIn(0, sampled.pending_actions)
            self.assertEqual(acting_seat(sampled), 1)
            self.assertEqual(legal_actions(sampled, 1), observation.own_legal_actions)
            self.assertEqual(sampled.players[1].melds, game.players[1].melds)

    def test_bad_observation_is_rejected_instead_of_clipping_tile_counts(self):
        observation = observe(Game(seed=44, human_seat=-1), 0)
        with self.assertRaises(ValueError):
            sample_world(replace(observation, wall_length=observation.wall_length + 1), random.Random(0))
        bad = list(observation.hand_counts)
        bad[0] = 5
        with self.assertRaises(ValueError):
            sample_world(replace(observation, hand_counts=tuple(bad)), random.Random(0))

    def test_clone_is_independent_and_rule_identical(self):
        game = fixture([0, 3, 4, 5, 9, 10, 11, 18, 19, 22, 22],
                       melds=[{"type": "peng", "tile": 0, "wr": 50}], tail=20)
        game.gang_records = [{"seat": 1, "kind": "ming", "tile": 26, "from": 2}]
        game.last_drawn = {"seat": 0, "tile": 0}
        copied = clone(game)
        self.assertEqual(state(copied), state(game))
        copied.players[0].melds[0]["type"] = "gang"
        copied.players[0].hand.pop()
        copied.wall.pop()
        copied.gang_records[0]["seat"] = 3
        copied.last_drawn["tile"] = 1
        copied.log.append("copy only")
        self.assertEqual(game.players[0].melds[0]["type"], "peng")
        self.assertEqual(game.gang_records[0]["seat"], 1)
        self.assertEqual(game.last_drawn["tile"], 0)
        self.assertNotEqual(len(copied.wall), len(game.wall))
        self.assertEqual(game.log, [])
        self.assertIsNot(copied.rng, game.rng)

    def test_all_three_gangs_match_engine_and_draw_from_tail(self):
        fixtures = [
            ("an", fixture([0, 0, 0, 0, 3, 4, 5, 9, 10, 11, 18, 19, 22, 22], tail=20)),
            ("bu", fixture([0, 3, 4, 5, 9, 10, 11, 18, 19, 22, 22],
                           melds=[{"type": "peng", "tile": 0, "wr": 50}], tail=20)),
            ("ming", fixture([0, 0, 0, 3, 4, 5, 9, 10, 11, 18, 19, 22, 22],
                             claim=(1, 0), tail=20)),
        ]
        for kind, game in fixtures:
            with self.subTest(kind=kind):
                reference = copy.deepcopy(game)
                action = Action("gang", 0)
                self.assertIn(action, legal_actions(game, 0))
                self.assertEqual(base_action(game, 0, SimplePolicy), action)
                apply_action(game, action)
                reference.action_gang(0, None if kind == "ming" else 0)
                self.assertEqual(state(game), state(reference))
                self.assertEqual(game.gang_records[-1]["kind"], kind)
                self.assertEqual(game.last_drawn, {"seat": 0, "tile": 20})
                self.assertEqual(game.winner, 0)
                self.assertEqual(game.win_kind, "gangshang")
                self.assertEqual(sum(p.score_delta for p in game.players), 0)
                self.assertEqual(tile_counts(game), [4] * 28)

    def test_peng_requires_discard_and_does_not_consume_wall(self):
        game = fixture([0, 0, 3, 4, 5, 9, 10, 11, 18, 19, 22, 22, 26], claim=(1, 0))
        reference = copy.deepcopy(game)
        old_wall = game.wall[:]
        apply_action(game, Action("peng", 0))
        reference.action_peng(0)
        self.assertEqual(state(game), state(reference))
        self.assertEqual(game.wall, old_wall)
        self.assertEqual(game.phase, "discard_wait")
        self.assertEqual(game.turn, 0)
        self.assertEqual(len(game.players[0].hand), 11)
        self.assertEqual(game.players[1].discards.count(0), 0)
        self.assertEqual(tile_counts(game), [4] * 28)

    def test_normal_draw_stops_at_six_but_gang_still_draws(self):
        # An ordinary discard of red cannot create a reaction and must trigger
        # the normal <=6-tile exhaustion rule, even with an unpaid gang ledger.
        normal = fixture([0, 0, 0, 0, 3, 4, 5, 9, 10, 11, 18, 19, 22, 27], wall_size=6)
        normal.gang_records = [{"seat": 0, "kind": "an", "tile": 0}]
        old_wall = normal.wall[:]
        apply_action(normal, Action("discard", 27))
        self.assertEqual(normal.wall, old_wall)
        self.assertTrue(normal.huangzhuang)
        self.assertEqual(expected_scores(normal), (0.0,) * 4)
        self.assertEqual([p.score_delta for p in normal.players], [0] * 4)
        gang = fixture([0, 0, 0, 0, 3, 4, 5, 9, 10, 11, 18, 19, 22, 22], wall_size=6, tail=20)
        apply_action(gang, Action("gang", 0))
        self.assertEqual(len(gang.wall), 5)
        self.assertEqual(gang.winner, 0)
        self.assertEqual(gang.n_159, 0)
        self.assertEqual(gang.fan_159, [])
        self.assertEqual(expected_scores(gang), (6.0, -2.0, -2.0, -2.0))
        self.assertEqual(tuple(p.score_delta for p in gang.players), expected_scores(gang))

    def test_normal_draw_uses_front_of_wall(self):
        game = fixture([0, 0, 0, 0, 3, 4, 5, 9, 10, 11, 18, 19, 22, 27])
        first, previous = game.wall[0], game.wall[:]
        apply_action(game, Action("discard", 27))
        self.assertEqual(game.wall, previous[1:])
        self.assertEqual(game.last_drawn, {"seat": 1, "tile": first})

    def test_expected_scores_preserve_conditional_settlement_and_no_step_cost(self):
        game = Game(seed=19, human_seat=-1)
        with self.assertRaises(ValueError):
            expected_scores(game)
        game.phase, game.winner = "game_over", 2
        game.wall = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # four 1/5/9 tiles.
        game.gang_records = [{"seat": 0, "kind": "ming", "tile": 11, "from": 1},
                             {"seat": 2, "kind": "bu", "tile": 20}]
        multiplier = 1 + 6 * 4 / 10
        expected = (2 - multiplier, -4 - multiplier, 3 + 3 * multiplier, -1 - multiplier)
        self.assertEqual(expected_scores(game), expected)
        self.assertAlmostEqual(sum(expected_scores(game)), 0.0)
        game.wall.reverse()
        self.assertEqual(expected_scores(game), expected)
        game.winner = None
        self.assertEqual(expected_scores(game), (0.0,) * 4)

    def test_inactive_illegal_actions_and_red_claims_are_rejected(self):
        game = Game(seed=212, human_seat=-1)
        self.assertEqual(legal_actions(game, 1), ())
        with self.assertRaises(ValueError):
            apply_action(game, Action("discard", game.players[1].hand[0]), seat=1)
        with self.assertRaises(ValueError):
            apply_action(game, Action("pass"))
        red = fixture([27, 27, 27, 27, 0, 3, 6, 9, 12, 15, 18, 21, 24, 26])
        self.assertNotIn(Action("gang", 27), legal_actions(red, 0))

    def test_fixed_action_sequences_match_direct_engine_through_terminal(self):
        kinds = set()
        for seed in range(30, 40):
            game = Game(seed=seed, human_seat=-1)
            reference = copy.deepcopy(game)
            steps = 0
            while game.phase != "game_over":
                seat = acting_seat(game)
                action = base_action(game, seat, SimplePolicy)
                kinds.add(action.kind)
                apply_action(game, action, seat)
                if action.kind == "discard":
                    reference.action_discard(seat, action.tile)
                elif action.kind == "peng":
                    reference.action_peng(seat)
                elif action.kind == "gang":
                    reference.action_gang(seat, action.tile if reference.phase == "discard_wait" else None)
                else:
                    reference.action_pass(seat)
                self.assertEqual(state(game), state(reference))
                self.assertEqual(tile_counts(game), [4] * 28)
                steps += 1
                self.assertLess(steps, 500)
            self.assertEqual(sum(p.score_delta for p in game.players), 0)
        self.assertTrue({"discard", "peng", "gang"}.issubset(kinds))

    def test_advance_reports_public_actions_only_and_stops_at_hero(self):
        game = fixture([0, 0, 3, 4, 5, 9, 10, 11, 18, 19, 22, 22, 26],
                       hero=1, claim=(0, 0))
        # Seat 1 passes privately; the trace should begin with its eventual
        # visible discard, not a pass marker or the identity of a private draw.
        trace = advance(game, 0, policies=NoCalls)
        self.assertTrue(all(kind in {"discard", "peng", "gang"} for _, kind, _ in trace))
        self.assertFalse(any(kind in {"pass", "draw"} for _, kind, _ in trace))
        self.assertIn(acting_seat(game), {None, 0})
        self.assertEqual(advance(game, 0, policies=NoCalls), ())

    def test_rollout_matches_manual_full_protocol_and_honors_guard(self):
        for seed in range(4):
            manual = Game(seed=200 + seed, human_seat=-1)
            game = clone(manual)
            result = rollout(game, 2, policies=SimplePolicy)
            while manual.phase != "game_over":
                seat = acting_seat(manual)
                apply_action(manual, base_action(manual, seat, SimplePolicy), seat)
            self.assertEqual(state(game), state(manual))
            self.assertEqual(result.hero_score, expected_scores(manual)[2])
            self.assertEqual(result.winner, manual.winner)
            self.assertEqual(result.draw, manual.winner is None)
        with self.assertRaises(RuntimeError):
            rollout(Game(seed=400, human_seat=-1), 0, policies=NoCalls, max_steps=0)
        with self.assertRaises(RuntimeError):
            advance(Game(seed=401, human_seat=-1), 2, policies=NoCalls, max_steps=0)


if __name__ == "__main__":
    unittest.main()
