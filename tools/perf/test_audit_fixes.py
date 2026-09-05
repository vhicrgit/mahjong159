"""Behavioral regression checks for the September independent audit fixes."""

import unittest
import numpy as np
import torch

from backend.game.bot_driver import try_self_gang
from backend.game.engine import Game
from backend.rl import eval_crn, cf_collect
from backend.rl.features_v2 import encode_state
from tools.dagger_train import legal_from_features, policy_loss, split_indices
from backend.rl.hidden_worlds import sample_world
from backend.analysis.opp_model import OppTracker


class AuditFixes(unittest.TestCase):
    def test_hard_label_learns_among_legal_alternatives(self):
        q = torch.tensor([[1., 2., 3., 99.]], requires_grad=True)
        p = torch.tensor([[1., 0., 0., 0.]])
        legal = torch.tensor([[True, True, True, False]])
        loss = policy_loss(q, p, legal).mean()
        loss.backward()
        self.assertGreater(loss.item(), 2)
        self.assertLess(q.grad[0, 0].item(), 0)
        self.assertGreater(q.grad[0, 2].item(), 0)
        self.assertEqual(q.grad[0, 3].item(), 0)

    def test_sparse_soft_target_penalizes_omitted_legal_action(self):
        q = torch.zeros(1, 4, requires_grad=True)
        policy_loss(q, torch.tensor([[.7, .3, 0., 0.]]),
                    torch.tensor([[True, True, True, False]])).sum().backward()
        self.assertGreater(q.grad[0, 2].item(), 0)
        self.assertEqual(q.grad[0, 3].item(), 0)

    def test_recovered_legality_matches_real_hands(self):
        for seed in range(10):
            g = Game(seed=seed, human_seat=-1)
            for s in range(4):
                x = torch.from_numpy(encode_state(g, s)).unsqueeze(0)
                actual = torch.tensor([[n > 0 for n in g.players[s].hand_counts]])
                self.assertTrue(torch.equal(legal_from_features(x), actual))

    def test_invalid_target_rejected(self):
        with self.assertRaises(ValueError):
            policy_loss(torch.zeros(1, 2), torch.tensor([[1., 0.]]),
                        torch.tensor([[False, True]]))

    def test_group_split_and_tiny_dataset(self):
        groups = np.repeat(np.arange(10), 3)
        tr, va = split_indices(len(groups), 123, groups)
        self.assertFalse(set(groups[tr.tolist()]) & set(groups[va.tolist()]))
        tr, va = split_indices(2, 123)
        self.assertEqual((len(tr), len(va)), (1, 1))

    def test_correlated_seats_use_seed_standard_error(self):
        d = np.repeat(np.array([[1.], [2.], [4.], [7.]]), 4, axis=1)
        s = eval_crn._stat(d)
        self.assertEqual(s['n'], 4)
        self.assertAlmostEqual(s['se'], d[:, 0].std(ddof=1) / 2)
        self.assertEqual(s['observations'], 16)

    def test_concealed_and_added_kong(self):
        class Bot:
            def __init__(self): self.kinds = []
            def decide_gang(self, tile, kind):
                self.kinds.append(kind)
                return True
        for kind in ('an', 'bu'):
            g = Game(seed=123, human_seat=-1)
            g.phase, g.turn = 'discard_wait', 0
            g.players[0].hand = ([0] * 4 if kind == 'an' else [0]) + [3,4,6,9,11,13,15,17,19,21]
            g.players[0].melds = [] if kind == 'an' else [{'type':'peng', 'tile':0}]
            g.wall = [2,5,8,12,14,16,18,20]
            b = Bot()
            self.assertTrue(try_self_gang(g, b))
            self.assertEqual(b.kinds, [kind])
            self.assertEqual(g.gang_records[-1]['kind'], kind)
            self.assertEqual(len(g.wall), 7)

    def test_first_win_draw_does_not_reward_unsettled_kong(self):
        g = Game(seed=123, human_seat=-1)
        g.huangzhuang, g.phase = True, 'game_over'
        g.gang_records = [{'seat':0, 'kind':'ming', 'from':1}]
        self.assertEqual(cf_collect.default_reward(g, 0), 0)

    def test_worlds_preserve_observation_and_ignore_actual_hidden_allocation(self):
        import copy
        g = Game(seed=123, human_seat=-1)
        alt = copy.deepcopy(g)
        alt.wall.reverse()
        alt.players[1].hand, alt.players[2].hand = alt.players[2].hand, alt.players[1].hand
        a = sample_world(g, 0, np.random.default_rng(91))
        b = sample_world(alt, 0, np.random.default_rng(91))
        c = sample_world(g, 0, np.random.default_rng(92))
        self.assertEqual(a.wall, b.wall)
        self.assertEqual([p.hand for p in a.players], [p.hand for p in b.players])
        self.assertNotEqual(a.wall, c.wall)
        self.assertEqual(a.players[0].hand, g.players[0].hand)
        self.assertEqual(len(a.wall), len(g.wall))
        self.assertTrue(np.array_equal(encode_state(a, 0), encode_state(g, 0)))
        counts = np.bincount(a.wall + sum([p.hand for p in a.players], []), minlength=28)
        self.assertTrue(np.array_equal(counts, np.full(28, 4)))

    def test_tracker_kong_materializes_pending_draw(self):
        tracker = OppTracker(1, [0] * 28, policy=False, n_init=10, beam=10)
        h = [0] * 28
        h[0] = 3
        for t in range(1, 11): h[t] = 1
        tracker.particles = {tuple(h): 1.}
        tracker.pending_draw = True
        tracker.notify_self_gang(1, 0, 'an')
        self.assertFalse(tracker.pending_draw)
        self.assertEqual(tracker.melds[1], [('gang', 0, 'an')])
        self.assertTrue(tracker.particles)
        self.assertTrue(all(sum(h) == 10 and h[0] == 0 for h in tracker.particles))

    def test_tracker_added_kong_upgrades_existing_meld(self):
        tracker = OppTracker(1, [0] * 28, policy=False, n_init=10, beam=10)
        h = [0] * 28
        for t in range(11): h[t] = 1
        tracker.particles = {tuple(h): 1.}
        tracker.melds[1] = [('peng', 0, None)]
        tracker.notify_self_gang(1, 0, 'bu')
        self.assertEqual(tracker.melds[1], [('gang', 0, 'bu')])
        self.assertTrue(all(sum(h) == 10 and h[0] == 0 for h in tracker.particles))

    def test_collectors_and_evaluator_match_with_same_policy(self):
        from backend.ai.bot_native import NativeV31
        from backend.rl.model import build_model, legal_discard_mask
        from backend.rl.vec_selfplay import VectorizedSelfPlay
        torch.manual_seed(7)
        model = build_model('tiny').eval()
        class Seat(NativeV31):
            def choose_discard(self):
                x = torch.from_numpy(encode_state(self.game, self.seat)).unsqueeze(0)
                mask = legal_discard_mask(self.game.players[self.seat].hand_counts).unsqueeze(0)
                with torch.no_grad():
                    return int(model.q(x, mask)[0].argmax())
        vec = VectorizedSelfPlay(model, 12, 'cpu', seed0=200, record=False)
        vec.bots = [{s: NativeV31(g, s) for s in range(4)} for g in vec.games]
        vec.run(temperature=0)
        games, _ = cf_collect.run_games(model, 'cpu',
                                        [Game(seed=s, human_seat=-1) for s in range(200, 212)], 0)
        for i, seed in enumerate(range(200, 212)):
            ref = eval_crn._play(seed, False, {s: Seat for s in range(4)})
            for game in [vec.games[i], games[i]]:
                self.assertEqual(game.phase, 'game_over')
                self.assertEqual(game.log, ref.log)
                self.assertEqual([p.score_delta for p in game.players],
                                 [p.score_delta for p in ref.players])


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
