"""Production Bot factory and legacy server-loop integration checks."""
import unittest

from backend.ai.roster import make_bot
from backend.ai.search159 import simulator as sim
from backend.game.engine import Game


class SearchAdapterTests(unittest.TestCase):
    def test_factory_bots_complete_games_through_legacy_server_protocol(self):
        for kind, param in (("search159", 1), ("finite159", 2)):
            with self.subTest(kind=kind):
                game = Game(seed=766107, human_seat=-1)
                bot = make_bot(kind, game, 2, param)
                steps = 0
                while game.phase != "game_over":
                    seat = sim.acting_seat(game)
                    if seat != 2:
                        action = sim.base_action(game, seat)
                    else:
                        planned = bot.choose_action()
                        searches_before = dict(bot.stats)
                        if game.phase == "react_wait":
                            pending = game.pending_actions[seat]
                            tile = game.last_discard
                            if pending.get("gang") and bot.decide_gang(tile, "ming"):
                                action = sim.Action("gang", tile)
                            elif pending.get("peng") and bot.decide_peng(tile):
                                action = sim.Action("peng", tile)
                            else:
                                action = sim.Action("pass")
                        else:
                            action = None
                            for tile in game._gang_options(seat):
                                gang_kind = "an" if game.players[seat].hand.count(tile) == 4 else "bu"
                                if bot.decide_gang(tile, gang_kind):
                                    action = sim.Action("gang", tile)
                                    break
                            if action is None:
                                action = sim.Action("discard", bot.choose_discard())
                        self.assertEqual(action, planned)
                        self.assertEqual(bot.stats, searches_before)
                    self.assertIn(action, sim.legal_actions(game, seat))
                    sim.apply_action(game, action, seat)
                    steps += 1
                    self.assertLess(steps, 1000)
                self.assertEqual(sum(p.score_delta for p in game.players), 0)


if __name__ == "__main__":
    unittest.main()
