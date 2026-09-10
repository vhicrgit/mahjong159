import unittest

from backend.native import native
from backend.rules.win import is_win, shanten


class FourMeldBoundaryTests(unittest.TestCase):
    def test_all_singletons_and_pairs(self):
        for a in range(28):
            hand = [0] * 28
            hand[a] = 1
            self.assertEqual(native.shanten(hand), shanten(hand))
            for b in range(a, 28):
                two = hand.copy()
                two[b] += 1
                self.assertEqual(native.shanten(two), shanten(two), (a, b))
                self.assertEqual(native.is_win(two), is_win(two), (a, b))


if __name__ == "__main__":
    unittest.main()
