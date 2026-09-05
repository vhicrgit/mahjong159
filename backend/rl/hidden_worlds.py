"""Uniform count-consistent hidden worlds for first-win search.

This is a baseline, not a history-conditioned opponent posterior. Never samples
from the realized allocation/order; only the hero hand and public tiles count.
Blood-battle games expose wall tiles after wins and need a different sampler.
"""

import copy
import numpy as np


def sample_world(game, hero, rng):
    if game.bloody or game.phase != "discard_wait" or game.turn != hero:
        raise ValueError("Uniform sampler supports only first-win hero discard states")
    visible = np.array(game.players[hero].hand_counts, dtype=int)
    for p in game.players:
        for tile in p.discards:
            visible[tile] += 1
        for meld in p.melds:
            visible[meld['tile']] += 3 if meld['type'] == 'peng' else 4
    if np.any(visible > 4):
        raise ValueError("Visible tile counts exceed deck")
    unseen = np.repeat(np.arange(28), 4 - visible)
    expected = len(game.wall) + sum(len(p.hand) for p in game.players if p.seat != hero)
    if len(unseen) != expected:
        raise ValueError("Hidden tile accounting does not match hand sizes and wall")
    rng.shuffle(unseen)
    world = copy.deepcopy(game)
    pos = 0
    for p in world.players:
        if p.seat != hero:
            size = len(p.hand)
            p.hand = sorted(unseen[pos:pos + size].tolist())
            pos += size
    world.wall = unseen[pos:].tolist()
    return world
