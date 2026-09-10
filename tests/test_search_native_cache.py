"""Exact second-step memo parity and a paired real-search benchmark.

The optional benchmark uses separate forked children with the same warmed
primitive memo state. Its enabled second-step cache starts empty, so its gains
must arise inside the actual multi-candidate shared-world search workload.
"""

import ctypes
import random
import time
import unittest

from backend.native import native
from backend.ai.search159.fast_rollout import rollout_fast, warmup
from backend.ai.search159.planner import Planner, SearchConfig, candidates
from backend.ai.search159 import simulator as sim

try:
    from test_fast_rollout import collect_snapshots, rule_state
except ImportError:
    from tests.test_fast_rollout import collect_snapshots, rule_state


def cache_api():
    lib = native.lib()
    i8p = ctypes.POINTER(ctypes.c_int8)
    lib.mj_second_step_cache_set_enabled.argtypes = [ctypes.c_int]
    lib.mj_second_step_cache_set_enabled.restype = None
    lib.mj_second_step_cache_clear.argtypes = []
    lib.mj_second_step_cache_clear.restype = None
    lib.mj_second_step_cache_stats.argtypes = [ctypes.POINTER(ctypes.c_uint64)]
    lib.mj_second_step_cache_stats.restype = None
    lib.mj_second_step_value.argtypes = [i8p, i8p]
    lib.mj_second_step_value.restype = ctypes.c_double
    return lib


def cache_stats(lib):
    out = (ctypes.c_uint64 * 6)()
    lib.mj_second_step_cache_stats(out)
    return dict(zip(("calls", "hits", "misses", "collisions", "capacity", "bytes"), out))


def counts(tiles):
    hand = [0] * 28
    for tile in tiles:
        hand[tile] += 1
    return hand


def random_key(rng, hand_size):
    hand = counts(rng.sample(list(range(28)) * 4, hand_size))
    unseen = [rng.randrange(5 - n) for n in hand]
    return hand, unseen


def cache_index(hand, unseen):
    """Mirror only the public source hash to locate a direct-map collision."""
    mask = (1 << 64) - 1
    value = 14695981039346656037
    for h, u in zip(hand, unseen):
        value = ((value ^ h) * 1099511628211) & mask
        value = ((value ^ u) * 1099511628211) & mask
    value ^= value >> 33
    value = (value * 0xff51afd7ed558ccd) & mask
    value ^= value >> 33
    value = (value * 0xc4ceb9fe1a85ec53) & mask
    value ^= value >> 33
    return value & ((1 << 17) - 1)


class NativeSecondStepCacheTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        warmup()
        cls.lib = cache_api()
        cls.snapshots = collect_snapshots(32, seed0=984500)

    def setUp(self):
        self.lib.mj_second_step_cache_set_enabled(1)
        self.lib.mj_second_step_cache_clear()

    def tearDown(self):
        self.lib.mj_second_step_cache_set_enabled(1)
        self.lib.mj_second_step_cache_clear()

    def value(self, key):
        return self.lib.mj_second_step_value(native._i8(key[0]), native._i8(key[1]))

    def test_250_distinct_keys_match_disabled_bit_for_bit(self):
        rng = random.Random(538240)
        keys = [random_key(rng, (1, 4, 7, 10, 13)[i % 5]) for i in range(250)]
        self.assertEqual(len({tuple(h + u) for h, u in keys}), 250)
        self.lib.mj_second_step_cache_set_enabled(0)
        expected = [self.value(key).hex() for key in keys]
        self.assertEqual(cache_stats(self.lib)["hits"], 0)
        self.lib.mj_second_step_cache_clear()
        self.lib.mj_second_step_cache_set_enabled(1)
        for index, key in enumerate(keys):
            with self.subTest(index=index):
                self.assertEqual(self.value(key).hex(), expected[index])
                self.assertEqual(self.value(key).hex(), expected[index])
        stats = cache_stats(self.lib)
        self.assertEqual(stats["calls"], 500)
        self.assertEqual(stats["hits"], 250)
        self.assertEqual(stats["misses"], 250)
        self.assertEqual(stats["capacity"], 1 << 17)
        self.assertEqual(stats["bytes"], 10 * 1024 * 1024)

    def test_each_hand_and_unseen_component_including_red_is_in_key(self):
        hand = counts([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 27])
        unseen = [1] * 28
        keys = [(hand, unseen)]
        for tile in range(28):
            changed_hand = hand.copy()
            source = 0 if tile != 0 else 1
            changed_hand[source] -= 1
            changed_hand[tile] += 1
            keys.append((changed_hand, unseen))
            changed_unseen = unseen.copy()
            changed_unseen[tile] += 1
            changed_unseen[(tile + 1) % 28] -= 1  # Same pool total.
            keys.append((hand, changed_unseen))
        self.lib.mj_second_step_cache_set_enabled(0)
        expected = [self.value(key).hex() for key in keys]
        self.lib.mj_second_step_cache_clear()
        self.lib.mj_second_step_cache_set_enabled(1)
        for index, key in enumerate(keys):
            self.assertEqual(self.value(key).hex(), expected[index])
        stats = cache_stats(self.lib)
        self.assertEqual(stats["hits"], 0)
        self.assertEqual(stats["misses"], 57)

    def test_empty_pool_and_mutated_buffers_are_not_stale(self):
        hand = counts([0, 1, 2, 3, 4, 5, 9, 10, 11, 27, 27, 27, 27])
        hbuf, ubuf = native._i8(hand), native._i8([0] * 28)
        self.assertEqual(self.lib.mj_second_step_value(hbuf, ubuf), 0.0)
        self.assertEqual(self.lib.mj_second_step_value(hbuf, ubuf), 0.0)
        ubuf[20] = 3
        changed = self.lib.mj_second_step_value(hbuf, ubuf)
        self.assertEqual(cache_stats(self.lib)["misses"], 2)
        self.lib.mj_second_step_cache_set_enabled(0)
        self.assertEqual(self.lib.mj_second_step_value(hbuf, ubuf).hex(), changed.hex())
        self.lib.mj_second_step_cache_clear()
        self.assertEqual(cache_stats(self.lib)["calls"], 0)
        # Clear preserves disabled state, and disabled calls neither read nor write.
        self.lib.mj_second_step_value(hbuf, ubuf)
        self.lib.mj_second_step_value(hbuf, ubuf)
        self.assertEqual(cache_stats(self.lib)["hits"], 0)
        self.lib.mj_second_step_cache_set_enabled(1)
        self.lib.mj_second_step_value(hbuf, ubuf)
        self.assertEqual(cache_stats(self.lib)["misses"], 3)

    def test_direct_map_collision_evicts_without_reusing_another_value(self):
        rng, occupied, collision = random.Random(834512), {}, None
        for _ in range(10000):
            key = random_key(rng, 13)
            index = cache_index(*key)
            if index in occupied and occupied[index] != key:
                collision = (occupied[index], key)
                break
            occupied[index] = key
        self.assertIsNotNone(collision)
        self.lib.mj_second_step_cache_set_enabled(0)
        expected = [self.value(key).hex() for key in collision]
        self.lib.mj_second_step_cache_clear()
        self.lib.mj_second_step_cache_set_enabled(1)
        for index in (0, 1, 0, 1, 1):
            self.assertEqual(self.value(collision[index]).hex(), expected[index])
        stats = cache_stats(self.lib)
        self.assertEqual((stats["hits"], stats["misses"], stats["collisions"]), (1, 4, 3))

    def test_continuation_and_scores_remain_exact_when_weights_change(self):
        hand = counts([0, 1, 2, 3, 4, 5, 9, 10, 11, 18, 19, 20, 21, 27])
        unseen = [4 - n for n in hand]
        for sw, uw, cw, rw, eg, maximum in (
            (100, 1, .5, 0, 0, 2), (85.5, 1.3, .17, 7.5, .8, 1),
            (100, 1, 0, 1, 1, -1), (70, 2, 3, 4, .2, 4),
        ):
            args = dict(sw=sw, uw=uw, cw=cw, rw=rw, cont_max=maximum)
            self.lib.mj_second_step_cache_set_enabled(0)
            expected = native.score_discards_v10(hand, unseen, [0] * 28, eg, **args)
            self.lib.mj_second_step_cache_set_enabled(1)
            actual = native.score_discards_v10(hand, unseen, [0] * 28, eg, **args)
            self.assertEqual(actual, expected)
        self.assertGreater(cache_stats(self.lib)["hits"], 0)

    def test_shared_world_candidate_rollouts_match_complete_final_state(self):
        rng, cfg = random.Random(781509), SearchConfig(candidate_limit=4)
        trials = 0
        for snapshot in self.snapshots:
            hero = sim.acting_seat(snapshot)
            actions, _ = candidates(snapshot, hero, cfg)
            world = sim.sample_world(sim.observe(snapshot, hero), rng)
            for action in actions:
                results = {}
                order = (0, 1) if trials % 2 else (1, 0)
                for enabled in order:
                    self.lib.mj_second_step_cache_set_enabled(enabled)
                    game = sim.clone(world, keep_log=False)
                    sim.apply_action(game, action, hero)
                    results[enabled] = rollout_fast(game, hero, mutate=True)
                self.assertEqual(results[0], results[1])
                trials += 1
        self.assertGreaterEqual(trials, 80)

    def test_full_python_action_traces_match_with_cache_on_and_off(self):
        for snapshot in self.snapshots[::4]:
            traces = []
            for enabled in (0, 1):
                self.lib.mj_second_step_cache_set_enabled(enabled)
                game, trace = sim.clone(snapshot), []
                while game.phase != "game_over":
                    actor = sim.acting_seat(game)
                    action = sim.base_action(game, actor)
                    trace.append((actor, action))
                    sim.apply_action(game, action, actor)
                traces.append((trace, rule_state(game), sim.expected_scores(game)))
            self.assertEqual(traces[0], traces[1])


def _timed_search_child(pipe, observation, config, seed, enabled):
    """Only called in a sequential fork inheriting identical primitive caches."""
    try:
        lib = cache_api()
        lib.mj_second_step_cache_clear()
        lib.mj_second_step_cache_set_enabled(enabled)
        planner = Planner(config, seed=seed)
        cpu0, wall0 = time.process_time(), time.perf_counter()
        result = planner.plan(observation)
        cpu, wall = time.process_time() - cpu0, time.perf_counter() - wall0
        pipe.send({"result": result, "cpu": cpu, "wall": wall,
                   "cache": cache_stats(lib), "simulations": planner.stats["simulations"],
                   "candidates": planner.stats["root_candidates"]})
    except BaseException as exc:
        pipe.send({"error": repr(exc)})
    finally:
        pipe.close()


def benchmark_shared_world_search(n=12, simulations=8, depth=2):
    """Real Planner.plan AB/BA timings, including clones, tree and rollouts.

    Run from a standalone, single-threaded Python process. Before each pair,
    the parent warms the exact search with second-step caching disabled. Both
    sequential children inherit identical SH/TI/WIN memo contents and empty
    second-step caches. Thus mode ordering cannot confer primitive-cache warmth.
    Fork/startup, initial compile, snapshot collection and parent warmup are
    outside timing; all actual planning and serialization are inside timing.
    """
    import multiprocessing

    warmup()
    lib = cache_api()
    cfg = SearchConfig(simulations=simulations, confirmation=simulations,
                       depth=depth, candidate_limit=4, fast_rollout=True)
    snapshots = collect_snapshots(max(64, n * 8), seed0=983100)
    eligible = [g for g in snapshots if g.phase == "discard_wait"
                and len(sim.legal_actions(g, sim.acting_seat(g))) >= 4]
    if len(eligible) < n:
        raise ValueError("not enough multi-candidate snapshots")
    selected = [eligible[i * len(eligible) // n] for i in range(n)]
    ctx, records = multiprocessing.get_context("fork"), []
    try:
        for i, snapshot in enumerate(selected):
            hero, seed = sim.acting_seat(snapshot), 690300 + i
            observation = sim.observe(snapshot, hero)
            lib.mj_second_step_cache_set_enabled(0)
            lib.mj_second_step_cache_clear()
            reference = Planner(cfg, seed=seed).plan(observation)
            modes = {}
            for enabled in ((0, 1) if i % 2 == 0 else (1, 0)):
                receiving, sending = ctx.Pipe(duplex=False)
                process = ctx.Process(target=_timed_search_child,
                                      args=(sending, observation, cfg, seed, enabled))
                process.start()
                sending.close()
                if not receiving.poll(60):
                    process.terminate()
                    process.join()
                    raise RuntimeError("search timing child timed out")
                data = receiving.recv()
                receiving.close()
                process.join()
                if process.exitcode or "error" in data:
                    raise RuntimeError(data.get("error", f"child exited {process.exitcode}"))
                if data.pop("result") != reference:
                    raise AssertionError(f"cached search changed its action or values: pair {i}")
                modes[enabled] = data
            records.append({"index": i, "uncached": modes[0], "cached": modes[1]})
    finally:
        lib.mj_second_step_cache_set_enabled(1)
        lib.mj_second_step_cache_clear()
    totals = {name: {metric: sum(row[name][metric] for row in records)
                     for metric in ("cpu", "wall", "simulations")}
              for name in ("uncached", "cached")}
    hits = sum(row["cached"]["cache"]["hits"] for row in records)
    calls = sum(row["cached"]["cache"]["calls"] for row in records)
    return {"searches_per_mode": n, "depth": depth,
            "root_candidates": [r["cached"]["candidates"] for r in records],
            "totals": totals, "cache_hits": hits, "cache_calls": calls,
            "cache_hit_rate": hits / calls if calls else 0.0,
            "cpu_speedup": totals["uncached"]["cpu"] / totals["cached"]["cpu"],
            "wall_speedup": totals["uncached"]["wall"] / totals["cached"]["wall"],
            "records": records,
            "note": "Actual multi-candidate shared-world Planner; empty second-step cache; "
                    "identically warmed primitive caches inherited by sequential paired forks; "
                    "AB/BA order balanced; all decisions and estimates exactly equal."}


if __name__ == "__main__":
    unittest.main()
