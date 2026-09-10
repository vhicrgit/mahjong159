"""Optimal finite-horizon *solitary* 159 Mahjong completion values.

Unlike a greedy ukeire/E rollout, this Bellman solver consumes every draw from
an integer pool and optimizes every subsequent discard. It includes jokers and
concealed hand sizes after melds. It does not model opponents, calls, kong
payments, or the real game's turn schedule. ``unseen_counts`` is the pool of
the explicitly chosen solitary abstraction, not a fractional posterior mean.

With discount=1, the objective is completion probability within the horizon.
With 0 < discount < 1, it is E[discount**(T-1) * 1(T <= horizon)] under the
optimal policy, where T is the completion draw. This constant survival-discount
model favors earlier completion; it does not model actual opponent win events.

``solve(h13, u, k)`` returns the model value and search diagnostics.
``solve_details`` is an equivalent descriptive entry point. Budgeted
results are intervals; ``value`` is their LOWER bound and ``exact`` is
false unless both bounds coincide. ``rank_discards`` reuses exact cached states
across candidates. Its budget, when supplied, is per candidate; ordering by
lower bounds is not a certified action ordering when intervals overlap.
``probability`` equals value only at discount=1; otherwise it is None and
``discounted_completion_value`` names the modeled quantity explicitly.
``nodes`` counts expanded chance and post-draw decision states; cache hits,
terminal states, and shanten impossibility bounds do not consume the budget.

The C kernel calls the existing native win/shanten functions directly through
function pointers. Its build products go under work/search159/build. A pure
Python implementation using the independent rules module is available with
backend="python"; backend="auto" falls back to it if C cannot be loaded.
"""
from __future__ import annotations

import ctypes
import hashlib
import math
import operator
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import threading
import warnings

_ROOT = Path(__file__).resolve().parents[2]
_SOURCE = _ROOT / "backend" / "native" / "finite_horizon.c"
_ABI_VERSION = 2
_LIB = None
_BUILD_LOCK = threading.Lock()
# Existing native primitives use process-global caches; serialize calls from
# this module so its contexts cannot race those caches with one another.
_SOLVE_LOCK = threading.RLock()
_I8 = ctypes.c_int8 * 28


def _integer(value, label):
    if isinstance(value, bool):
        raise ValueError(f"{label} must be an integer, not bool")
    try:
        return operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{label} must be an integer") from exc


def _validate(hand_counts, unseen_counts, horizon, *, postdraw=False):
    hand = tuple(_integer(x, "hand count") for x in hand_counts)
    unseen = tuple(_integer(x, "unseen count") for x in unseen_counts)
    if len(hand) != 28 or len(unseen) != 28:
        raise ValueError("hand_counts and unseen_counts must each have 28 entries")
    if any(h < 0 or u < 0 or h + u > 4 for h, u in zip(hand, unseen)):
        raise ValueError("counts must be nonnegative and hand[t] + unseen[t] <= 4")
    total = sum(hand)
    valid_sizes = (2, 5, 8, 11, 14) if postdraw else (1, 4, 7, 10, 13)
    if total not in valid_sizes:
        raise ValueError(f"concealed hand size must be one of {valid_sizes}")
    horizon = _integer(horizon, "horizon")
    if horizon < 0:
        raise ValueError("horizon must be nonnegative")
    return hand, unseen, min(horizon, sum(unseen))


def _budget(max_nodes):
    if max_nodes is None:
        return 0  # C's convention for unbounded search.
    max_nodes = _integer(max_nodes, "max_nodes")
    if not 1 <= max_nodes < 2**64:
        raise ValueError("max_nodes must be a positive 64-bit integer or None")
    return max_nodes


def _discount(value):
    if isinstance(value, bool):
        raise ValueError("discount must be a real number in (0, 1]")
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("discount must be a real number in (0, 1]") from exc
    if not math.isfinite(value) or not 0.0 < value <= 1.0:
        raise ValueError("discount must be a real number in (0, 1]")
    return value


def _library():
    global _LIB
    with _BUILD_LOCK:
        if _LIB is not None:
            return _LIB
        digest = hashlib.sha256(_SOURCE.read_bytes()).hexdigest()[:16]
        build = Path(os.environ.get("FINITE_HORIZON_BUILD_DIR",
                                    str(_ROOT / "work" / "search159" / "build")))
        build.mkdir(parents=True, exist_ok=True)
        target = build / f"finite_horizon_{digest}.so"
        if not target.exists():
            fd, tmp = tempfile.mkstemp(prefix="finite_horizon_", suffix=".so.tmp", dir=build)
            os.close(fd)
            try:
                command = shlex.split(os.environ.get("CC", "cc"))
                command += ["-O3", "-std=c99", "-fPIC", "-shared", "-o", tmp, str(_SOURCE)]
                subprocess.run(command, check=True, capture_output=True, text=True)
                os.replace(tmp, target)
            finally:
                if os.path.exists(tmp):
                    os.unlink(tmp)
        lib = ctypes.CDLL(str(target))
        try:
            version = lib.fh_abi_version
        except AttributeError as exc:
            raise RuntimeError(f"finite-horizon library has no ABI version: {target}") from exc
        version.argtypes = []
        version.restype = ctypes.c_int
        actual = version()
        if actual != _ABI_VERSION:
            raise RuntimeError(f"finite-horizon ABI mismatch: Python expects {_ABI_VERSION}, "
                               f"C provides {actual} ({target}); restart the process")
        lib.fh_create.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int,
                                 ctypes.c_double]
        lib.fh_create.restype = ctypes.c_void_p
        lib.fh_destroy.argtypes = [ctypes.c_void_p]
        lib.fh_destroy.restype = None
        lib.fh_solve.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int8),
                                ctypes.POINTER(ctypes.c_int8), ctypes.c_int,
                                ctypes.c_uint64, ctypes.POINTER(ctypes.c_double),
                                ctypes.POINTER(ctypes.c_uint64)]
        lib.fh_solve.restype = ctypes.c_int
        _LIB = lib
        return lib


def _result(lo, hi, nodes, hits, cutoffs, horizon, backend, discount):
    # Weighted sums may stray outside [0, 1] by a final rounding bit.
    lo = max(0.0, min(1.0, float(lo)))
    hi = max(lo, min(1.0, float(hi)))
    result = {"value": lo, "discount": discount,
            "objective": "completion_probability" if discount == 1.0 else "discounted_completion",
            "probability": lo if discount == 1.0 else None,
            "lower_bound": lo, "upper_bound": hi,
            "exact": lo == hi, "nodes": int(nodes), "cache_hits": int(hits),
            "cutoffs": int(cutoffs), "horizon": horizon, "backend": backend}
    if discount != 1.0:
        result["discounted_completion_value"] = lo
    return result


class _NativeSolver:
    def __init__(self, cache_bits, discount):
        from ..native import native
        self.lib = _library()
        # Keep the primitive CDLL alive for the lifetime of its function pointers.
        self.primitives = native.lib()
        self.discount = discount
        self.ctx = self.lib.fh_create(
            ctypes.cast(self.primitives.mj_is_win, ctypes.c_void_p),
            ctypes.cast(self.primitives.mj_shanten, ctypes.c_void_p), cache_bits, discount)
        if not self.ctx:
            raise MemoryError("could not allocate finite-horizon transposition table")

    def close(self):
        if self.ctx:
            self.lib.fh_destroy(self.ctx)
            self.ctx = None

    def solve(self, hand, unseen, horizon, budget):
        bounds = (ctypes.c_double * 2)()
        stats = (ctypes.c_uint64 * 3)()
        rc = self.lib.fh_solve(self.ctx, _I8(*hand), _I8(*unseen), horizon,
                               budget, bounds, stats)
        if rc:
            raise RuntimeError(f"finite-horizon C solver rejected input: {rc}")
        return _result(*bounds, *stats, horizon, "c", self.discount)


class _PythonSolver:
    def __init__(self, discount):
        from ..rules.win import is_win, shanten_cached
        self.win = is_win
        self.shanten = shanten_cached
        self.memo = {}
        self.discount = discount

    def close(self):
        pass

    def solve(self, hand, unseen, horizon, budget):
        self.nodes = self.hits = self.cutoffs = 0
        self.budget = budget
        lo, hi = self._chance(hand, unseen, horizon)
        return _result(lo, hi, self.nodes, self.hits, self.cutoffs, horizon, "python", self.discount)

    def _enter(self, key):
        cached = self.memo.get(key)
        if cached is not None:
            self.hits += 1
            return (cached, cached)
        if self.budget and self.nodes >= self.budget:
            self.cutoffs += 1
            return (0.0, 1.0)
        self.nodes += 1
        return None

    def _finish(self, key, lo, hi):
        if lo == hi:
            self.memo[key] = lo
        return lo, hi

    def _chance(self, hand, unseen, k):
        total = sum(unseen)
        if k <= 0 or total <= 0:
            return 0.0, 0.0
        k = min(k, total)
        if self.shanten(hand) >= k:
            return 0.0, 0.0
        key = (0, hand, unseen, k)
        cached = self._enter(key)
        if cached is not None:
            return cached
        lo = hi = 0.0
        for t, weight in enumerate(unseen):
            if not weight:
                continue
            h = list(hand); h[t] += 1
            u = list(unseen); u[t] -= 1
            if self.win(h):
                a = b = 1.0
            else:
                a, b = self._decision(tuple(h), tuple(u), k - 1)
                a *= self.discount
                b *= self.discount
            lo += weight * a
            hi += weight * b
        return self._finish(key, lo / total, hi / total)

    def _decision(self, hand, unseen, k):
        if k <= 0:
            return 0.0, 0.0
        key = (1, hand, unseen, k)
        cached = self._enter(key)
        if cached is not None:
            return cached
        lo = hi = 0.0
        for d, count in enumerate(hand):
            if not count:
                continue
            h = list(hand); h[d] -= 1
            a, b = self._chance(tuple(h), unseen, k)
            lo, hi = max(lo, a), max(hi, b)
            if lo == 1.0:
                hi = 1.0
                break
        return self._finish(key, lo, hi)


def _make_solver(backend, cache_bits, discount):
    if backend not in ("auto", "c", "python"):
        raise ValueError("backend must be 'auto', 'c', or 'python'")
    cache_bits = _integer(cache_bits, "cache_bits")
    if not 8 <= cache_bits <= 22:
        raise ValueError("cache_bits must be between 8 and 22")
    if backend == "python":
        return _PythonSolver(discount)
    try:
        return _NativeSolver(cache_bits, discount)
    except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
        if backend == "c":
            raise
        warnings.warn(f"C finite-horizon backend unavailable; using Python: {exc}",
                      RuntimeWarning, stacklevel=3)
        return _PythonSolver(discount)


def solve(hand_counts, unseen_counts, horizon, *, max_nodes=None,
          discount=1.0, backend="auto", cache_bits=18):
    """Return optimal solitary completion value, bounds, and diagnostics.

    Input concealed hand has 1, 4, 7, 10 or 13 tiles. Exhausting the supplied
    pool without winning is failure. Without max_nodes the result is exact
    within that model; large horizons can be expensive. With a budget,
    value is a lower bound; inspect exact and upper_bound. At discount<1 the
    value favors earlier completion and the probability field is None.
    """
    return solve_details(hand_counts, unseen_counts, horizon, max_nodes=max_nodes,
                         discount=discount, backend=backend, cache_bits=cache_bits)


def solve_details(hand_counts, unseen_counts, horizon, *, max_nodes=None,
                  discount=1.0, backend="auto", cache_bits=18):
    """As ``solve``, with diagnostics and optional per-call node budget."""
    hand, unseen, horizon = _validate(hand_counts, unseen_counts, horizon)
    budget = _budget(max_nodes)
    discount = _discount(discount)
    with _SOLVE_LOCK:
        solver = _make_solver(backend, cache_bits, discount)
        try:
            return solver.solve(hand, unseen, horizon, budget)
        finally:
            solver.close()


def rank_discards(hand_counts, unseen_counts, horizon, *, max_nodes=None,
                  discount=1.0, backend="auto", cache_bits=18):
    """Return legal discard rows sorted by objective value descending, then tile.

    The supplied hand has 2, 5, 8, 11 or 14 tiles; horizon counts FUTURE draws
    after the candidate discard. Discarded tiles stay visible, so the input
    unseen pool does not grow. Candidate rows include bounds/exactness/nodes.
    max_nodes, if given, applies separately to each candidate. Shared exact
    transpositions can save work for later candidates.
    """
    hand, unseen, horizon = _validate(hand_counts, unseen_counts, horizon, postdraw=True)
    budget = _budget(max_nodes)
    discount = _discount(discount)
    with _SOLVE_LOCK:
        solver = _make_solver(backend, cache_bits, discount)
        try:
            rows = []
            for tile, count in enumerate(hand):
                if not count:
                    continue
                h = list(hand); h[tile] -= 1
                row = solver.solve(tuple(h), unseen, horizon, budget)
                rows.append({"tile": tile, **row})
            return sorted(rows, key=lambda row: (-row["value"], row["tile"]))
        finally:
            solver.close()
