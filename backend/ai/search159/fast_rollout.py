"""One-FFI-call v31 continuations using the production native primitives.

All four players must use v31/v31n. Other policies deliberately fall back at the
planner level instead of silently changing the continuation policy. Environment
weights are read through the same V10Bot constructor as NativeV31 on every call.

The C implementation owns only game transitions. It receives native C function
addresses from libmj159; no tile table is duplicated and no Python callbacks run
inside the rollout. Generated libraries are content-addressed under build/.
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import platform
import random
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from ...rules.tiles import tile_name

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[2]
_SOURCE = _HERE / "fast_rollout.c"
_BUILD = _ROOT / "build" / "search159"
_LIB = None
_NATIVE_LIB = None
_FUNCTIONS = None

_PHASES = {"discard_wait": 0, "react_wait": 1, "game_over": 2}
_PHASE_NAMES = {value: key for key, value in _PHASES.items()}
_KINDS = {"ming": 0, "an": 1, "bu": 2}
_KIND_NAMES = {value: key for key, value in _KINDS.items()}
_WIN_KINDS = {None: 0, "zimo": 1, "gangshang": 2, "tianhu": 3}
_WIN_NAMES = {value: key for key, value in _WIN_KINDS.items()}


class _Meld(ctypes.Structure):
    _fields_ = [("tile", ctypes.c_int32), ("type", ctypes.c_int32),
                ("kind", ctypes.c_int32), ("wr", ctypes.c_int32)]


class _Player(ctypes.Structure):
    _fields_ = [
        ("hand", ctypes.c_int8 * 28),
        ("discards", ctypes.c_int8 * 112),
        ("n_discards", ctypes.c_int32),
        ("melds", _Meld * 4),
        ("n_melds", ctypes.c_int32),
        ("score", ctypes.c_int32),
    ]


class _Gang(ctypes.Structure):
    _fields_ = [("seat", ctypes.c_int32), ("kind", ctypes.c_int32),
                ("tile", ctypes.c_int32), ("from_seat", ctypes.c_int32)]


class _State(ctypes.Structure):
    _fields_ = [
        ("players", _Player * 4),
        ("wall", ctypes.c_int8 * 112),
        ("head", ctypes.c_int32), ("tail", ctypes.c_int32),
        ("gangs", _Gang * 16), ("n_gangs", ctypes.c_int32),
        ("phase", ctypes.c_int32), ("turn", ctypes.c_int32),
        ("last_discard", ctypes.c_int32), ("last_discarder", ctypes.c_int32),
        ("pending", ctypes.c_uint8 * 4), ("pending_order", ctypes.c_int8 * 4),
        ("n_pending", ctypes.c_int32),
        ("winner", ctypes.c_int32), ("win_tile", ctypes.c_int32),
        ("win_kind", ctypes.c_int32), ("huangzhuang", ctypes.c_int32),
        ("n_159", ctypes.c_int32), ("fan_159", ctypes.c_int8 * 6),
        ("n_fan", ctypes.c_int32),
        ("last_drawn_seat", ctypes.c_int32), ("last_drawn_tile", ctypes.c_int32),
        ("last_action_kind", ctypes.c_int32), ("last_action_seat", ctypes.c_int32),
        ("last_action_tile", ctypes.c_int32), ("steps", ctypes.c_int32),
        ("expected_scores", ctypes.c_double * 4),
    ]


class _Functions(ctypes.Structure):
    _fields_ = [("is_win", ctypes.c_void_p), ("choose", ctypes.c_void_p),
                ("peng", ctypes.c_void_p), ("gang", ctypes.c_void_p)]


class _Params(ctypes.Structure):
    _fields_ = [("sw", ctypes.c_double), ("uw", ctypes.c_double),
                ("cw", ctypes.c_double), ("rw", ctypes.c_double),
                ("cont_max", ctypes.c_int32)]


@dataclass(frozen=True, slots=True)
class FastRolloutResult:
    scores: tuple[float, ...]
    hero_score: float
    winner: int | None
    draw: bool
    steps: int
    actual_scores: tuple[int, ...]
    wall_remaining: int
    final_state: dict


def supports(policies="v31") -> bool:
    if isinstance(policies, Mapping):
        items = [policies.get(seat, "v31") for seat in range(4)]
    elif isinstance(policies, Sequence) and not isinstance(policies, str):
        if len(policies) != 4:
            return False
        items = policies
    else:
        items = [policies] * 4
    return all(isinstance(policy, str) and policy in {"v31", "v31n"} for policy in items)


def _load():
    global _LIB, _NATIVE_LIB, _FUNCTIONS
    if _LIB is not None:
        return _LIB
    flags = ("-O3", "-fPIC", "-shared", "-std=c11")
    compiler = os.environ.get("CC", "cc")
    identity = b"\0".join((
        _SOURCE.read_bytes(), platform.system().encode(), platform.machine().encode(),
        compiler.encode(), " ".join(flags).encode(),
    ))
    digest = hashlib.sha256(identity).hexdigest()[:20]
    target = _BUILD / f"libsearch159_{digest}.so"
    if not target.exists():
        _BUILD.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f"{target.name}.{os.getpid()}.{random.getrandbits(32):08x}.tmp")
        try:
            subprocess.run([compiler, *flags, "-o", str(temporary), str(_SOURCE)], check=True,
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            os.replace(temporary, target)
        except subprocess.CalledProcessError as error:
            raise RuntimeError(f"search159 C compilation failed: {error.stderr.strip()}") from error
        finally:
            if temporary.exists():
                temporary.unlink()
    lib = ctypes.CDLL(str(target))
    lib.search159_abi_version.restype = ctypes.c_int
    lib.search159_sizeof_state.restype = ctypes.c_size_t
    lib.search159_sizeof_params.restype = ctypes.c_size_t
    if (lib.search159_abi_version() != 1 or
            lib.search159_sizeof_state() != ctypes.sizeof(_State) or
            lib.search159_sizeof_params() != ctypes.sizeof(_Params)):
        raise RuntimeError("search159 C/Python structure ABI mismatch")
    lib.search159_rollout.argtypes = [ctypes.POINTER(_State), ctypes.POINTER(_Functions),
                                      ctypes.POINTER(_Params), ctypes.c_int]
    lib.search159_rollout.restype = ctypes.c_int
    from ...native import native

    native_lib = native.lib()
    functions = _Functions(*[
        ctypes.cast(getattr(native_lib, name), ctypes.c_void_p).value
        for name in ("mj_is_win", "mj_choose_discard_v10", "mj_decide_peng", "mj_decide_gang")
    ])
    _NATIVE_LIB = native_lib  # Keep the DLL alive as long as its C pointers exist.
    _FUNCTIONS = functions
    _LIB = lib
    return lib


def warmup():
    """Compile/load once before a timed benchmark or latency-sensitive search."""
    _load()


def _seat(seat):
    if not isinstance(seat, int) or not 0 <= seat < 4:
        raise ValueError(f"invalid seat: {seat}")
    return seat


def _tile(tile):
    if not isinstance(tile, int) or not 0 <= tile < 28:
        raise ValueError(f"invalid tile: {tile}")
    return tile


def _nullable(value):
    return -1 if value is None else int(value)


def _serialize(game):
    if game.phase not in _PHASES:
        raise ValueError("fast rollout requires a stable decision or terminal state")
    if len(game.players) != 4 or len(game.wall) > 112 or len(game.gang_records) > 16:
        raise ValueError("state exceeds the physical game capacities")
    output = _State()
    counts = [0] * 28
    for seat, source in enumerate(game.players):
        target = output.players[seat]
        if len(source.discards) > 112 or len(source.melds) > 4:
            raise ValueError("player exceeds the physical game capacities")
        for tile in source.hand:
            target.hand[_tile(tile)] += 1
            counts[tile] += 1
        for i, tile in enumerate(source.discards):
            target.discards[i] = _tile(tile)
            counts[tile] += 1
        target.n_discards = len(source.discards)
        target.n_melds = len(source.melds)
        target.score = source.score_delta
        for i, meld in enumerate(source.melds):
            tile = _tile(meld["tile"])
            if tile == 27 or meld["type"] not in {"peng", "gang"}:
                raise ValueError("invalid exposed meld")
            kind = _KINDS.get(meld.get("kind"), -1)
            target.melds[i] = _Meld(tile, int(meld["type"] == "gang"), kind,
                                    _nullable(meld.get("wr")))
            counts[tile] += 3 if meld["type"] == "peng" else 4
    for i, tile in enumerate(game.wall):
        output.wall[i] = _tile(tile)
        counts[tile] += 1
    if counts != [4] * 28:
        raise ValueError("fast rollout input violates total tile conservation")
    output.head = 0
    output.tail = len(game.wall)
    for i, record in enumerate(game.gang_records):
        kind = _KINDS[record["kind"]]
        source = _seat(record["from"]) if kind == 0 else -1
        output.gangs[i] = _Gang(_seat(record["seat"]), kind, _tile(record["tile"]), source)
    output.n_gangs = len(game.gang_records)
    output.phase = _PHASES[game.phase]
    output.turn = _seat(game.turn)
    output.last_discard = -1 if game.last_discard is None else _tile(game.last_discard)
    output.last_discarder = -1 if game.last_discarder is None else _seat(game.last_discarder)
    for i, (seat, actions) in enumerate(game.pending_actions.items()):
        _seat(seat)
        tile = game.last_discard
        if (game.phase != "react_wait" or tile is None or not 0 <= tile < 27 or
                seat == game.last_discarder or
                not (actions.get("peng") or actions.get("gang")) or
                (actions.get("peng") and output.players[seat].hand[tile] < 2) or
                (actions.get("gang") and output.players[seat].hand[tile] < 3)):
            raise ValueError("pending response is inconsistent with the physical hand")
        output.pending_order[i] = seat
        output.pending[seat] = int(bool(actions.get("peng"))) | (int(bool(actions.get("gang"))) << 1)
    output.n_pending = len(game.pending_actions)
    if output.phase == 1 and (output.n_pending == 0 or output.last_discard < 0 or output.last_discarder < 0):
        raise ValueError("reaction state is missing its pending action or discard")
    output.winner = -1 if game.winner is None else _seat(game.winner)
    output.win_tile = -1 if game.win_tile is None else _tile(game.win_tile)
    output.win_kind = _WIN_KINDS[game.win_kind]
    output.huangzhuang = int(game.huangzhuang)
    output.n_159 = game.n_159
    if len(game.fan_159) > 6:
        raise ValueError("too many flip tiles")
    output.n_fan = len(game.fan_159)
    for i, tile in enumerate(game.fan_159):
        output.fan_159[i] = _tile(tile)
    if game.last_drawn is None:
        output.last_drawn_seat = output.last_drawn_tile = -1
    else:
        output.last_drawn_seat = _seat(game.last_drawn["seat"])
        output.last_drawn_tile = _tile(game.last_drawn["tile"])
    return output


def _params(game):
    from ..bot_v10 import Bot as V10Bot

    # Exactly the same constructor supplies NativeV31's actual env overrides.
    bots = [V10Bot(game, seat) for seat in range(4)]
    return (_Params * 4)(*[
        _Params(bot.shanten_weight, bot.ukeire_weight, bot.cont_weight,
                bot.risk_weight, bot.cont_max_shanten) for bot in bots
    ])


def _decode(output, game):
    players = []
    for seat, source in enumerate(output.players):
        melds = []
        for i in range(source.n_melds):
            m = source.melds[i]
            meld = {"type": "peng" if m.type == 0 else "gang", "tile": m.tile}
            if m.kind >= 0:
                meld["kind"] = _KIND_NAMES[m.kind]
            if m.wr >= 0:
                meld["wr"] = m.wr
            melds.append(meld)
        players.append({
            "seat": seat,
            "hand": [tile for tile, count in enumerate(source.hand) for _ in range(count)],
            "melds": melds,
            "discards": list(source.discards[:source.n_discards]),
            "score_delta": source.score,
        })
    gangs = []
    for i in range(output.n_gangs):
        g = output.gangs[i]
        record = {"seat": g.seat, "kind": _KIND_NAMES[g.kind], "tile": g.tile}
        if g.kind == 0:
            record["from"] = g.from_seat
        gangs.append(record)
    pending = {}
    for i in range(output.n_pending):
        seat = output.pending_order[i]
        flags = output.pending[seat]
        pending[seat] = {"peng": bool(flags & 1), "gang": bool(flags & 2)}
    last_action = game.last_action
    if output.last_action_kind == 1:
        last_action = f"座位{output.last_action_seat} 打出 {tile_name(output.last_action_tile)}"
    elif output.last_action_kind == 2:
        last_action = f"座位{output.last_action_seat} 摸牌"
    elif output.last_action_kind == 3:
        last_action = f"座位{output.last_action_seat} 杠后补牌"
    return {
        "wall": list(output.wall[output.head:output.tail]),
        "players": players,
        "turn": output.turn, "phase": _PHASE_NAMES[output.phase],
        "last_discard": None if output.last_discard < 0 else output.last_discard,
        "last_discarder": None if output.last_discarder < 0 else output.last_discarder,
        "pending_actions": pending,
        "gang_records": gangs,
        "winner": None if output.winner < 0 else output.winner,
        "win_tile": None if output.win_tile < 0 else output.win_tile,
        "win_kind": _WIN_NAMES[output.win_kind],
        "huangzhuang": bool(output.huangzhuang),
        "n_159": output.n_159, "fan_159": list(output.fan_159[:output.n_fan]),
        "last_drawn": None if output.last_drawn_seat < 0 else {
            "seat": output.last_drawn_seat, "tile": output.last_drawn_tile},
        "last_action": last_action,
    }


def _write_back(game, state):
    for seat, data in enumerate(state["players"]):
        player = game.players[seat]
        player.hand = list(data["hand"])
        player.melds = [dict(m) for m in data["melds"]]
        player.discards = list(data["discards"])
        player.score_delta = data["score_delta"]
    for name, value in state.items():
        if name != "players":
            setattr(game, name, value)


def rollout_fast(game, hero: int, policies="v31", *, mutate=False, max_steps=500) -> FastRolloutResult:
    """Continue a complete sampled world in C, optionally updating its Game.

    `mutate=True` restores every rule-relevant final field, including exact
    physical hands/discards/melds/wall, actual flip scores and gang records. The
    existing engine log is left as a prefix; C does not allocate per-action log
    strings. The final_state dict likewise intentionally contains no log or RNG.
    """
    _seat(hero)
    if not supports(policies):
        raise ValueError("fast rollout supports only four-seat v31/v31n continuations")
    if not isinstance(max_steps, int) or max_steps < 0:
        raise ValueError("max_steps must be a nonnegative integer")
    lib = _load()
    data = _serialize(game)
    params = _params(game)
    rc = lib.search159_rollout(ctypes.byref(data), ctypes.byref(_FUNCTIONS), params, max_steps)
    if rc:
        detail = {1: "invalid state", 2: "action guard reached before terminal state",
                  3: "invalid continuation action", 4: "physical capacity exceeded"}.get(rc, str(rc))
        raise RuntimeError(f"fast search159 rollout failed: {detail}")
    final = _decode(data, game)
    scores = tuple(data.expected_scores)
    result = FastRolloutResult(
        scores=scores, hero_score=scores[hero], winner=final["winner"],
        draw=final["winner"] is None, steps=data.steps,
        actual_scores=tuple(p.score for p in data.players),
        wall_remaining=data.tail - data.head, final_state=final,
    )
    if mutate:
        _write_back(game, final)
    return result
