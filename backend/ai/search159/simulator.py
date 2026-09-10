"""Faithful Game transitions and a deliberately limited search observation.

The simulator uses the production Game state machine. No alternate rules engine
or synthetic reward is introduced. Only sample_world sees its own generated
hidden hands and wall; its input is an immutable public-information observation.

The initial belief is uniform subject to tile conservation and known hand sizes.
It is not a posterior conditioned on opponents' complete action histories.
"""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from ...game.engine import Game, Player
from ...rules.tiles import is_159, tiles_from_counts

RED = 27
PublicEvent = tuple[int, str, int]


@dataclass(frozen=True, slots=True, order=True)
class Action:
    """A legal game action. Claims normally carry the claimed tile as well."""

    kind: str
    tile: int = -1

    def __post_init__(self):
        if self.kind not in {"discard", "peng", "gang", "pass"}:
            raise ValueError(f"unknown action kind: {self.kind}")
        if not isinstance(self.tile, int) or not -1 <= self.tile < 28:
            raise ValueError(f"invalid tile: {self.tile}")
        if self.kind == "pass" and self.tile != -1:
            raise ValueError("pass has no tile")


@dataclass(frozen=True, slots=True)
class Meld:
    type: str
    tile: int
    kind: str | None = None
    wr: int | None = None

    @classmethod
    def from_dict(cls, value):
        return cls(value["type"], value["tile"], value.get("kind"), value.get("wr"))

    def to_dict(self):
        value = {"type": self.type, "tile": self.tile}
        if self.kind is not None:
            value["kind"] = self.kind
        if self.wr is not None:
            value["wr"] = self.wr
        return value


@dataclass(frozen=True, slots=True)
class GangRecord:
    seat: int
    kind: str
    tile: int
    from_seat: int | None = None

    @classmethod
    def from_dict(cls, value):
        return cls(value["seat"], value["kind"], value["tile"], value.get("from"))

    def to_dict(self):
        value = {"seat": self.seat, "kind": self.kind, "tile": self.tile}
        if self.from_seat is not None:
            value["from"] = self.from_seat
        return value


@dataclass(frozen=True, slots=True)
class Observation:
    """Everything the focal player may use as a search node key.

    Other players' pending legality, concealed tiles and drawn tiles, wall order,
    engine log strings, and engine RNG state are deliberately absent. Hand sizes
    and exposed melds are public. `passed_seats` is supplied by the caller only
    when an already completed response is actually known; observe never infers it
    from the private pending_actions dictionary.
    """

    hero: int
    hand_counts: tuple[int, ...]
    discards: tuple[tuple[int, ...], ...]
    melds: tuple[tuple[Meld, ...], ...]
    hand_sizes: tuple[int, ...]
    dealer: int
    turn: int
    phase: str
    wall_length: int
    last_discard: int | None
    last_discarder: int | None
    gang_records: tuple[GangRecord, ...]
    own_legal_actions: tuple[Action, ...]
    passed_seats: tuple[int, ...] = ()
    own_drawn_tile: int | None = None
    winner: int | None = None

    def key(self):
        """The immutable observation itself is a hashable information-set key."""
        return self

    @property
    def wall_remaining(self):
        return self.wall_length


@dataclass(frozen=True, slots=True)
class RolloutResult:
    scores: tuple[float, ...]
    hero_score: float
    winner: int | None
    draw: bool
    steps: int


def _check_seat(seat):
    if not isinstance(seat, int) or not 0 <= seat < 4:
        raise ValueError(f"invalid seat: {seat}")


def acting_seat(game: Game) -> int | None:
    """Internal engine scheduler; do not expose other claimants as observations."""
    if game.phase == "game_over":
        return None
    if game.phase == "discard_wait":
        return game.turn
    if game.phase == "react_wait":
        if not game.pending_actions:
            raise ValueError("react_wait requires at least one pending response")
        return next(iter(game.pending_actions))
    raise ValueError(f"not a stable decision phase: {game.phase}")


def _claim_actions(game: Game, seat: int) -> tuple[Action, ...]:
    """Read this player's offered actions only, not other players' eligibility."""
    offered = game.pending_actions.get(seat)
    if not offered:
        return ()
    out = [Action("pass")]
    if offered.get("peng"):
        out.append(Action("peng", game.last_discard))
    if offered.get("gang"):
        out.append(Action("gang", game.last_discard))
    return tuple(out)


def legal_actions(game: Game, seat: int) -> tuple[Action, ...]:
    """All actions for the current actor; an inactive seat has no actions."""
    _check_seat(seat)
    if acting_seat(game) != seat:
        return ()
    if game.phase == "react_wait":
        return _claim_actions(game, seat)
    counts = game.players[seat].hand_counts
    discards = tuple(Action("discard", t) for t, n in enumerate(counts) if n > 0)
    gangs = tuple(Action("gang", t) for t in game._gang_options(seat))
    return discards + gangs


def observe(game: Game, hero: int, *, passed_seats=()) -> Observation:
    """Capture a player's information without calling Game.public_state().

    Game.public_state currently includes all private reaction options and may
    contain another player's drawn tile, so it is not a safe search boundary.
    For reactions we read only this player's own offered options; opponents'
    pending entries never influence this observation or its key.
    """
    _check_seat(hero)
    passed = tuple(sorted(set(passed_seats)))
    for seat in passed:
        _check_seat(seat)
    if game.phase == "discard_wait" and game.turn == hero:
        own_actions = legal_actions(game, hero)
    elif game.phase == "react_wait":
        own_actions = _claim_actions(game, hero)
    else:
        own_actions = ()
    drawn = game.last_drawn
    own_drawn = drawn["tile"] if drawn is not None and drawn["seat"] == hero else None
    return Observation(
        hero=hero,
        hand_counts=tuple(game.players[hero].hand_counts),
        discards=tuple(tuple(p.discards) for p in game.players),
        melds=tuple(tuple(Meld.from_dict(m) for m in p.melds) for p in game.players),
        hand_sizes=tuple(len(p.hand) for p in game.players),
        dealer=game.dealer,
        turn=game.turn,
        phase=game.phase,
        wall_length=game.wall_remaining(),
        last_discard=game.last_discard,
        last_discarder=game.last_discarder,
        gang_records=tuple(GangRecord.from_dict(r) for r in game.gang_records),
        own_legal_actions=own_actions,
        passed_seats=passed,
        own_drawn_tile=own_drawn,
        winner=game.winner,
    )


def clone(game: Game, *, keep_log: bool = True) -> Game:
    """Copy the production state without redealing or aliasing its mutable data."""
    result = object.__new__(Game)
    result.__dict__ = game.__dict__.copy()
    result.wall = list(game.wall)
    result.players = []
    for source in game.players:
        player = object.__new__(Player)
        player.__dict__ = source.__dict__.copy()
        player.hand = list(source.hand)
        player.discards = list(source.discards)
        player.melds = [dict(m) for m in source.melds]
        result.players.append(player)
    result.pending_actions = {s: dict(actions) for s, actions in game.pending_actions.items()}
    result.gang_records = [dict(record) for record in game.gang_records]
    result.fan_159 = list(game.fan_159)
    result.log = list(game.log) if keep_log else []
    result.last_drawn = None if game.last_drawn is None else dict(game.last_drawn)
    result.rng = random.Random(0)
    result.rng.setstate(game.rng.getstate())
    return result


def _unseen_pool(observation: Observation) -> list[int]:
    if len(observation.hand_counts) != 28 or len(observation.hand_sizes) != 4:
        raise ValueError("observation must contain 28 tile types and four seats")
    if len(observation.discards) != 4 or len(observation.melds) != 4:
        raise ValueError("observation must contain four public player records")
    if any(not 0 <= n <= 4 for n in observation.hand_counts):
        raise ValueError("invalid focal hand counts")
    if sum(observation.hand_counts) != observation.hand_sizes[observation.hero]:
        raise ValueError("focal hand length disagrees with its counts")
    visible = list(observation.hand_counts)
    for discards, melds in zip(observation.discards, observation.melds):
        for tile in discards:
            if not 0 <= tile < 28:
                raise ValueError("invalid public discard")
            visible[tile] += 1
        for meld in melds:
            if meld.type not in {"peng", "gang"} or not 0 <= meld.tile < 27:
                raise ValueError("invalid public meld")
            visible[meld.tile] += 3 if meld.type == "peng" else 4
    if any(n > 4 for n in visible):
        raise ValueError("public observation violates the four-copy tile limit")
    pool = [t for t, n in enumerate(visible) for _ in range(4 - n)]
    hidden_hands = sum(n for seat, n in enumerate(observation.hand_sizes)
                       if seat != observation.hero)
    if any(n < 0 for n in observation.hand_sizes) or observation.wall_length < 0:
        raise ValueError("negative hand or wall length")
    if len(pool) != hidden_hands + observation.wall_length:
        raise ValueError("observation violates total tile conservation")
    return pool


def _empty_from_observation(observation: Observation) -> Game:
    """Construct from the allowlist, never by cloning a source hidden Game."""
    game = object.__new__(Game)
    game.rng = random.Random(0)  # The engine does not use RNG after dealing.
    game.wall = []
    game.dealer = observation.dealer
    game.human_seat = -1
    game.players = [Player(seat, is_bot=True) for seat in range(4)]
    for seat, player in enumerate(game.players):
        player.discards = list(observation.discards[seat])
        player.melds = [m.to_dict() for m in observation.melds[seat]]
    game.players[observation.hero].hand = tiles_from_counts(observation.hand_counts)
    game.turn = observation.turn
    game.phase = observation.phase
    game.last_discard = observation.last_discard
    game.last_discarder = observation.last_discarder
    game.pending_actions = {}
    game.winner = None
    game.win_tile = None
    game.win_kind = None
    game.fan_159 = []
    game.n_159 = 0
    game.huangzhuang = False
    game.gang_records = [record.to_dict() for record in observation.gang_records]
    game.log = []
    game.last_action = ""
    game.last_drawn = (None if observation.own_drawn_tile is None else
                       {"seat": observation.hero, "tile": observation.own_drawn_tile})
    return game


def sample_world(observation: Observation, rng: random.Random) -> Game:
    """Uniformly allocate all unseen tiles jointly to opponents and the wall.

    Reactions are reconstructed from the sampled hands. If the focal player has
    a response, seats below it have already been considered by the engine's
    ascending-seat scheduler, so they are not offered a response again. This
    does *not* infer that their hands lack pairs, or that an unasked player passed.
    Explicit passed_seats likewise removes offers without imposing policy beliefs.

    For a non-acting observer at react_wait, the public phase implies that some
    remaining response exists. We rejection-sample that condition; searches
    normally call this function only at the focal player's own decision.
    """
    _check_seat(observation.hero)
    if observation.phase not in {"discard_wait", "react_wait"}:
        raise ValueError("sample_world requires a live decision observation")
    if observation.hero in observation.passed_seats and observation.own_legal_actions:
        raise ValueError("focal player cannot be both passed and currently offered")
    pool0 = _unseen_pool(observation)
    if observation.own_drawn_tile is not None:
        tile = observation.own_drawn_tile
        if not 0 <= tile < 28 or observation.hand_counts[tile] <= 0:
            raise ValueError("known focal drawn tile is absent from its hand")
    for _attempt in range(4096):
        pool = list(pool0)
        rng.shuffle(pool)
        game = _empty_from_observation(observation)
        pos = 0
        for seat, player in enumerate(game.players):
            if seat == observation.hero:
                continue
            count = observation.hand_sizes[seat]
            player.hand = sorted(pool[pos:pos + count])
            pos += count
        game.wall = pool[pos:]
        if observation.phase == "discard_wait":
            if game.turn == observation.hero:
                if legal_actions(game, observation.hero) != observation.own_legal_actions:
                    raise ValueError("focal discard actions disagree with the observation")
            return game
        tile = observation.last_discard
        if tile is None or tile == RED or observation.last_discarder is None:
            raise ValueError("reaction observation needs an ordinary last discard")
        has_focal_offer = bool(observation.own_legal_actions)
        for seat, player in enumerate(game.players):
            if seat == observation.last_discarder or seat in observation.passed_seats:
                continue
            if has_focal_offer and seat < observation.hero:
                continue
            count = player.hand.count(tile)
            if count >= 2:
                game.pending_actions[seat] = {"peng": True, "gang": count >= 3}
        if has_focal_offer:
            if (acting_seat(game) != observation.hero or
                    _claim_actions(game, observation.hero) != observation.own_legal_actions):
                raise ValueError("focal reaction actions disagree with its own hand")
            return game
        if game.pending_actions:
            return game
    raise ValueError("no sampled world supports the observed pending reaction")


def apply_action(game: Game, action: Action, seat: int | None = None):
    """Validate once and delegate the transition to the production engine."""
    actor = acting_seat(game)
    if actor is None:
        raise ValueError("the game has already ended")
    if seat is None:
        seat = actor
    if actor != seat:
        raise ValueError(f"seat {seat} is not the current actor")
    # Compatibility with the engine's tile-free reaction methods.
    if game.phase == "react_wait" and action.kind in {"peng", "gang"} and action.tile == -1:
        action = Action(action.kind, game.last_discard)
    if action not in legal_actions(game, seat):
        raise ValueError(f"illegal action for seat {seat}: {action}")
    if action.kind == "discard":
        return game.action_discard(seat, action.tile)
    if action.kind == "peng":
        return game.action_peng(seat)
    if action.kind == "gang":
        return game.action_gang(seat, action.tile if game.phase == "discard_wait" else None)
    return game.action_pass(seat)


class _NativeHV:
    """Native scholar adapter; every method rebuilds its own public context."""

    def __init__(self, game, seat):
        self.game, self.seat = game, seat

    def _set(self):
        from ...native import native

        hand = self.game.players[self.seat].hand_counts
        visible = list(hand)
        for player in self.game.players:
            for tile in player.discards:
                visible[tile] += 1
            for meld in player.melds:
                visible[meld["tile"]] += 3 if meld["type"] == "peng" else 4
        lib = native.lib()
        lib.mj_hv_set2(native._i8(hand), native._i8(visible), 1.0, 0, 2, 1, 6)
        return lib

    def choose_discard(self):
        return self._set().mj_hv_choose_discard()

    def decide_peng(self, tile):
        return bool(self._set().mj_hv_decide_peng(tile))

    def decide_gang(self, tile, kind):
        return bool(self._set().mj_hv_decide_gang(tile, {"ming": 0, "an": 1, "bu": 2}[kind]))


def _make_bot(game, seat, policy):
    if not isinstance(policy, str):
        if isinstance(policy, type) or (callable(policy) and not hasattr(policy, "choose_discard")):
            return policy(game, seat)
        return policy
    if policy in {"v31", "v31n", "v10", "v10n", "v1", "v1n"}:
        from ..bot_native import NativeV1, NativeV10, NativeV31

        cls = NativeV1 if policy in {"v1", "v1n"} else (
            NativeV10 if policy in {"v10", "v10n"} else NativeV31)
        return cls(game, seat)
    if policy in {"hv", "scholar"}:
        return _NativeHV(game, seat)
    raise ValueError(f"unsupported information-limited policy: {policy}")


def _action_from_bot(game, seat, bot):
    actions = legal_actions(game, seat)
    if not actions:
        raise ValueError(f"seat {seat} has no current decision")
    if game.phase == "discard_wait":
        for action in actions:
            if action.kind == "gang":
                kind = "an" if game.players[seat].hand.count(action.tile) == 4 else "bu"
                if bot.decide_gang(action.tile, kind):
                    return action
        action = Action("discard", int(bot.choose_discard()))
    else:
        tile = game.last_discard
        gang = Action("gang", tile)
        peng = Action("peng", tile)
        if gang in actions and bot.decide_gang(tile, "ming"):
            return gang
        if peng in actions and bot.decide_peng(tile):
            return peng
        action = Action("pass")
    if action not in actions:
        raise ValueError(f"continuation policy returned an illegal action: {action}")
    return action


def base_action(game: Game, seat: int, policy="v31") -> Action:
    """One complete base-policy decision, including all three kinds of gang."""
    return _action_from_bot(game, seat, _make_bot(game, seat, policy))


def _policy_bots(game, policies):
    if isinstance(policies, Mapping):
        selected = [policies.get(seat, "v31") for seat in range(4)]
    elif isinstance(policies, Sequence) and not isinstance(policies, str):
        if len(policies) != 4:
            raise ValueError("policies must contain exactly four seats")
        selected = list(policies)
    else:
        selected = [policies] * 4
    return [_make_bot(game, seat, selected[seat]) for seat in range(4)]


def expected_scores(game: Game) -> tuple[float, ...]:
    """Terminal net scores with only the six flip tiles integrated out.

    Conditional on the remaining tile multiset, unused wall order is exchangeable
    for information-limited policies. The expected multiplier is 1 + 6*M/W.
    When fewer than six tiles remain, Game._hu specifies multiplier 1 instead.
    Drawn games settle neither win scores nor earlier gang records.
    """
    if game.phase != "game_over":
        raise ValueError("scores require a terminal game")
    if game.winner is None:
        return (0.0, 0.0, 0.0, 0.0)
    scores = [0.0] * 4
    for record in game.gang_records:
        seat = record["seat"]
        scores[seat] += 3.0
        if record["kind"] == "ming":
            scores[record["from"]] -= 3.0
        else:
            for other in range(4):
                if other != seat:
                    scores[other] -= 1.0
    count = len(game.wall)
    multiplier = 1.0 + (6.0 * sum(is_159(t) for t in game.wall) / count if count >= 6 else 0.0)
    for seat in range(4):
        scores[seat] += multiplier * (3.0 if seat == game.winner else -1.0)
    return tuple(scores)


def advance(game: Game, hero: int, policies="v31", *, max_steps=500) -> tuple[PublicEvent, ...]:
    """Advance opponents to the focal player's next decision or game over.

    The returned trace includes only visible discard/peng/gang actions. It never
    reveals another player's pass eligibility or privately drawn tile.
    """
    _check_seat(hero)
    bots = _policy_bots(game, policies)
    trace = []
    for _step in range(max_steps):
        seat = acting_seat(game)
        if seat is None or seat == hero:
            return tuple(trace)
        action = _action_from_bot(game, seat, bots[seat])
        if action.kind != "pass":
            trace.append((seat, action.kind, action.tile))
        apply_action(game, action, seat)
    if acting_seat(game) in {None, hero}:
        return tuple(trace)
    raise RuntimeError("advance exceeded its action guard before the next focal decision")


def rollout(game: Game, hero: int, policies="v31", *, max_steps=500) -> RolloutResult:
    """Mutate a sampled game through the complete production action protocol."""
    _check_seat(hero)
    bots = _policy_bots(game, policies)
    steps = 0
    while game.phase != "game_over" and steps < max_steps:
        seat = acting_seat(game)
        apply_action(game, _action_from_bot(game, seat, bots[seat]), seat)
        steps += 1
    if game.phase != "game_over":
        raise RuntimeError("rollout exceeded its action guard without a terminal result")
    scores = expected_scores(game)
    return RolloutResult(scores, scores[hero], game.winner, game.winner is None, steps)
