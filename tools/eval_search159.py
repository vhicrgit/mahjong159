"""Fixed-sample paired evaluation, with all four hero seats clustered by deal.

Example:
  python -m tools.eval_search159 --a v31 --b search --opp v31 \
      --seeds 256 --seed0 5100000 --workers 4 --out work/search_eval.json

Every seed runs both arms in every seat. All players may peng or take ming,
an, and bu gangs. Results use Game.score_delta, including actual 159 flips;
search-internal expected scores and shaped rewards never enter the report.
Confidence intervals treat the four seat pairs of one deal as one cluster.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import hashlib
import math
import multiprocessing as mp
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


KINDS = ("v31", "hv", "search", "root", "finite")
_CORE_SOURCES = (
    "backend/ai/search159/planner.py", "backend/ai/bot_search159.py",
    "backend/ai/search159/simulator.py", "backend/native/mj159.c",
    "backend/analysis/finite_horizon.py", "backend/native/finite_horizon.c",
    "backend/ai/search159/fast_rollout.py", "backend/ai/search159/fast_rollout.c",
    "backend/native/native.py", "backend/game/engine.py",
    "backend/ai/bot_native.py", "mobile/wasm/hv_engine_inc.c",
    "backend/ai/bot_v10.py", "backend/ai/bot_v31.py",
    "tools/eval_search159.py",
)


def _valid_git_revision(value):
    """Only complete SHA-1/SHA-256 object names support a revision comparison."""
    return (isinstance(value, str) and len(value) in (40, 64)
            and all(c in "0123456789abcdefABCDEF" for c in value))


def compare_source_versions(before, after):
    """Distinguish an observed change from an incomplete revision check.

    A missing Git lookup does not establish either a revision change or stable
    revision. Source/environment comparisons remain available independently.
    Query diagnostics are audit data, not part of the algorithm fingerprint.
    """
    source_changed = before.get("source_sha256") != after.get("source_sha256")
    environment_changed = before.get("policy_environment") != after.get("policy_environment")
    before_head, after_head = before.get("git_head"), after.get("git_head")
    revision_complete = _valid_git_revision(before_head) and _valid_git_revision(after_head)
    revision_changed = (before_head.lower() != after_head.lower()) if revision_complete else None
    return {
        "source_changed_during_run": source_changed or environment_changed or revision_changed is True,
        "git_revision_verification_complete": revision_complete,
        "source_change_details": {"source_sha256_changed": source_changed,
                                  "policy_environment_changed": environment_changed,
                                  "git_revision_changed": revision_changed},
    }


def source_fingerprint():
    """Identify uncommitted algorithm revisions as well as the repository HEAD."""
    repo = Path(__file__).resolve().parents[1]
    hashes = {}
    for relative in _CORE_SOURCES:
        path = repo / relative
        hashes[relative] = hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None
    head, git_error = None, None
    try:
        git = subprocess.run(["git", "rev-parse", "--verify", "HEAD"], cwd=repo,
                             capture_output=True, text=True, timeout=5, check=True)
        candidate = git.stdout.strip()
        if _valid_git_revision(candidate):
            head = candidate.lower()
        else:
            git_error = {"type": "InvalidRevision",
                         "message": "git rev-parse returned no valid full revision",
                         "stdout": candidate[:2000]}
    except (OSError, subprocess.SubprocessError) as exc:
        git_error = {"type": type(exc).__name__, "message": str(exc)[:2000]}
        if isinstance(exc, subprocess.TimeoutExpired):
            git_error["timeout_seconds"] = exc.timeout
        if isinstance(exc, subprocess.CalledProcessError):
            git_error["returncode"] = exc.returncode
            stderr = exc.stderr or ""
            if isinstance(stderr, bytes):
                stderr = stderr.decode("utf-8", errors="replace")
            git_error["stderr"] = str(stderr)[:2000]
    policy_env = {key: os.environ.get(key) for key in (
        "V10_SHANTEN_W", "V10_UKEIRE_W", "V10_CONT_W", "V10_RISK_W",
        "V10_CONT_MAX_SH", "MJ_NATIVE_MARCH")}
    return {"git_head": head, "source_sha256": hashes,
            "policy_environment": policy_env,
            "git_revision_check": {"status": "verified" if head else "unavailable",
                                   "error": git_error}}


@dataclass(frozen=True)
class EvalConfig:
    a: str = "v31"
    b: str = "v31"
    opp: str = "v31"
    seeds: int = 32
    seed0: int = 5100000
    workers: int = 1
    simulations: int = 64
    depth: int = 2
    candidate_limit: int = 4
    confirmation: int | None = None
    confidence_z: float = 0.0
    max_search_shanten: int | None = None
    finite_horizon: int = 2
    finite_discount: float = 1.0
    opponent_policy: str = "v31"
    tree_key: str = "history"
    paired_future: bool = False
    future_candidate_limit: int = 2
    max_steps: int = 1000


def search_seed(game_seed: int, hero: int) -> int:
    """Common algorithm randomness across arms, stable across processes/runs."""
    return ((game_seed * 6364136223846793005) ^ ((hero + 1) * 1442695040888963407)) & ((1 << 64) - 1)


class UnifiedRule:
    def __init__(self, game, seat, policy):
        self.game, self.seat, self.policy = game, seat, policy

    def choose_action(self):
        from backend.ai.search159.simulator import base_action
        return base_action(self.game, self.seat, policy=self.policy)


def _make_bot(kind, game, seat, config, rng_seed):
    if kind == "v31":
        from backend.ai.bot_native import NativeV31
        return NativeV31(game, seat)
    if kind == "hv":
        return UnifiedRule(game, seat, "hv")
    from backend.ai.bot_search159 import Bot
    return Bot(game, seat, simulations=config.simulations,
               depth=1 if kind == "root" else config.depth,
               candidate_limit=config.candidate_limit, seed=rng_seed,
               confirmation=config.confirmation, confidence_z=config.confidence_z,
               max_search_shanten=config.max_search_shanten,
               finite_horizon=config.finite_horizon,
               finite_discount=config.finite_discount,
               opponent_policy=config.opponent_policy,
               tree_key=config.tree_key, paired_future=config.paired_future,
               future_candidate_limit=config.future_candidate_limit,
               mode={"search": "tree", "root": "root", "finite": "finite"}[kind])


def _choose_action(game, seat, bot):
    """Call unified search once; adapt legacy rules to the same complete API."""
    from backend.ai.search159.simulator import Action
    choose = getattr(bot, "choose_action", None)
    if choose is not None:
        return choose()
    if game.phase == "discard_wait":
        for tile in game._gang_options(seat):
            kind = "an" if game.players[seat].hand.count(tile) == 4 else "bu"
            if bot.decide_gang(tile, kind):
                return Action("gang", tile)
        return Action("discard", bot.choose_discard())
    if game.phase == "react_wait":
        pending = game.pending_actions[seat]
        tile = game.last_discard
        if pending.get("gang") and bot.decide_gang(tile, "ming"):
            return Action("gang", tile)
        if pending.get("peng") and bot.decide_peng(tile):
            return Action("peng", tile)
        return Action("pass")
    raise RuntimeError(f"Unexpected active phase: {game.phase!r}")


def _action_label(game, seat, action):
    if action.kind != "gang":
        return action.kind
    if game.phase == "react_wait":
        return "ming_gang"
    return "an_gang" if game.players[seat].hand.count(action.tile) == 4 else "bu_gang"


def play_game(game_seed: int, hero: int, kind: str, config: EvalConfig) -> dict:
    from backend.game.engine import Game
    from backend.ai.search159.simulator import acting_seat, apply_action, base_action, legal_actions

    started = time.perf_counter()
    game = Game(seed=game_seed, human_seat=-1)
    rng_seed = search_seed(game_seed, hero)
    bots = {s: _make_bot(kind if s == hero else config.opp, game, s,
                        config, rng_seed if s == hero else search_seed(game_seed, s))
            for s in range(4)}
    all_actions, hero_actions = Counter(), Counter()
    n_decisions = n_changes = steps = 0
    decision_seconds = reference_seconds = 0.0
    while game.phase != "game_over":
        if steps >= config.max_steps:
            raise RuntimeError(f"Nonterminal game at step limit: seed={game_seed}, hero={hero}, arm={kind}")
        seat = acting_seat(game)
        if seat is None:
            raise RuntimeError(f"No actor in nonterminal phase {game.phase!r}")
        t0 = time.perf_counter()
        action = _choose_action(game, seat, bots[seat])
        decision_time = time.perf_counter() - t0
        if action not in legal_actions(game, seat):
            raise RuntimeError(f"Illegal action {action!r}: seed={game_seed}, seat={seat}, arm={kind}")
        label = _action_label(game, seat, action)
        all_actions[label] += 1
        if seat == hero:
            n_decisions += 1
            decision_seconds += decision_time
            hero_actions[label] += 1
            # These are state-matched changes against v31 in this arm's own
            # trajectory. They are deliberately not labelled A/B divergence.
            if kind != "v31":
                t0 = time.perf_counter()
                reference = base_action(game, seat, policy="v31")
                reference_seconds += time.perf_counter() - t0
                n_changes += action != reference
        apply_action(game, action, seat=seat)
        steps += 1
    scores = [float(p.score_delta) for p in game.players]
    if not math.isclose(sum(scores), 0.0, abs_tol=1e-9):
        raise RuntimeError(f"Settlement does not sum to zero: {scores}")
    if game.winner is None and any(scores):
        raise RuntimeError(f"Draw must cancel gang settlement: {scores}")
    stats = getattr(bots[hero], "stats", {})
    numeric_stats = {str(k): float(v) for k, v in stats.items()
                     if isinstance(v, (int, float)) and math.isfinite(v)} if isinstance(stats, dict) else {}
    return {"score": scores[hero], "scores": scores, "win": game.winner == hero,
            "draw": game.winner is None, "winner": game.winner, "steps": steps,
            "hero_decisions": n_decisions, "decision_changes_vs_v31": n_changes,
            "hero_actions": dict(hero_actions), "all_actions": dict(all_actions),
            "decision_seconds": decision_seconds, "reference_seconds": reference_seconds,
            "elapsed_seconds": time.perf_counter() - started,
            "search_seed": rng_seed, "search_stats": numeric_stats}


def play_seed(task) -> dict:
    game_seed, config = task
    seats = []
    for hero in range(4):
        # Alternating arm order avoids systematically giving one arm cold caches.
        arms = ("a", "b") if (game_seed + hero) % 2 == 0 else ("b", "a")
        result = {"hero": hero}
        for arm in arms:
            result[arm] = play_game(game_seed, hero, getattr(config, arm), config)
        seats.append(result)
    return {"seed": game_seed, "seats": seats}


# Two-sided .95 Student-t critical values, df=1..30. Above 30 the standard
# inverse-t expansion has sub-.0001 error here and tends to the normal limit.
_T95 = (0, 12.706205, 4.302653, 3.182446, 2.776445, 2.570582, 2.446912,
        2.364624, 2.306004, 2.262157, 2.228139, 2.200985, 2.178813, 2.160369,
        2.144787, 2.131450, 2.119905, 2.109816, 2.100922, 2.093024, 2.085963,
        2.079614, 2.073873, 2.068658, 2.063899, 2.059539, 2.055529, 2.051831,
        2.048407, 2.045230, 2.042272)


def cluster_ci(values) -> dict:
    values = list(values)
    n = len(values)
    if not n:
        raise ValueError("At least one independent seed is required")
    mean = statistics.fmean(values)
    if n < 2:
        return {"mean": mean, "ci95": None, "se": None, "seed_clusters": n}
    df = n - 1
    z = 1.959963984540054
    critical = (_T95[df] if df <= 30 else
                z + (z ** 3 + z) / (4 * df)
                + (5 * z ** 5 + 16 * z ** 3 + 3 * z) / (96 * df ** 2)
                + (3 * z ** 7 + 19 * z ** 5 + 17 * z ** 3 - 15 * z) / (384 * df ** 3))
    se = statistics.stdev(values) / math.sqrt(n)
    half_width = critical * se
    return {"mean": mean, "ci95": [mean - half_width, mean + half_width],
            "se": se, "seed_clusters": n}


def summarize(rows: list[dict], config: EvalConfig, wall_seconds: float) -> dict:
    report = {"config": asdict(config), "created_utc": datetime.now(timezone.utc).isoformat(),
              "comparison": "b_minus_a", "confidence_method": "95% Student-t on per-seed four-seat means",
              "primary_metric": "actual_score_delta", "paired_games": len(rows) * 4,
              "executed_games": len(rows) * 8, "wall_seconds": wall_seconds,
              "games_per_second": len(rows) * 8 / wall_seconds if wall_seconds > 0 else None,
              "arms": {}, "b_minus_a": {}, "results": rows}
    cluster_values = {arm: {} for arm in ("a", "b")}
    for arm in ("a", "b"):
        games = [s[arm] for row in rows for s in row["seats"]]
        entry = {"kind": getattr(config, arm)}
        for metric in ("score", "win", "draw"):
            values = [statistics.fmean(float(s[arm][metric]) for s in row["seats"]) for row in rows]
            cluster_values[arm][metric] = values
            entry[metric] = cluster_ci(values)
        for field in ("steps", "hero_decisions", "decision_changes_vs_v31", "decision_seconds",
                      "reference_seconds", "elapsed_seconds"):
            entry[field] = sum(g[field] for g in games)
        entry["decision_change_rate_vs_v31"] = entry["decision_changes_vs_v31"] / max(1, entry["hero_decisions"])
        entry["seconds_per_hero_decision"] = entry["decision_seconds"] / max(1, entry["hero_decisions"])
        for field in ("hero_actions", "all_actions", "search_stats"):
            counts = Counter()
            for game in games:
                counts.update(game[field])
            entry[field] = dict(counts)
        report["arms"][arm] = entry
    for metric in ("score", "win", "draw"):
        report["b_minus_a"][metric] = cluster_ci(
            b - a for a, b in zip(cluster_values["a"][metric], cluster_values["b"][metric]))
    return report


def evaluate(config: EvalConfig, progress=False) -> dict:
    if config.seeds < 1 or config.workers < 1 or config.simulations < 1 or config.depth < 1 or config.candidate_limit < 1:
        raise ValueError("seeds, workers, simulations, depth and candidate_limit must be positive")
    if config.a not in KINDS or config.b not in KINDS or config.opp not in ("v31", "hv") or config.opponent_policy not in ("v31", "hv"):
        raise ValueError("Unknown policy")
    if config.confirmation is not None and config.confirmation < 2:
        raise ValueError("confirmation must be at least two")
    if not math.isfinite(config.confidence_z) or config.confidence_z < 0 or config.finite_horizon < 1:
        raise ValueError("confidence_z must be finite and nonnegative; finite_horizon must be positive")
    sources_before = source_fingerprint()
    start = time.perf_counter()
    tasks = [(config.seed0 + i, config) for i in range(config.seeds)]
    if config.workers == 1:
        iterator = map(play_seed, tasks)
        pool = None
    else:
        pool = ProcessPoolExecutor(max_workers=config.workers, mp_context=mp.get_context("spawn"))
        iterator = pool.map(play_seed, tasks, chunksize=1)
    rows = []
    try:
        for row in iterator:
            rows.append(row)
            if progress and (len(rows) % max(1, config.seeds // 10) == 0 or len(rows) == config.seeds):
                print(f"Completed {len(rows)}/{config.seeds} seed clusters ({8 * len(rows)} games)", file=sys.stderr, flush=True)
    finally:
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=True)
    report = summarize(rows, config, time.perf_counter() - start)
    report["source_version"] = sources_before
    sources_after = source_fingerprint()
    # Keep the second observation even when a failed Git lookup makes revision
    # verification incomplete. Existing result files are never rewritten here.
    report["source_version_after_run"] = sources_after
    report.update(compare_source_versions(sources_before, sources_after))
    return report


def _format_ci(result):
    ci = result["ci95"]
    return f"{result['mean']:+.4f}" + (f" [{ci[0]:+.4f}, {ci[1]:+.4f}]" if ci else " [CI unavailable: one seed]")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a", choices=KINDS, default="v31")
    parser.add_argument("--b", choices=KINDS, default="v31")
    parser.add_argument("--opp", choices=("v31", "hv"), default="v31")
    parser.add_argument("--seeds", type=int, default=32)
    parser.add_argument("--seed0", type=int, default=5100000)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--simulations", type=int, default=64)
    parser.add_argument("--depth", type=int, default=2, help="Hero decision opportunities, including claims; root mode uses depth=1")
    parser.add_argument("--candidate-limit", type=int, default=4)
    parser.add_argument("--confirmation", type=int, help="Independent fixed-continuation worlds; default max(16, simulations//2)")
    parser.add_argument("--confidence-z", type=float, default=0.0)
    parser.add_argument("--max-search-shanten", type=int)
    parser.add_argument("--finite-horizon", type=int, default=2)
    parser.add_argument("--finite-discount", type=float, default=1.0)
    parser.add_argument("--tree-key", choices=("history", "public_hand"), default="history")
    parser.add_argument("--paired-future", action="store_true")
    parser.add_argument("--future-candidate-limit", type=int, default=2)
    parser.add_argument("--opponent-policy", choices=("v31", "hv"), default="v31",
                        help="Search's simulated opponent; --opp sets actual opponents")
    args = parser.parse_args(argv)
    config = EvalConfig(**{key: value for key, value in vars(args).items() if key != "out"})
    report = evaluate(config, progress=True)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"{config.seeds} independent deal seeds, {report['paired_games']} seat pairs, {report['executed_games']} full games")
    for arm in ("a", "b"):
        entry = report["arms"][arm]
        print(f"{arm.upper()}={entry['kind']}: real score {_format_ci(entry['score'])}; win {entry['win']['mean']:.2%}; draw {entry['draw']['mean']:.2%}")
        print(f"  changes vs v31 in own trajectory: {entry['decision_changes_vs_v31']}/{entry['hero_decisions']}; hero decision {entry['seconds_per_hero_decision']:.5f}s")
    print(f"B-A real score {_format_ci(report['b_minus_a']['score'])}; win {_format_ci(report['b_minus_a']['win'])}")
    print(f"Wall {report['wall_seconds']:.2f}s; {report['games_per_second']:.2f} games/s; intervals cluster all four seats by seed")
    if report["source_changed_during_run"]:
        print("A source, policy-environment or verified Git revision change was observed during this run; rerun with fixed inputs before comparing policies.")
    if not report["git_revision_verification_complete"]:
        print("Git revision verification is incomplete; inspect both saved source-version snapshots and Git query diagnostics.")
    return report


if __name__ == "__main__":
    main()
