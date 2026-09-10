"""Inspect one reproducible, public-information search decision.

python -m tools.analyze_search159 --seed 123 --seat 0 --decision 3 \
    --mode search --simulations 64 --out work/search159/decision.json

The game reaches the requested hero decision through complete v31 play. The
saved observation excludes opponents' hands and the actual wall order.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.ai.bot_search159 import Bot
from backend.ai.search159 import simulator as sim
from backend.game.engine import Game
from backend.rules.tiles import tile_name


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--seat", type=int, choices=range(4), default=0)
    parser.add_argument("--decision", type=int, default=1)
    parser.add_argument("--mode", choices=("search", "root", "finite"), default="search")
    parser.add_argument("--simulations", type=int, default=32)
    parser.add_argument("--confirmation", type=int)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--candidate-limit", type=int, default=4)
    parser.add_argument("--confidence-z", type=float, default=0.0)
    parser.add_argument("--finite-horizon", type=int, default=3)
    parser.add_argument("--finite-discount", type=float, default=1.0)
    parser.add_argument("--tree-key", choices=("history", "public_hand"), default="history")
    parser.add_argument("--paired-future", action="store_true")
    parser.add_argument("--future-candidate-limit", type=int, default=2)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.decision < 1:
        parser.error("--decision must be positive")
    game = Game(seed=args.seed, human_seat=-1)
    seen = 0
    while game.phase != "game_over":
        seat = sim.acting_seat(game)
        if seat == args.seat:
            seen += 1
            if seen == args.decision:
                break
        sim.apply_action(game, sim.base_action(game, seat), seat)
    if game.phase == "game_over":
        parser.error(f"game ended after only {seen} hero decisions")
    bot = Bot(game, args.seat, mode=args.mode, simulations=args.simulations,
              confirmation=args.confirmation, depth=args.depth,
              candidate_limit=args.candidate_limit, confidence_z=args.confidence_z,
              finite_horizon=args.finite_horizon,
              finite_discount=args.finite_discount, seed=args.seed,
              tree_key=args.tree_key, paired_future=args.paired_future,
              future_candidate_limit=args.future_candidate_limit)
    action = bot.choose_action()
    result = {"seed": args.seed, "seat": args.seat, "decision": args.decision,
              "mode": args.mode, "observation": asdict(sim.observe(game, args.seat)),
              "hand_names": [tile_name(t) for t in game.players[args.seat].hand],
              "action": asdict(action), "analysis": bot.explain(), "stats": bot.stats}
    output = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(output)
    print(output)


if __name__ == "__main__":
    main()
