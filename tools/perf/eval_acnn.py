"""acnn(actor-critic NN bot) 的验收评估: 同种子配对。

三组对局(每种子各打两局配对):
  A: acnn@座0 vs 3×v1n     / 参照: v31n@座0 vs 3×v1n
  B: acnn@座0 vs 3×v31n    / 参照: scholar@座0 vs 3×v31n (学者慢, 可关)
报告: 胜率/场均 + McNemar 精确检验(acnn vs 参照)。

用法: python -m tools.perf.eval_acnn --games 200 --procs 10 \
        --model models/acnn_latest_best.pt
"""

import argparse
import math
import multiprocessing as mp
import os

import numpy as np

from backend.ai.bot_native import NativeV1, NativeV31
from backend.game.engine import Game

_MODEL = None


def _get_nn():
    global _MODEL
    if _MODEL is None:
        import torch
        from backend.rl.model import build_model, legal_discard_mask
        from backend.rl.features_v2 import encode_state
        ckpt = torch.load(os.environ["ACNN_MODEL"], map_location="cpu",
                          weights_only=True)
        net = build_model(ckpt["size"])
        net.load_state_dict(ckpt["model"])
        net.eval()

        class G:
            def __init__(self, game, seat):
                self.game, self.seat = game, seat
                from backend.ai.bot_v1 import Bot as V1
                self._v1 = V1(game, seat)

            def choose_discard(self):
                feat = encode_state(self.game, self.seat)
                x = torch.from_numpy(feat).unsqueeze(0)
                mask = legal_discard_mask(
                    self.game.players[self.seat].hand_counts).unsqueeze(0)
                with torch.no_grad():
                    q = net.q(x, mask)[0]
                return int(q.argmax().item())

            def decide_peng(self, tile):
                from backend.native import native
                return native.decide_peng(31, self.game.players[self.seat].hand_counts, tile)

            def decide_gang(self, tile, kind):
                from backend.native import native
                return native.decide_gang(31, self.game.players[self.seat].hand_counts, tile, kind)

        _MODEL = G
    return _MODEL


def play(seed, hero_kind):
    """hero_kind: 'nn' | 'v31' | 'v1'。其余三座 v31n。"""
    g = Game(seed=seed, human_seat=-1)
    if hero_kind == "nn":
        bots = {0: _get_nn()(g, 0)}
    elif hero_kind == "v31":
        bots = {0: NativeV31(g, 0)}
    else:
        bots = {0: NativeV1(g, 0)}
    for i in range(1, 4):
        bots[i] = NativeV31(g, i)
    guard = 0
    while g.phase != "game_over" and guard < 500:
        guard += 1
        if g.phase == "discard_wait":
            g.action_discard(g.turn, bots[g.turn].choose_discard())
        else:
            s = list(g.pending_actions.keys())[0]
            pend = g.pending_actions[s]
            b = bots[s]
            if pend.get("gang") and b.decide_gang(g.last_discard, "ming"):
                g.action_gang(s)
            elif pend.get("peng") and b.decide_peng(g.last_discard):
                g.action_peng(s)
            else:
                g.action_pass(s)
    return (1 if g.winner == 0 else 0, g.players[0].score_delta)


def work(seed):
    return play(seed, "nn"), play(seed, "v31")


def mcnemar(wins_a, wins_b):
    b = sum(1 for x, y in zip(wins_a, wins_b) if x == 1 and y == 0)
    c = sum(1 for x, y in zip(wins_a, wins_b) if x == 0 and y == 1)
    n = b + c
    if n == 0:
        return b, c, 1.0
    k = min(b, c)
    p = 2 * min(1.0, sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n)
    return b, c, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--seed0", type=int, default=3000000)
    ap.add_argument("--procs", type=int, default=10)
    ap.add_argument("--model", type=str, default="models/acnn_latest_best.pt")
    args = ap.parse_args()

    if not os.path.exists(args.model):
        args.model = args.model.replace("_best", "")
    os.environ["ACNN_MODEL"] = args.model
    print(f"模型: {args.model}")

    with mp.get_context("fork").Pool(args.procs) as pool:
        rets = pool.map(work, [args.seed0 + i for i in range(args.games)],
                        chunksize=4)
    A = [r[0] for r in rets]
    B = [r[1] for r in rets]

    def stat(rows):
        w = np.array([r[0] for r in rows], dtype=float)
        s = np.array([r[1] for r in rows], dtype=float)
        return (w.mean() * 100,
                1.96 * math.sqrt(w.mean() * (1 - w.mean()) / len(w)) * 100,
                s.mean())

    for name, rows in (("acnn vs 3×v31n", A), ("v31n(参照)  ", B)):
        wr, ci, sc = stat(rows)
        print(f"{name:16s} 胜率 {wr:.1f}% (±{ci:.1f})  场均 {sc:+.2f}")
    b_, c_, p_ = mcnemar([r[0] for r in A], [r[0] for r in B])
    print(f"配对: b={b_} c={c_}, McNemar p={p_:.4f}")


if __name__ == "__main__":
    main()
