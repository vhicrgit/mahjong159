"""CRN(共同随机数)配对能降多少方差 —— 决定"自对弈进化"路线是否成立。

做法: 同一批 seed, 座0 分别放策略 A / B, 其余三家固定 v31n(确定性)。
比较未配对与配对两种差分估计量的方差:
  未配对 var = var(G_A) + var(G_B)
  配对   var = var(G_A - G_B)
比值就是等效样本量的放大倍数。

同时报"轨迹分叉"诊断: 两条臂的总弃牌数/胜者是否一致 —— 分叉越早,
共享的摸牌顺序越少, CRN 越失效。

用法:
  python -m tools.perf.diag_crn --games 600 --pair v31n:v10
  python -m tools.perf.diag_crn --games 600 --pair v31n:v31n_eps
"""

import argparse

import numpy as np
import torch

from backend.ai.bot_native import NativeV1, NativeV10, NativeV31
from backend.game.engine import Game
from backend.native import native
from backend.rl.features_v2 import encode_state
from backend.rl.model import build_model, legal_discard_mask

_MODEL = {}


def _nn(path):
    if path not in _MODEL:
        ck = torch.load(path, map_location="cpu", weights_only=True)
        m = build_model(ck["size"])
        m.load_state_dict(ck["model"])
        m.eval()
        _MODEL[path] = m
    return _MODEL[path]


class NNSeat:
    """NN 弃牌 + v31 规则的碰/杠(与其它臂一致, 只让弃牌不同)。"""

    def __init__(self, game, seat, path, temp=0.0, rng=None):
        self.game, self.seat = game, seat
        self.model = _nn(path)
        self.temp, self.rng = temp, rng

    def choose_discard(self):
        x = torch.from_numpy(encode_state(self.game, self.seat)).unsqueeze(0)
        mask = legal_discard_mask(
            self.game.players[self.seat].hand_counts).unsqueeze(0)
        with torch.no_grad():
            q = self.model.q(x, mask)[0]
        if self.temp <= 0:
            return int(q.argmax().item())
        p = torch.softmax(q / self.temp, -1).numpy()
        return int(self.rng.choice(len(p), p=p / p.sum()))

    def decide_peng(self, tile):
        return native.decide_peng(
            31, self.game.players[self.seat].hand_counts, tile)

    def decide_gang(self, tile, kind):
        return native.decide_gang(
            31, self.game.players[self.seat].hand_counts, tile, kind)


class EpsSeat(NativeV31):
    """v31n 但以 eps 概率改打次优 —— 模拟"相邻检查点"这种微小策略差。"""

    def __init__(self, game, seat, eps, rng):
        super().__init__(game, seat)
        self.eps, self.rng = eps, rng

    def choose_discard(self):
        best = super().choose_discard()
        if self.rng.random() >= self.eps:
            return best
        alt = [t for t, n in enumerate(self.game.players[self.seat].hand_counts)
               if n and t != best]
        return int(self.rng.choice(alt)) if alt else best


def make_seat(spec, game, seat, rng):
    if spec == "v31n":
        return NativeV31(game, seat)
    if spec == "v10n":
        return NativeV10(game, seat)
    if spec == "v1n":
        return NativeV1(game, seat)
    if spec.startswith("eps"):
        return EpsSeat(game, seat, float(spec[3:]) / 100.0, rng)
    if spec.startswith("nn"):
        # nn 或 nn@路径
        path = spec.split("@", 1)[1] if "@" in spec else "models/acnn_v2.pt"
        return NNSeat(game, seat, path)
    raise SystemExit(f"未知策略 {spec}")


def adjusted(game, seat):
    s = game.players[seat].score_delta
    for rec in game.gang_records:
        if rec["kind"] == "ming" and rec["from"] == seat:
            s += 3
    return float(s)


def play(seed, spec, rng):
    g = Game(seed=seed, human_seat=-1)
    bots = {s: NativeV31(g, s) for s in range(1, 4)}
    bots[0] = make_seat(spec, g, 0, rng)
    nd, guard = 0, 0
    while g.phase != "game_over" and guard < 500:
        guard += 1
        if g.phase == "discard_wait":
            g.action_discard(g.turn, bots[g.turn].choose_discard())
            nd += 1
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
    return adjusted(g, 0), (1.0 if g.winner == 0 else 0.0), nd, g.winner


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=600)
    ap.add_argument("--pair", default="v31n:v10n",
                    help="A:B, 可选 v31n/v10n/v1n/epsN/nn[@路径]")
    ap.add_argument("--seed0", type=int, default=9000000)
    args = ap.parse_args()
    A, B = args.pair.split(":")
    print(f"座0: A={A}  B={B}   其余三家 v31n   {args.games} 局")

    ga, gb, wa, wb, same = [], [], [], [], []
    for i in range(args.games):
        seed = args.seed0 + i
        # 每条臂用独立同种子的 rng, 保证随机策略在两臂间也共享随机数
        a = play(seed, A, np.random.default_rng(seed))
        b = play(seed, B, np.random.default_rng(seed))
        ga.append(a[0]); wa.append(a[1])
        gb.append(b[0]); wb.append(b[1])
        same.append(1.0 if (a[2] == b[2] and a[3] == b[3]) else 0.0)
        if (i + 1) % 200 == 0:
            print(f"  ...{i + 1}/{args.games}")

    ga, gb = np.array(ga), np.array(gb)
    d = ga - gb
    n = len(ga)
    v_unpaired = ga.var() + gb.var()
    v_paired = d.var()
    print(f"\nA: 得分 {ga.mean():+.3f}±{ga.std() / n ** .5:.3f} "
          f"标准差 {ga.std():.2f}  胜率 {np.mean(wa):.1%}")
    print(f"B: 得分 {gb.mean():+.3f}±{gb.std() / n ** .5:.3f} "
          f"标准差 {gb.std():.2f}  胜率 {np.mean(wb):.1%}")
    print(f"corr(G_A, G_B) = {np.corrcoef(ga, gb)[0, 1]:+.3f}")
    print(f"\n未配对 var(A)+var(B) = {v_unpaired:.2f}")
    print(f"配对   var(A-B)      = {v_paired:.2f}")
    print(f"**降方差倍数 = {v_unpaired / v_paired:.2f}x**"
          f"   (等效样本量放大同样倍数)")
    print(f"\n差分: 均 {d.mean():+.3f} ± {d.std() / n ** .5:.3f}  "
          f"t = {d.mean() / (d.std() / n ** .5):+.2f}")
    print(f"轨迹完全一致的局: {np.mean(same):.1%}  "
          f"(弃牌总数与胜者都相同)")


if __name__ == "__main__":
    main()
