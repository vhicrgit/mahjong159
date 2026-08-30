"""用 CRN 配对协议评估一个检查点 —— 两种规则都测。

训练在血战到底上做, 但线上跑的是首胡规则, 所以必须两边都看:
血战上变强不代表首胡上变强。基线是 v31n(轮坐四座位, 同 seed 配对)。

用法:
  python -m tools.perf.eval_ckpt models/bloody_latest_best.pt [--seeds 400]
  python -m tools.perf.eval_ckpt models/acnn_v2.pt --seeds 400   # 老模型基准
  python -m tools.perf.eval_ckpt v31n --seeds 400                # 自检(应≈0)
"""

import argparse

import torch

from backend.ai.bot_native import NativeV1, NativeV10, NativeV31
from backend.rl import eval_crn
from backend.rl.model import build_model
from tools.rl_bloody_train import NNSeat


class HVSeat:
    """牌型分析器(E 引擎)本体, 走 C 后端。

    存在的意义: BC 的目标就是它。如果它在得分上并不强于 v31n, 那么克隆它
    的网络上限也就被锁死了 —— 这一条必须单独验证, 不能假设。

    v31_claim=True 时碰/杠改用 v31 规则 —— NN 的包装就是这么做的(网络只
    建模弃牌), 所以这个变体能把"弃牌策略"和"碰杠策略"的贡献分开。
    """

    def __init__(self, game, seat, kai=1, v31_claim=False):
        self.game, self.seat, self.kai = game, seat, kai
        self._v31 = NativeV31(game, seat) if v31_claim else None

    def _vis(self):
        v = [0] * 28
        for q in self.game.players:
            for t in q.discards:
                v[t] += 1
            for m in q.melds:
                v[m["tile"]] += 3 if m["type"] == "peng" else 4
        for t, n in enumerate(self.game.players[self.seat].hand_counts):
            v[t] += n
        return v

    def _setup(self):
        from backend.analysis import hv_native
        hv_native.set_hand(list(self.game.players[self.seat].hand_counts),
                           self._vis(), 1.0, self.kai > 0, 2, self.kai, 6)
        return hv_native

    def choose_discard(self):
        hv = self._setup()
        t = hv.choose_discard()
        if t < 0:
            return self.game.players[self.seat].hand[-1]
        return t

    def decide_peng(self, tile):
        if self._v31 is not None:
            return self._v31.decide_peng(tile)
        return self._setup().decide_peng(tile)

    def decide_gang(self, tile, kind):
        if self._v31 is not None:
            return self._v31.decide_gang(tile, kind)
        return self._setup().decide_gang(tile, kind)


class NNHVClaim(NNSeat):
    """NN 弃牌 + 分析器 E 判据的碰/杠。

    网络只建模弃牌, 碰/杠一直是 v31 规则代劳的。分析器本体(自己的 E 碰杠)
    是 -0.061 分/局, 而 92% 模仿它的网络配 v31 碰杠只有 -0.324 —— 碰杠策略
    是这 0.26 分差距的头号嫌疑。这个变体用来验证。
    """

    def __init__(self, game, seat, model, kai=0):
        super().__init__(game, seat, model)
        self._hv = HVSeat(game, seat, kai)

    def decide_peng(self, tile):
        return self._hv.decide_peng(tile)

    def decide_gang(self, tile, kind):
        return self._hv.decide_gang(tile, kind)


def make_factory(spec, claim="v31"):
    if spec == "v31n":
        return NativeV31, "v31n"
    if spec == "v10n":
        return NativeV10, "v10n"
    if spec == "v1n":
        return NativeV1, "v1n"
    if spec.startswith("hv"):
        rest = spec[2:]
        v31c = rest.endswith("v31")
        if v31c:
            rest = rest[:-3]
        kai = int(rest) if rest else 1
        return ((lambda g, s: HVSeat(g, s, kai, v31c)),
                f"牌型分析器 kai={kai}"
                + (" + v31 碰杠" if v31c else " (自己的 E 碰杠)"))
    ck = torch.load(spec, map_location="cpu", weights_only=True)
    m = build_model(ck["size"])
    m.load_state_dict(ck["model"])
    m.eval()
    tag = f"{spec} (size={ck['size']}, iter={ck.get('iter')}, 碰杠={claim})"
    if claim == "hv":
        return (lambda g, s: NNHVClaim(g, s, m)), tag
    return (lambda g, s: NNSeat(g, s, m)), tag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--seeds", type=int, default=400)
    ap.add_argument("--seed0", type=int, default=41000000)
    ap.add_argument("--claim", choices=["v31", "hv"], default="v31",
                    help="模型的碰/杠由谁决定(网络本身不建模碰杠)")
    args = ap.parse_args()
    fac, tag = make_factory(args.ckpt, args.claim)
    seeds = list(range(args.seed0, args.seed0 + args.seeds))
    print(f"候选: {tag}\n基线: v31n   {args.seeds} seed × 4 座位 (CRN 配对)")
    for bloody, name in ((False, "首胡(线上规则)"), (True, "血战到底")):
        ev = eval_crn.paired_vs_v31(fac, seeds, bloody=bloody)
        r, s = ev["rank"], ev["score"]
        print(f"\n[{name}]")
        print(f"  名次奖励差 {r['mean']:+.4f} ± {r['se']:.4f}  t={r['t']:+.2f}"
              f"   (n={r['n']})")
        print(f"  调整得分差 {s['mean']:+.4f} ± {s['se']:.4f}  t={s['t']:+.2f}")
        print(f"  候选自身: 名次奖励均 {ev['cand_rank_mean']:+.3f}  "
              f"得分均 {ev['cand_score_mean']:+.3f}")


if __name__ == "__main__":
    main()
