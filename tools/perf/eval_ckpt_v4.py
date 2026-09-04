"""v4(对手模型特征)模型的 CRN 配对评估。

与 tools/perf/eval_ckpt.py 的区别: 被测 NN 的输入是 718 维 v4 特征, 需要
在整局事件流上实时维护 3 个 OppTracker(policy=False, 与训练采集同配置)。
对手与基线仍是 v31n。输出与 eval_crn 同口径, 可直接与 r2(628维) 的
历史数字对比。

用法:
  python -m tools/perf/eval_ckpt_v4.py models/bc_v4_r1.pt --seeds 400
"""

import argparse

import numpy as np
import torch

from backend.ai.bot_native import NativeV31
from backend.analysis.opp_model import OppTracker
from backend.game.engine import Game
from backend.rl import eval_crn
from backend.rl.features_v2 import encode_state as encode_v2
from backend.rl.features_v4 import FEAT_DIM, opp_features
from backend.rl.model import build_model, legal_discard_mask


class V4Seat:
    """NN(v4特征) 弃牌 + 分析器 E 碰杠(与 r2 部署口径一致)。
    trackers: {opp_seat: OppTracker}, 由驱动器喂事件。"""

    def __init__(self, game, seat, model, trackers):
        self.game, self.seat, self.model = game, seat, model
        self.trackers = trackers
        self._hv = None

    def choose_discard(self):
        base = encode_v2(self.game, self.seat)
        feat = np.concatenate([base, opp_features(self.trackers, self.seat)])
        x = torch.from_numpy(feat).unsqueeze(0)
        m = legal_discard_mask(
            self.game.players[self.seat].hand_counts).unsqueeze(0)
        with torch.no_grad():
            q = self.model.q(x, m)[0]
        return int(q.argmax().item())

    def _setup_hv(self):
        from backend.analysis import hv_native
        vis = [0] * 28
        for q in self.game.players:
            for t in q.discards:
                vis[t] += 1
            for m in q.melds:
                vis[m["tile"]] += 3 if m["type"] == "peng" else 4
        for t, n in enumerate(self.game.players[self.seat].hand_counts):
            vis[t] += n
        hv_native.set_hand(list(self.game.players[self.seat].hand_counts),
                           vis, 1.0, False, 2, 0, 6)
        return hv_native

    def decide_peng(self, tile):
        return self._setup_hv().decide_peng(tile)

    def decide_gang(self, tile, kind):
        return self._setup_hv().decide_gang(tile, kind)


def play_v4(seed, bloody, model, hero, n_init=1500, beam=300):
    """hero 座位用 V4Seat(实时 tracker), 其余 v31n。返回终局 Game。"""
    g = Game(seed=seed, human_seat=-1, bloody=bloody)
    trs = {}
    for rel in (1, 2, 3):
        opp = (hero + rel) % 4
        trs[opp] = OppTracker(opp, list(g.players[hero].hand_counts),
                              n_init=n_init, beam=beam, policy=False,
                              seed=seed * 10 + opp, hero_seat=hero)
    for tr in trs.values():
        tr.notify_deal(g.dealer)
    bots = {s: NativeV31(g, s) for s in range(4)}
    bots[hero] = V4Seat(g, hero, model, trs)

    def feed_draw(seat, tile):
        for tr in trs.values():
            tr.notify_draw(seat, tile if seat == hero else None,
                           g.wall_remaining())

    guard = 0
    while g.phase != "game_over" and guard < 900:
        guard += 1
        if g.phase == "discard_wait":
            s = g.turn
            d = bots[s].choose_discard()
            for tr in trs.values():
                tr.notify_discard(s, d, g.wall_remaining())
            ev = g.action_discard(s, d)
        else:
            s = list(g.pending_actions.keys())[0]
            pend = g.pending_actions[s]
            tile, disc = g.last_discard, g.last_discarder
            b = bots[s]
            if pend.get("gang") and b.decide_gang(tile, "ming"):
                for tr in trs.values():
                    tr.notify_claim(s, "gang", tile, disc)
                ev = g.action_gang(s)
            elif pend.get("peng") and b.decide_peng(tile):
                for tr in trs.values():
                    tr.notify_claim(s, "peng", tile, disc)
                ev = g.action_peng(s)
            else:
                for tr in trs.values():
                    tr.notify_claim(s, None, tile, disc)
                ev = g.action_pass(s)
        if ev.get("event") in ("draw", "gang_draw"):
            feed_draw(ev["seat"], ev.get("tile"))
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--seeds", type=int, default=400)
    ap.add_argument("--seed0", type=int, default=150000000)
    args = ap.parse_args()
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    fd = ck.get("feat_dim", FEAT_DIM)
    model = build_model(ck["size"], feat_dim=fd)
    model.load_state_dict(ck["model"])
    model.eval()
    print(f"候选: {args.ckpt} (size={ck['size']}, feat_dim={fd})")
    print(f"基线: v31n   {args.seeds} seed × 4 座位 (CRN 配对)")

    for bloody, name in ((False, "首胡(线上规则)"), (True, "血战到底")):
        rr_a = np.zeros((args.seeds, 4))
        sc_a = np.zeros((args.seeds, 4))
        for i in range(args.seeds):
            seed = args.seed0 + i
            for s in range(4):
                g = play_v4(seed, bloody, model, s)
                rr_a[i, s] = g.rank_rewards()[s]
                sc_a[i, s] = eval_crn._adjusted(g, s)
        rr_b, sc_b = eval_crn.baseline_arm(NativeV31,
                                           list(range(args.seed0,
                                                      args.seed0 + args.seeds)),
                                           bloody)
        r = eval_crn._stat(rr_a - rr_b)
        sc = eval_crn._stat(sc_a - sc_b)
        print(f"\n[{name}]")
        print(f"  名次奖励差 {r['mean']:+.4f} ± {r['se']:.4f}  t={r['t']:+.2f}"
              f"  (n={r['n']})")
        print(f"  调整得分差 {sc['mean']:+.4f} ± {sc['se']:.4f}  t={sc['t']:+.2f}")


if __name__ == "__main__":
    main()
