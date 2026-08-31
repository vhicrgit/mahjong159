"""采样损失随 q 缩放的变化: 缩放不改 argmax(部署不变), 但让 softmax(q) 变尖。
若某个缩放下"采样策略 ≈ 贪心策略", PPO 优化的目标就与部署一致了。"""
import sys, numpy as np, torch, torch.nn.functional as F
from backend.rl import eval_crn
from backend.rl.features_v2 import encode_state
from backend.rl.model import build_model, legal_discard_mask
from tools.rl_bloody_train import NNSeat

ck = torch.load("models/bc_k0_r2.pt", map_location="cpu", weights_only=True)
m = build_model(ck["size"]); m.load_state_dict(ck["model"]); m.eval()

class Samp(NNSeat):
    def __init__(self, g, s, model, scale):
        super().__init__(g, s, model); self.scale = scale
    def choose_discard(self):
        x = torch.from_numpy(encode_state(self.game, self.seat)).unsqueeze(0)
        msk = legal_discard_mask(self.game.players[self.seat].hand_counts).unsqueeze(0)
        with torch.no_grad():
            q = self.model.q(x, msk) * self.scale
            p = F.softmax(q.masked_fill(~msk, -1e9), -1)[0]
        return int(torch.multinomial(p, 1).item())

seeds = list(range(160000000, 160000000 + int(sys.argv[1] if len(sys.argv)>1 else 150)))
# 熵参考
ent = []
from backend.game.engine import Game
g = Game(seed=1, human_seat=-1)
for scale in (1.0, 2.0, 3.0, 5.0, 8.0):
    ev = eval_crn.paired_head2head(lambda gg,s,sc=scale: Samp(gg,s,m,sc),
                                   lambda gg,s: NNSeat(gg,s,m), seeds, bloody=True)
    r = ev["rank"]
    print(f"q×{scale:4.1f}  采样 - 贪心 = {r['mean']:+.4f} ± {r['se']:.4f}  t={r['t']:+.2f}", flush=True)
