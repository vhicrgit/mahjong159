"""安康159 - 神经网络Bot (Q值版)

出牌用训练好的 Q 网络: argmax Q(s,a) over legal actions;
碰/杠用牌型分析器的 E 判据(网络本身不建模碰杠), 再加 v32 的选择性碰牌分支。

为什么碰杠用 E 而不是规则: CRN 配对实测(200 seed × 4 座位), 同一个 92%
模仿分析器的网络 ——
  配 v31 规则碰杠   首胡 -0.133 / 血战 -0.324 分/局 vs v31n
  配 E 判据碰杠     首胡 -0.083 / 血战 -0.210
也就是白拿 0.05~0.11 分/局。原先这里继承的是 bot_v1 的碰杠, 比 v31 还弱。

为什么**不**接 v32 的选择性碰牌: 规则侧 v31 的 decide_peng 只认"碰后向听严格
下降", 已听牌时活听口再宽也拒碰, 补上这条分支实测 +0.0215 分/局(bot_v32)。
但 E 判据没有这个缺口 —— 1500 局 1047 次碰机会里 ready_peng_plan 命中 17 次,
这 17 次 E 全部已经接受碰, 净增量 0; 1500 种子配对 A/B 两臂 6000 局结果逐局
相同(+0.0000 ± 0.0000)。所以 NN 侧不接, 免得多一条永不触发的分支。
"""

import torch

from ..analysis import hv_native
from ..rl.features_v2 import encode_state
from ..rl.model import build_model, legal_discard_mask
from ..ai.bot_v1 import Bot

RED = 27


class NetBot(Bot):
    def __init__(self, game, seat: int, model_path: str,
                 temperature: float = 0.0, claim_kai: int = 0):
        super().__init__(game, seat)
        # 自家训练产物, 内容只有张量与基本类型 -> weights_only 加载
        ckpt = torch.load(model_path, map_location="cpu", weights_only=True)
        self.model = build_model(ckpt["size"], feat_dim=ckpt.get("feat_dim", 628))
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()
        self.temperature = temperature
        self.claim_kai = claim_kai

    def choose_discard(self) -> int:
        feat = encode_state(self.game, self.seat)
        x = torch.from_numpy(feat).unsqueeze(0)
        mask = legal_discard_mask(
            self.game.players[self.seat].hand_counts).unsqueeze(0)
        with torch.no_grad():
            q = self.model.q(x, mask)[0]
        if self.temperature <= 0:
            return int(q.argmax().item())
        # Boltzmann exploration
        logits = q / self.temperature
        logits = logits - logits.max()  # numerical stability
        probs = torch.softmax(logits, dim=-1)
        return int(torch.multinomial(probs, 1).item())

    # ---- 碰/杠: 牌型分析器的 E 判据 ----
    def _hv(self):
        vis = [0] * 28
        for q in self.game.players:
            for t in q.discards:
                vis[t] += 1
            for m in q.melds:
                vis[m["tile"]] += 3 if m["type"] == "peng" else 4
        for t, n in enumerate(self.game.players[self.seat].hand_counts):
            vis[t] += n
        ok = hv_native.set_hand(
            list(self.game.players[self.seat].hand_counts), vis, 1.0,
            self.claim_kai > 0, 2, self.claim_kai, 6)
        return hv_native if ok else None

    def decide_peng(self, tile: int) -> bool:
        hv = self._hv()
        if hv is None:
            return super().decide_peng(tile)     # C 库不可用时退回规则
        return hv.decide_peng(tile)

    def decide_gang(self, tile: int, kind: str) -> bool:
        hv = self._hv()
        if hv is None:
            return super().decide_gang(tile, kind)
        return hv.decide_gang(tile, kind)
