"""安康159 - 混合决策 Bot

不修改模型, 在推理时结合:
1. 神经网络 Q 值 (从BC学到的模式识别)
2. 规则Bot 分析分数 (精确的向听/进张/风险计算)

加权组合: final_score = Q(s,a) + alpha * rule_score(a)
alpha 控制规则Bot的影响力

这样可以在不训练的情况下提升模型表现。
"""

import torch

from ..rl.features_v2 import encode_state
from ..rl.model import build_model, legal_discard_mask
from ..rl.vec_selfplay import _rule_score
from ..ai.bot_v1 import Bot


class HybridBot(Bot):
    """混合决策: Q值 + 规则Bot评分"""

    def __init__(self, game, seat: int, model_path: str, alpha: float = 0.5):
        super().__init__(game, seat)
        ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
        self.model = build_model(ckpt["size"])
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()
        self.alpha = alpha

    def choose_discard(self) -> int:
        feat = encode_state(self.game, self.seat)
        x = torch.from_numpy(feat).unsqueeze(0)
        mask = legal_discard_mask(
            self.game.players[self.seat].hand_counts).unsqueeze(0)
        with torch.no_grad():
            q = self.model.q(x, mask)[0]

        # 对每个可选牌, 计算规则Bot评分
        hand = self.game.players[self.seat].hand_counts
        best_tile, best_combined = None, -1e9
        for t in range(28):
            if hand[t] <= 0:
                continue
            rule_s = _rule_score(self.game, self.seat, t)
            combined = float(q[t]) + self.alpha * rule_s
            if combined > best_combined:
                best_combined = combined
                best_tile = t
        return best_tile
