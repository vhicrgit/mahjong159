"""安康159 - 神经网络Bot (Q值版)

用训练好的 Q 网络做出牌决策(碰/杠沿用规则)。
推理: argmax Q(s,a) over legal actions
训练探索: Boltzmann(Q/T) 采样
"""

import torch

from ..rl.features_v2 import encode_state
from ..rl.model import build_model, legal_discard_mask
from ..ai.bot_v1 import Bot


class NetBot(Bot):
    def __init__(self, game, seat: int, model_path: str, temperature: float = 0.0):
        super().__init__(game, seat)
        ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
        self.model = build_model(ckpt["size"])
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()
        self.temperature = temperature

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
