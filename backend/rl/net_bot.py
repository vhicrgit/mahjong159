"""安康159 - 神经网络Bot

用训练好的模型做出牌决策(碰/杠沿用规则)。
可选温度采样: temperature=0 时取 argmax(最强), >0 时按概率采样(多样性)。
"""

import torch

from ..rl.features import encode_state
from ..rl.model import build_model, legal_discard_mask
from ..ai.bot import Bot


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
            probs = self.model.policy(x, mask)[0]
        if self.temperature <= 0:
            return int(probs.argmax().item())
        probs = probs.clamp(min=1e-9)
        dist = torch.distributions.Categorical(
            probs ** (1.0 / self.temperature))
        return int(dist.sample().item())
