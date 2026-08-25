"""安康159 - 向量化并行自对弈引擎

核心思路: 同时运行 N 局游戏, 每一步把所有需要模型决策的状态
batch 成一个 tensor, 一次前向传播完成所有推理。

支持两种模式:
- self-play: model_seats=[0,1,2,3], 所有座位由模型控制
- vs-rule:   model_seats=[0], 只有座位0由模型控制, 其他用规则Bot
  (编码量减少 4x, 且环境是静态的, 训练更稳定)
"""

import numpy as np
import torch

from ..game.engine import Game
from ..ai.bot import Bot
from ..rules.ting import discard_options
from .features_v2 import encode_state
from .model import legal_discard_mask


def _rule_score(game, seat, tile: int) -> float:
    """计算规则Bot对打出 tile 的评分 (用于 reward shaping)"""
    from ..rules.win import shanten
    from ..rules.ting import waiting_tiles
    RED = 27
    p = game.players[seat]
    counts = list(p.hand_counts)
    counts[tile] -= 1
    s = shanten(counts)
    waits = waiting_tiles(counts) if s == 0 else []
    wait_count = sum(4 - counts[w] for w in waits) if s == 0 else 0

    # 放杠风险
    visible = [0] * 28
    for q in game.players:
        for t in q.discards:
            visible[t] += 1
        for m in q.melds:
            visible[m["tile"]] += 3 if m["type"] == "peng" else 4
    visible[tile] += 1  # 加上即将打出的这张

    risk = 0.0
    if tile != RED:
        remain = 4 - visible[tile]
        if remain >= 3: risk = 0.4
        elif remain == 2: risk = 0.2
        elif remain == 1: risk = 0.05
        # 对手碰了这张牌
        for q in game.players:
            if q.seat != seat and any(
                    m["tile"] == tile and m["type"] == "peng" for m in q.melds):
                risk = 1.0
                break

    return -100.0 * s + 3.0 * wait_count - 25.0 * risk


class VectorizedSelfPlay:
    def __init__(self, model, n_games: int, device, seed0: int = 0,
                 model_seats: list[int] | None = None, grp_model=None,
                 grp_version: int = 2):
        self.model = model
        self.device = device
        self.n_games = n_games
        self.model_seats = model_seats or [0, 1, 2, 3]
        self.grp_model = grp_model
        self.grp_version = grp_version
        self.games = [Game(seed=seed0 + i, human_seat=-1)
                      for i in range(n_games)]
        self.bots = [{s: Bot(g, s) for s in range(4)}
                     for g in self.games]
        self.records: list[list] = [[] for _ in range(n_games)]
        # 与 records 平行的 GRP 预测值 (决策玩家自己的预期最终得分)
        self.grp_vals: list[list] = [[] for _ in range(n_games)]

    def run(self, temperature: float = 0.5) -> list[dict]:
        guard = 0
        max_steps = 500 * self.n_games
        while any(g.phase != "game_over" for g in self.games) and guard < max_steps:
            guard += 1
            self._step_discard(temperature)
            self._step_react()
        return self._collect_results()

    def _step_discard(self, temperature: float):
        # 分出哪些游戏需要模型推理, 哪些用规则Bot
        model_idxs = [i for i, g in enumerate(self.games)
                      if g.phase == "discard_wait" and g.turn in self.model_seats]
        rule_idxs = [i for i, g in enumerate(self.games)
                     if g.phase == "discard_wait" and g.turn not in self.model_seats]

        # 模型批量推理
        if model_idxs:
            feats = np.stack([encode_state(self.games[i], self.games[i].turn)
                              for i in model_idxs])
            masks = torch.stack([legal_discard_mask(
                self.games[i].players[self.games[i].turn].hand_counts)
                for i in model_idxs])

            x = torch.from_numpy(feats).to(self.device)
            m = masks.to(self.device)
            with torch.no_grad():
                logits, values = self.model(x)
                logits = logits.masked_fill(~m, -1e9)
                probs = torch.softmax(logits, dim=-1)

                if temperature > 0 and temperature != 1.0:
                    # 温度化采样分布; log_prob 必须取自同一分布, 否则
                    # PPO ratio 系统性偏大 -> 策略退化 (已实测)
                    p = probs.clamp(min=1e-9) ** (1.0 / temperature)
                    p = p / p.sum(dim=-1, keepdim=True)
                    # T>1 时 clamp 下限被幂放大, 非法动作重新获得非零概率
                    # (曾导致 "手里没有这张牌" 崩溃), 必须重新掩零
                    p = p * m.float()
                    p = p / p.sum(dim=-1, keepdim=True)
                    actions = torch.multinomial(p, 1).squeeze(-1)
                    log_probs = torch.log(p.gather(
                        1, actions.unsqueeze(1)).squeeze(1).clamp(min=1e-9))
                elif temperature == 1.0:
                    actions = torch.multinomial(probs, 1).squeeze(-1)
                    log_probs = torch.log(probs.gather(
                        1, actions.unsqueeze(1)).squeeze(1).clamp(min=1e-9))
                else:
                    # 贪心 (评估)
                    actions = probs.argmax(dim=-1)
                if self.grp_model is not None:
                    if self.grp_version == 3:
                        # GRP3: v2特征 + 相对杠分5维; 输出[0]=决策者得分
                        from .grp_train3 import gang_scores_so_far
                        gang_feats = []
                        for i in model_idxs:
                            g = self.games[i]
                            seat = g.turn
                            gs = gang_scores_so_far(g)
                            gang_feats.append(
                                [gs[(seat + r) % 4] / 12.0 for r in range(4)]
                                + [len(g.gang_records) / 8.0])
                        x_grp = torch.cat([
                            x,
                            torch.tensor(np.asarray(
                                gang_feats, dtype=np.float32),
                                device=self.device)], dim=1)
                        grp_pred, _ = self.grp_model(x_grp)  # (B,4) 相对
                    else:
                        grp_pred = self.grp_model(x)  # (B,4) 绝对座位

            actions_cpu = actions.cpu().numpy()
            log_probs_cpu = log_probs.cpu().numpy() if temperature > 0 \
                else np.zeros(len(actions_cpu), dtype=np.float32)
            values_cpu = values.cpu().numpy()
            masks_cpu = m.cpu().numpy()
            grp_cpu = grp_pred.cpu().numpy() if self.grp_model is not None \
                else None

            for j, i in enumerate(model_idxs):
                g = self.games[i]
                seat = g.turn
                tile = int(actions_cpu[j])
                # 计算规则Bot评分 (regret = best - chosen)
                hand = g.players[seat].hand_counts
                chosen_score = _rule_score(g, seat, tile)
                best_score = max(
                    (_rule_score(g, seat, t) for t in range(28) if hand[t] > 0),
                    default=chosen_score)
                regret = best_score - chosen_score
                self.records[i].append((
                    seat, feats[j], tile,
                    float(log_probs_cpu[j]),
                    float(values_cpu[j]),
                    float(regret),
                    masks_cpu[j].copy(),
                ))
                if grp_cpu is not None:
                    # GRP3 相对输出: [0]=决策者; GRP2 绝对输出: [seat]
                    v = grp_cpu[j, 0] if self.grp_version == 3 \
                        else grp_cpu[j, seat]
                    self.grp_vals[i].append(float(v))
                g.action_discard(seat, tile)

        # 规则Bot出牌
        for i in rule_idxs:
            g = self.games[i]
            seat = g.turn
            tile = self.bots[i][seat].choose_discard()
            g.action_discard(seat, tile)

    def _step_react(self):
        idxs = [i for i, g in enumerate(self.games)
                if g.phase == "react_wait"]
        for i in idxs:
            g = self.games[i]
            seat = list(g.pending_actions.keys())[0]
            b = self.bots[i][seat]
            if g.pending_actions[seat].get("gang") and \
                    b.decide_gang(g.last_discard, "ming"):
                g.action_gang(seat)
            elif g.pending_actions[seat].get("peng") and \
                    b.decide_peng(g.last_discard):
                g.action_peng(seat)
            else:
                g.action_pass(seat)

    def _collect_results(self) -> list[dict]:
        results = []
        for i, g in enumerate(self.games):
            scores = [p.score_delta for p in g.players]
            results.append({
                "records": self.records[i],
                "scores": scores,
                "winner": g.winner,
                "grp_vals": self.grp_vals[i],
            })
        return results
