"""安康159 - 自对弈数据生成

用规则Bot自对弈, 记录每个出牌决策点:
  (局面特征, 实际打出的牌, 该玩家本局最终收益)
用于行为克隆(BC)热身 + AWR 强化。

数据格式: npz, 字段:
  feats:  (N, 512) float32
  acts:   (N,)     int64    打出的牌 id
  rets:   (N,)     float32  局终收益
"""

import multiprocessing as mp
import numpy as np
import torch

from ..game.engine import Game
from ..ai.bot import Bot
from .features_v2 import encode_state
from .model import legal_discard_mask

_N_WORKERS = min(mp.cpu_count(), 32)


def _net_chooser_factory(model, temperature=1.0):
    """返回一个 choose_discard 函数, 用模型出牌(带温度采样)"""
    device = next(model.parameters()).device

    def choose(game, seat):
        feat = encode_state(game, seat)
        x = torch.from_numpy(feat).unsqueeze(0).to(device)
        mask = legal_discard_mask(
            game.players[seat].hand_counts).unsqueeze(0).to(device)
        with torch.no_grad():
            probs = model.policy(x, mask)[0].cpu()
        if temperature <= 0:
            return int(probs.argmax().item())
        p = probs.clamp(min=1e-9) ** (1.0 / temperature)
        p = p / p.sum()
        return int(torch.multinomial(p, 1).item())
    return choose


def play_one_game(seed: int, chooser=None) -> tuple[list[np.ndarray], list[int], list[float]]:
    """自对弈一局, 返回所有决策点的 (特征, 动作) 及各玩家局终收益

    chooser: 可选, choose(game, seat) -> tile; 为 None 时用规则Bot
    """
    g = Game(seed=seed, human_seat=-1)
    bots = {i: Bot(g, i) for i in range(4)}
    records: list[list] = []  # (seat, feat, act)

    guard = 0
    while g.phase != "game_over" and guard < 500:
        guard += 1
        if g.phase == "discard_wait":
            seat = g.turn
            feat = encode_state(g, seat)
            tile = chooser(g, seat) if chooser else bots[seat].choose_discard()
            records.append((seat, feat, tile))
            g.action_discard(seat, tile)
        elif g.phase == "react_wait":
            s = list(g.pending_actions.keys())[0]
            b = bots[s]
            if g.pending_actions[s].get("gang") and \
                    b.decide_gang(g.last_discard, "ming"):
                g.action_gang(s)
            elif g.pending_actions[s].get("peng") and \
                    b.decide_peng(g.last_discard):
                g.action_peng(s)
            else:
                g.action_pass(s)

    scores = [p.score_delta for p in g.players]
    feats, acts, rets = [], [], []
    for seat, feat, act in records:
        feats.append(feat)
        acts.append(act)
        rets.append(float(scores[seat]))
    return feats, acts, rets


def _play_one_game_wrapper(args):
    """multiprocessing worker wrapper"""
    seed, model_state_dict, size, temperature = args
    if model_state_dict is not None:
        from .model import build_model
        device = torch.device("cuda" if torch.cuda.is_available() and not True else "cpu")
        model = build_model(size)
        model.load_state_dict(model_state_dict)
        model.eval()
        chooser = _net_chooser_factory(model, temperature)
    else:
        chooser = None
    return play_one_game(seed, chooser=chooser)


def generate_dataset(n_games: int, seed0: int = 0, chooser=None,
                     workers: int = 0) -> dict:
    """生成 n_games 局自对弈数据

    chooser: None 时用规则Bot(并行), callable 时用模型采样(串行GPU)
    """
    workers = workers or _N_WORKERS

    if chooser is None and workers > 1:
        # 规则Bot: 并行多进程
        tasks = [(seed0 + i, None, None, 0.0) for i in range(n_games)]
        n_workers = min(workers, n_games)
        with mp.Pool(n_workers) as pool:
            results = list(pool.imap_unordered(
                _play_one_game_wrapper, tasks,
                chunksize=max(1, n_games // n_workers // 4)))
            print(f"  并行生成完成: {n_games} 局")
        all_feats, all_acts, all_rets = [], [], []
        for f, a, r in results:
            all_feats.extend(f)
            all_acts.extend(a)
            all_rets.extend(r)
    else:
        # 模型采样(串行, GPU)
        all_feats, all_acts, all_rets = [], [], []
        for i in range(n_games):
            f, a, r = play_one_game(seed0 + i, chooser=chooser)
            all_feats.extend(f)
            all_acts.extend(a)
            all_rets.extend(r)
            if (i + 1) % 50 == 0:
                print(f"  已生成 {i + 1}/{n_games} 局, 样本 {len(all_acts)}")

    return {
        "feats": np.stack(all_feats).astype(np.float32),
        "acts": np.asarray(all_acts, dtype=np.int64),
        "rets": np.asarray(all_rets, dtype=np.float32),
    }


if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    out = sys.argv[2] if len(sys.argv) > 2 else "selfplay_data.npz"
    data = generate_dataset(n, workers=8)
    np.savez_compressed(out, **data)
    print(f"已保存 {out}: {data['acts'].shape[0]} 个决策样本")
