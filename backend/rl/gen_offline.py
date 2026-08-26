"""安康159 - 离线 DQN 数据生成: (特征, 动作, 4家最终得分, 胜者)

规则Bot自对弈 + 记录动作, 用于离线 Q-learning (Mortal 路线):
  Q(s, a_taken) -> MC return (决策者本人最终得分)
  Q-greedy 可隐式超越行为策略(规则Bot)

用法:
  python -m backend.rl.gen_offline --games 200000 --workers 12
"""

import multiprocessing as mp
import os
import time

import numpy as np

from ..game.engine import Game
from ..rules.tiles import is_159


def _shaped_scores(g: Game, step_penalty: float) -> np.ndarray:
    """塑形奖励: 去除159翻牌的纯运气噪声 + 加速激励。

    - 胡牌分: 实际翻到的 n 替换为条件期望 E[n|牌墙] = 6 × 牌墙中159密度
      (Rao-Blackwell: 只削翻牌抽样噪声, 保留牌墙结构的策略相关性)
    - 杠分: 确定性真实收益, 保留原值
    - 步数惩罚: 每人 -δ×总摸牌数 (自摸竞速激励; 对胜负差是公共项,
      但通过压低长局价值鼓励快速成牌)
    """
    scores = np.zeros(4, dtype=np.float64)
    # 杠分(黄庄不结, 与引擎口径一致: 只在有 winner 时结算)
    if g.winner is not None:
        for rec in g.gang_records:
            s = rec["seat"]
            if rec["kind"] == "ming":
                scores[rec["from"]] -= 3
                scores[s] += 3
            else:
                for o in range(4):
                    if o != s:
                        scores[o] -= 1
                        scores[s] += 1
        # 胡牌分用条件期望
        wall = g.wall  # 翻牌未从 wall 移除, 含被翻的6张
        e_n = 6.0 * sum(1 for t in wall if is_159(t)) / max(1, len(wall))
        per = e_n + 1
        for o in range(4):
            if o != g.winner:
                scores[o] -= per
                scores[g.winner] += per
    # 步数惩罚: 摸牌数 = 发牌后牌墙(59) - 剩余 (杠补牌也从 wall pop)
    draws = 59 - len(g.wall)
    scores -= step_penalty * draws
    return scores

def _get_bot_class(bot_version):
    if bot_version == 4:
        from ..ai.bot_v4 import Bot as B
    elif bot_version == 2:
        from ..ai.bot_v2 import Bot as B
    elif bot_version == 10:
        from ..ai.bot_v10 import Bot as B
    else:
        from ..ai.bot_v1 import Bot as B
    return B


def _make(bot_version, g, i):
    """v4 需要搜索参数(worlds/horizon 通过环境变量调, 默认省算力配置)"""
    import os
    B = _get_bot_class(bot_version)
    if bot_version == 4:
        return B(g, i, worlds=int(os.environ.get("V4_W", 12)),
                 horizon=int(os.environ.get("V4_H", 10)))
    return B(g, i)

from .features_v2 import encode_state, FEAT_DIM as FEAT_DIM_V2
from .features_v3 import encode_state as encode_state_v3, FEAT_DIM as FEAT_DIM_V3


def play_one(seed: int, bot_version: int = 1, feat_version: int = 2,
             shaped: bool = False, step_penalty: float = 0.02):
    enc = encode_state_v3 if feat_version == 3 else encode_state
    feat_dim = FEAT_DIM_V3 if feat_version == 3 else FEAT_DIM_V2
    g = Game(seed=seed, human_seat=-1)
    bots = {i: _make(bot_version, g, i) for i in range(4)}
    records = []  # (seat, feat, act)

    guard = 0
    while g.phase != "game_over" and guard < 500:
        guard += 1
        if g.phase == "discard_wait":
            seat = g.turn
            feat = enc(g, seat)
            tile = bots[seat].choose_discard()
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

    if shaped:
        scores = _shaped_scores(g, step_penalty)
    else:
        scores = np.array([p.score_delta for p in g.players],
                          dtype=np.float64)
    n = len(records)
    feats = np.stack([r[1] for r in records]).astype(np.float32) \
        if records else np.zeros((0, feat_dim), dtype=np.float32)
    seats = np.array([r[0] for r in records], dtype=np.int64)
    acts = np.array([r[2] for r in records], dtype=np.int64)
    rets = np.array([scores[r[0]] for r in records], dtype=np.float32)
    return feats, seats, acts, rets


def _w(args):
    seed, bot_version, feat_version, shaped, step_penalty = args
    return play_one(seed, bot_version, feat_version, shaped, step_penalty)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=200000)
    ap.add_argument("--seed0", type=int, default=5000000)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", type=str, default="models/offline_data.npz")
    ap.add_argument("--bot-version", type=int, default=1,
                    choices=[1, 2, 4, 10])
    ap.add_argument("--feat-version", type=int, default=2, choices=[2, 3])
    ap.add_argument("--shaped", action="store_true",
                    help="塑形奖励: 胡牌分用E[n|wall], 加步数惩罚")
    ap.add_argument("--step-penalty", type=float, default=0.02,
                    help="每张摸牌的步数惩罚 δ")
    args = ap.parse_args()

    t0 = time.time()
    tasks = [(s, args.bot_version, args.feat_version, args.shaped,
              args.step_penalty)
             for s in range(args.seed0, args.seed0 + args.games)]
    with mp.Pool(min(args.workers, args.games)) as pool:
        results = list(pool.imap_unordered(
            _w, tasks,
            chunksize=max(1, args.games // args.workers // 4)))
    feats = np.concatenate([r[0] for r in results])
    seats = np.concatenate([r[1] for r in results])
    acts = np.concatenate([r[2] for r in results])
    rets = np.concatenate([r[3] for r in results])
    print(f"生成 {args.games} 局, {len(acts)} 样本, "
          f"耗时 {time.time()-t0:.0f}s")
    print(f"动作分布: 均匀基线 {1/28:.3f}, "
          f"top1动作占比 {np.bincount(acts, minlength=28).max()/len(acts):.3f}")
    print(f"回报: mean={rets.mean():.3f} std={rets.std():.3f}")
    np.savez_compressed(args.out, feats=feats, seats=seats,
                        acts=acts, rets=rets)
    print(f"已保存 {args.out}")


if __name__ == "__main__":
    main()
