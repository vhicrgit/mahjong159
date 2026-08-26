"""安康159 - 学习者视角教师数据(1v3): 座位0用强搜索Bot, 其余用v1规则Bot

相对 gen_offline 4家同版本:
- 只有座位0跑搜索(v4) → 单局只1家搜索, 比4家self-play快~4倍
- 只记录座位0(学习者)的决策 → 数据分布 = 1v3评估分布, 完全匹配
- 用于 AlphaZero 第一轮蒸馏的快速验证 + 正式教师数据

用法:
  V4_W=12 V4_H=10 python -m backend.rl.gen_learner --games 15000 \
      --workers 40 --out models/offline_v4_learner.npz
"""

import multiprocessing as mp
import time

import numpy as np

from ..game.engine import Game
from ..ai.bot_v1 import Bot as V1
from .features_v2 import encode_state


def _get_encoder():
    """FEAT=og → 盲打v3+oracle块(1146维, 渐进Oracle Guiding用); 默认 v2(628)"""
    import os
    if os.environ.get("FEAT") == "og":
        from .features_og import encode_state as enc, FEAT_DIM
        return enc, FEAT_DIM
    from .features_v2 import FEAT_DIM
    return encode_state, FEAT_DIM


def _make_learner(g, seat):
    """teacher 由环境变量 TEACHER 选择:
       oracle = 完美信息(读牌墙)决策做标签, NN 只看盲打特征 → Oracle Guiding
       v4     = 搜索精修(已证伪, 保留对照)"""
    import os
    teacher = os.environ.get("TEACHER", "oracle")
    if teacher == "oracle":
        from ..ai.bot_oracle import Bot as Oracle
        return Oracle(g, seat, beam=int(os.environ.get("ORACLE_BEAM", 24)))
    from ..ai.bot_v4 import Bot as V4
    return V4(g, seat, worlds=int(os.environ.get("V4_W", 12)),
              horizon=int(os.environ.get("V4_H", 10)))


def play_one(seed: int):
    """座位0=teacher, 1-3=v1; 只记录座位0的出牌决策"""
    enc, feat_dim = _get_encoder()
    g = Game(seed=seed, human_seat=-1)
    learner = _make_learner(g, 0)
    bots = {0: learner, 1: V1(g, 1), 2: V1(g, 2), 3: V1(g, 3)}
    records = []  # (feat, act) —— 仅座位0

    guard = 0
    while g.phase != "game_over" and guard < 500:
        guard += 1
        if g.phase == "discard_wait":
            seat = g.turn
            if seat == 0:
                feat = enc(g, 0)
                tile = learner.choose_discard()
                records.append((feat, tile))
                g.action_discard(0, tile)
            else:
                g.action_discard(seat, bots[seat].choose_discard())
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

    ret0 = float(g.players[0].score_delta)
    n = len(records)
    feats = np.stack([r[0] for r in records]).astype(np.float32) \
        if records else np.zeros((0, feat_dim), dtype=np.float32)
    seats = np.zeros(n, dtype=np.int64)
    acts = np.array([r[1] for r in records], dtype=np.int64)
    rets = np.full(n, ret0, dtype=np.float32)
    return feats, seats, acts, rets


def _w(seed):
    return play_one(seed)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=15000)
    ap.add_argument("--seed0", type=int, default=40000000)
    ap.add_argument("--workers", type=int, default=40)
    ap.add_argument("--out", type=str, default="models/offline_v4_learner.npz")
    ap.add_argument("--flush-every", type=int, default=3000,
                    help="每 N 局落盘一次(可中途取用部分数据)")
    args = ap.parse_args()

    t0 = time.time()
    acc_f, acc_s, acc_a, acc_r = [], [], [], []
    done = 0
    with mp.Pool(min(args.workers, args.games)) as pool:
        for f, s, a, r in pool.imap_unordered(
                _w, range(args.seed0, args.seed0 + args.games),
                chunksize=4):
            acc_f.append(f)
            acc_s.append(s)
            acc_a.append(a)
            acc_r.append(r)
            done += 1
            if done % args.flush_every == 0:
                _save(args.out, acc_f, acc_s, acc_a, acc_r)
                print(f"  中途落盘 {done}/{args.games} 局, "
                      f"{sum(len(x) for x in acc_a)} 样本, "
                      f"{time.time()-t0:.0f}s", flush=True)
    _save(args.out, acc_f, acc_s, acc_a, acc_r)
    print(f"完成 {args.games} 局, {sum(len(x) for x in acc_a)} 样本, "
          f"耗时 {time.time()-t0:.0f}s")


def _save(out, fs, ss, as_, rs):
    np.savez_compressed(
        out,
        feats=np.concatenate(fs), seats=np.concatenate(ss),
        acts=np.concatenate(as_), rets=np.concatenate(rs))


if __name__ == "__main__":
    main()
