"""诊断 GRPO 的信噪比: 组内优势到底是信号还是噪声?

做法: 对每个局面, 用 W_max 个共享世界把每个候选的**逐世界回报**全部算出来,
存成 (候选 × 世界) 矩阵, 然后离线做:

  1. 逐世界回报的标准差 σ, 以及候选均值的标准误 σ/√W
  2. 候选间极差 spread = max_a r̄_a − min_a r̄_a
  3. **对半复现性**: 把 W 个世界随机劈成两半, 各自算候选均值,
     看两半给出的排序/argmax 是否一致。这是最直接的检验 ——
     如果两半互不相关, 那训练用的优势就是噪声。
  4. 需要多少世界才能让 argmax 稳定(用 bootstrap 估)

理论依据: 策略梯度 E[A·∇logπ] 的信噪比 ∝ Δ/(σ/√W), Δ 是候选间真实价值差。
Δ 固定时把 W 翻 4 倍只让 SNR 翻 2 倍; 若 Δ 本身接近 0, 加世界数没用,
必须改 credit assignment(缩短 horizon / 换更贴近动作的塑形)。
"""

import argparse
import copy
import itertools
import multiprocessing as mp
import random
import statistics

import numpy as np

from backend.game.engine import Game
from backend.rl.gen_offline import _shaped_scores
from backend.rl.world_grpo import sample_world
from backend.rules.ting import discard_options


def _worker(task):
    """返回该候选在 n_worlds 个共享世界上的逐世界回报向量。"""
    snap, seat, tile, world_seed, n_worlds, step_penalty, bot_type = task
    from backend.rl.grpo_train import _BOT_REGISTRY
    cfg = _BOT_REGISTRY[bot_type]
    bot_cls, bot_kwargs = cfg["cls"], cfg["kwargs"]
    rng = random.Random(world_seed)
    out = []
    for _ in range(n_worlds):
        hands, wall = sample_world(snap, rng, hero_seat=seat)
        g = copy.deepcopy(snap)
        for s2, h in hands.items():
            g.players[s2].hand = list(h)
        g.wall = list(wall)
        bots = {i: bot_cls(g, i, **bot_kwargs) for i in range(4)}
        g.action_discard(seat, tile)
        guard = 0
        while g.phase != "game_over" and guard < 500:
            guard += 1
            if g.phase == "discard_wait":
                g.action_discard(g.turn, bots[g.turn].choose_discard())
            else:
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
        out.append(float(_shaped_scores(g, step_penalty)[seat]))
    return out


def make_snaps(n, seed0):
    from backend.ai.bot_native import NativeV10
    out, gi = [], 0
    while len(out) < n and gi < n * 40:
        seed = seed0 + gi
        gi += 1
        g = Game(seed=seed, human_seat=-1)
        bots = {i: NativeV10(g, i) for i in range(4)}
        rng = random.Random(seed ^ 0xC0FFEE)
        target = rng.randint(3, 13)
        tc, guard = 0, 0
        while g.phase != "game_over" and guard < 500:
            guard += 1
            if g.phase == "discard_wait":
                tc += 1
                seat = g.turn
                if tc == target and len(discard_options(
                        list(g.players[seat].hand_counts))) >= 4:
                    g.log = []
                    out.append((copy.deepcopy(g), seat))
                    break
                g.action_discard(seat, bots[seat].choose_discard())
            else:
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
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", type=int, default=8)
    ap.add_argument("--worlds", type=int, default=1024)
    ap.add_argument("--top-m", type=int, default=4)
    ap.add_argument("--bot", default="v31n")
    ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--step-penalty", type=float, default=0.02)
    ap.add_argument("--seed0", type=int, default=880000)
    ap.add_argument("--boot", type=int, default=400)
    args = ap.parse_args()

    snaps = make_snaps(args.states, args.seed0)
    print(f"局面 {len(snaps)} 个, 每个 {args.top_m} 候选 × {args.worlds} 共享世界, "
          f"bot={args.bot}")

    tasks, meta = [], []
    for si, (g, seat) in enumerate(snaps):
        opts = discard_options(list(g.players[seat].hand_counts))
        tiles = [o["tile"] for o in opts][:args.top_m]
        ws = args.seed0 + 7919 + si          # 同一局面所有候选共用世界种子
        for t in tiles:
            tasks.append((g, seat, t, ws, args.worlds, args.step_penalty,
                          args.bot))
            meta.append((si, t))
    with mp.Pool(args.procs) as pool:
        rets = pool.map(_worker, tasks, chunksize=1)

    R = {}
    for (si, t), v in zip(meta, rets):
        R.setdefault(si, {})[t] = np.array(v)

    rng = np.random.default_rng(12345)
    print(f"\n{'局面':>4s} {'候选':>4s} {'σ(逐世界)':>10s} {'spread':>8s} "
          f"{'SE(W=128)':>10s} {'SNR':>6s} {'对半argmax一致':>14s} "
          f"{'对半相关r':>10s}")
    print("-" * 82)
    snr_all, half_agree, half_corr, sig_all = [], [], [], []
    for si in sorted(R):
        mat = np.stack([R[si][t] for t in sorted(R[si])])   # (A, W)
        A, W = mat.shape
        sigma = float(np.mean(mat.std(axis=1, ddof=1)))
        means = mat.mean(axis=1)
        spread = float(means.max() - means.min())
        se128 = sigma / np.sqrt(128)
        snr = spread / (se128 * np.sqrt(2))
        # 对半复现性: 用前 128 个世界(训练实际用量)劈两半, 重复 boot 次
        agree, corrs = 0, []
        for _ in range(args.boot):
            idx = rng.permutation(W)[:128]
            h1, h2 = idx[:64], idx[64:]
            m1, m2 = mat[:, h1].mean(axis=1), mat[:, h2].mean(axis=1)
            if int(np.argmax(m1)) == int(np.argmax(m2)):
                agree += 1
            if m1.std() > 0 and m2.std() > 0:
                corrs.append(float(np.corrcoef(m1, m2)[0, 1]))
        ag = agree / args.boot
        cr = float(np.mean(corrs)) if corrs else float("nan")
        snr_all.append(snr)
        half_agree.append(ag)
        half_corr.append(cr)
        sig_all.append(sigma)
        print(f"{si:4d} {A:4d} {sigma:10.3f} {spread:8.3f} {se128:10.3f} "
              f"{snr:6.2f} {ag:14.1%} {cr:10.3f}")

    print(f"\n汇总: σ均值 {statistics.mean(sig_all):.3f}, "
          f"SNR均值 {statistics.mean(snr_all):.2f}, "
          f"对半 argmax 一致率 {statistics.mean(half_agree):.1%}, "
          f"对半相关 {statistics.mean(half_corr):+.3f}")
    print(f"随机猜的一致率基线 = 1/A = {1/args.top_m:.1%}")

    # 需要多少世界才能让 argmax 稳定到 90%
    print("\n不同世界数下的对半 argmax 一致率(用同一批数据子采样):")
    for w in (16, 32, 64, 128, 256, 512):
        if 2 * w > args.worlds:
            break
        accs = []
        for si in sorted(R):
            mat = np.stack([R[si][t] for t in sorted(R[si])])
            ok = 0
            for _ in range(args.boot):
                idx = rng.permutation(mat.shape[1])[:2 * w]
                m1 = mat[:, idx[:w]].mean(axis=1)
                m2 = mat[:, idx[w:]].mean(axis=1)
                ok += int(np.argmax(m1)) == int(np.argmax(m2))
            accs.append(ok / args.boot)
        print(f"  W={w:4d}: {statistics.mean(accs):.1%}")

    # 真实价值差有多大: 用全部世界当 ground truth
    print("\n以全 W 均值为准的候选价值差(Δ):")
    gaps = []
    for si in sorted(R):
        mat = np.stack([R[si][t] for t in sorted(R[si])])
        m = np.sort(mat.mean(axis=1))[::-1]
        gaps.append(float(m[0] - m[1]))
    print(f"  最优与次优的差 Δ: 均值 {statistics.mean(gaps):.4f}, "
          f"中位 {statistics.median(gaps):.4f}, 最大 {max(gaps):.4f}")
    s = statistics.mean(sig_all)
    print(f"  要让 SE < Δ/2 需要 W ≈ (2σ/Δ)² = "
          f"{(2*s/max(statistics.mean(gaps),1e-9))**2:.0f} 个世界")


if __name__ == "__main__":
    main()
