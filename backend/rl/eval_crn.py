"""CRN(共同随机数)配对评估 —— 自对弈进化的标尺。

为什么必须配对: 终局得分里 87~90% 的方差是牌运, 而两条臂共享同一副牌墙时
这部分是同一份, 差分就消掉了。实测(tools/perf/diag_crn.py)降方差倍数
= 1/(1-ρ): 相邻检查点档 ρ≈0.95~0.97 -> 20~37x, 远距离策略 ρ≈0.5 -> 2x。
不配对的评估在 96 局下标准误 4.4%, 根本看不见 2~3% 的真实进步。

同时把候选轮坐四个座位(座位偏差实测可达 0.59 分/局), 基线臂用"四家全对手"
的一局同时拿到四个座位的分, 所以每 seed 只要 4+1 局。
"""

import numpy as np

from ..ai.bot_native import NativeV31
from ..game.engine import Game
from ..game.bot_driver import try_self_gang

PROTOCOL = "first_or_bloody_full_actions_seed_cluster_v2"


def _play(seed, bloody, factories):
    """factories: {seat: make_bot(game, seat)}。返回终局的 Game。"""
    g = Game(seed=seed, human_seat=-1, bloody=bloody)
    bots = {s: f(g, s) for s, f in factories.items()}
    guard = 0
    while g.phase != "game_over" and guard < 900:
        guard += 1
        if g.phase == "discard_wait":
            s = g.turn
            if try_self_gang(g, bots[s]):
                continue
            g.action_discard(s, bots[s].choose_discard())
        else:
            s = list(g.pending_actions.keys())[0]
            pend = g.pending_actions[s]
            b = bots[s]
            if pend.get("gang") and b.decide_gang(g.last_discard, "ming"):
                g.action_gang(s)
            elif pend.get("peng") and b.decide_peng(g.last_discard):
                g.action_peng(s)
            else:
                g.action_pass(s)
    if g.phase != "game_over":
        raise RuntimeError("Evaluation exceeded action limit before terminal state")
    return g


def _adjusted(g, seat):
    """放杠不罚(与训练奖励口径一致)。"""
    v = float(g.players[seat].score_delta)
    for rec in g.gang_records:
        if rec["kind"] == "ming" and rec["from"] == seat:
            v += 3
    return v


def rotate_arm(make_bot, make_opp, seeds, bloody, raw_score=False):
    """候选轮坐四个座位, 其余三家用 make_opp。返回 (n,4) 的名次奖励与得分。"""
    rr = np.zeros((len(seeds), 4))
    sc = np.zeros((len(seeds), 4))
    for i, seed in enumerate(seeds):
        for s in range(4):
            fac = {k: make_opp for k in range(4)}
            fac[s] = make_bot
            g = _play(seed, bloody, fac)
            rr[i, s] = g.rank_rewards()[s]
            sc[i, s] = g.players[s].score_delta if raw_score else _adjusted(g, s)
    return rr, sc


def baseline_arm(make_opp, seeds, bloody, raw_score=False):
    """四家全用 make_opp 的一局同时给出四个座位的基线 —— 省 4x。"""
    rr = np.zeros((len(seeds), 4))
    sc = np.zeros((len(seeds), 4))
    for i, seed in enumerate(seeds):
        g = _play(seed, bloody, {k: make_opp for k in range(4)})
        r = g.rank_rewards()
        a = [g.players[s].score_delta if raw_score else _adjusted(g, s)
             for s in range(4)]
        for s in range(4):
            rr[i, s] = r[s]
            sc[i, s] = a[s]
    return rr, sc


def _stat(d):
    """Rows are independent seeds, columns are correlated seat rotations."""
    d = np.asarray(d, dtype=float)
    observations = d.size
    if d.ndim == 2:
        d = d.mean(axis=1)
    elif d.ndim != 1:
        raise ValueError("Expected one value per seed or (seeds, seats)")
    if not d.size or not np.isfinite(d).all():
        raise ValueError("Expected nonempty finite paired returns")
    n = len(d)
    se = d.std(ddof=1) / np.sqrt(n) if n > 1 else float("nan")
    return {"mean": float(d.mean()), "se": float(se), "n": n,
            "observations": observations,
            "ci95_normal": [float(d.mean() - 1.96 * se), float(d.mean() + 1.96 * se)],
            "t": float(d.mean() / se) if se and se > 0 else
                 (0.0 if se == 0 and d.mean() == 0 else float("nan"))}


def paired_vs_v31(make_bot, seeds, bloody=True, raw_score=False):
    """候选 vs v31n 的配对差分(基线臂 = 全 v31n, 一局拿四座位)。"""
    rr_a, sc_a = rotate_arm(make_bot, NativeV31, seeds, bloody, raw_score)
    rr_b, sc_b = baseline_arm(NativeV31, seeds, bloody, raw_score)
    return {"rank": _stat(rr_a - rr_b), "score": _stat(sc_a - sc_b),
            "cand_rank_mean": float(rr_a.mean()),
            "cand_score_mean": float(sc_a.mean())}


def paired_head2head(make_a, make_b, seeds, make_opp=NativeV31, bloody=True,
                     raw_score=False):
    """A 与 B 各自轮坐四个座位(对手相同), 配对差分 = A 强于 B 多少分/局。"""
    rr_a, sc_a = rotate_arm(make_a, make_opp, seeds, bloody, raw_score)
    rr_b, sc_b = rotate_arm(make_b, make_opp, seeds, bloody, raw_score)
    return {"rank": _stat(rr_a - rr_b), "score": _stat(sc_a - sc_b)}
