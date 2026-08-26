"""安康159 - World-GRPO 原型: 多世界组内比较的反事实动作评估

流程(验证信号阶段):
1. v10 自对弈采样座位0的出牌决策点(快照整个 Game)
2. 该局面取 v10 评分 top-m 候选弃牌
3. 采样 n 个**共享世界**: 未见牌重新分配给对手暗牌(尺寸固定)+牌墙,
   m 个候选在**同一组世界**里推演(共同随机数, 消世界运气噪声)
4. 每个 (候选, 世界) 用快速策略(v1)推演到底, 记塑形奖励(想法1)
5. 组内优势 A_a = mean_w r(a,w) - mean_a'; 报告与 v10 选择的一致率

用法: python -m backend.rl.world_grpo --states 64 --worlds 32 --top-m 4 --procs 40
"""

import argparse
import copy
import multiprocessing as mp
import random

import numpy as np

from ..game.engine import Game
from ..ai.bot_v1 import Bot as BotV1
from ..ai.bot_v10 import Bot as BotV10, _add, _ukeire, _second_step_value
from ..rules.ting import discard_options
from .gen_offline import _shaped_scores

RED = 27


def _v10_scores(g: Game, seat: int) -> dict[int, float]:
    """复刻 v10 的出牌评分(默认权重), 返回 {tile: score}。"""
    b = BotV10(g, seat)
    p = g.players[seat]
    counts14 = tuple(p.hand_counts)
    opts = discard_options(list(counts14))
    unseen = b._unseen_counts()
    penged = b._penged_by_others()
    eg = b._endgame_factor()
    min_sh = min(o["shanten"] for o in opts)
    scores = {}
    for o in opts:
        t = o["tile"]
        h = _add(counts14, t, -1)
        s = o["shanten"]
        if s > min_sh:
            score = -10.0 * b.shanten_weight - b.shanten_weight * s
        else:
            u = _ukeire(h, unseen)
            cont = _second_step_value(h, unseen) if s <= b.cont_max_shanten else 0.0
            score = b.ukeire_weight * u + b.cont_weight * cont
        if t != RED:
            risk = 1.0 if t in penged else \
                {3: 0.4, 2: 0.2, 1: 0.05, 0: 0.0}.get(unseen[t], 0.4)
            score -= b.risk_weight * (1.0 + 1.5 * eg) * risk
        scores[t] = score
    return scores


def collect_state(seed: int, skip_turns: int = 4):
    """v10 自对弈, 返回座位0某决策点的 (快照, v10评分表)。
    快照前清空 log 减小 deepcopy 开销。"""
    rng = random.Random(seed ^ 0x9e3779b9)
    g = Game(seed=seed, human_seat=-1)
    bots = {i: BotV10(g, i) for i in range(4)}
    target_turn = rng.randint(3, 10)
    turn_count = 0
    guard = 0
    while g.phase != "game_over" and guard < 500:
        guard += 1
        if g.phase == "discard_wait":
            if g.turn == 0:
                turn_count += 1
                if turn_count == target_turn:
                    opts = discard_options(list(g.players[0].hand_counts))
                    if len(opts) >= 3:
                        scores = _v10_scores(g, 0)
                        g.log = []
                        return copy.deepcopy(g), scores
            g.action_discard(g.turn, bots[g.turn].choose_discard())
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
    return None, None


def sample_world(snap: Game, rng: random.Random, hero_seat: int = 0):
    """重洗未见牌: 对手暗牌(尺寸不变) + 剩余牌墙。返回 (hands, wall)。
    hero_seat 的暗牌保持真实(全知者视角的基准)。"""
    unseen = []
    visible = [0] * 28
    for q in snap.players:
        for t in q.discards:
            visible[t] += 1
        for m in q.melds:
            visible[m["tile"]] += 3 if m["type"] == "peng" else 4
    me = snap.players[hero_seat]
    for t, c in enumerate(me.hand_counts):
        visible[t] += c
    for t in range(28):
        unseen += [t] * (4 - visible[t])
    rng.shuffle(unseen)
    pos = 0
    hands = {}
    for q in snap.players:
        if q.seat == hero_seat:
            continue
        need = len(q.hand)
        hands[q.seat] = sorted(unseen[pos:pos + need])
        pos += need
    wall = unseen[pos:]
    return hands, wall


def rollout_candidate(snap: Game, tile: int, hands: dict, wall: list,
                      step_penalty: float, hero_seat: int = 0) -> float:
    """注入世界, 打出候选牌, 四家 v1 推演到底, 返回 hero 塑形得分。"""
    g = copy.deepcopy(snap)
    for seat, h in hands.items():
        g.players[seat].hand = list(h)
    g.wall = list(wall)
    bots = {i: BotV1(g, i) for i in range(4)}
    g.action_discard(hero_seat, tile)
    guard = 0
    while g.phase != "game_over" and guard < 500:
        guard += 1
        if g.phase == "discard_wait":
            g.action_discard(g.turn, bots[g.turn].choose_discard())
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
    return float(_shaped_scores(g, step_penalty)[hero_seat])


def eval_state(args):
    seed, top_m, n_worlds, step_penalty = args
    snap, scores = collect_state(seed)
    if snap is None:
        return None
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    cands = [t for t, _ in ranked[:top_m]]
    rng = random.Random(seed ^ 0x51ab3f)
    worlds = [sample_world(snap, rng) for _ in range(n_worlds)]
    means = {t: float(np.mean([
        rollout_candidate(snap, t, hands, wall, step_penalty)
        for hands, wall in worlds])) for t in cands}
    v10_pick = ranked[0][0]
    grpo_pick = max(means.items(), key=lambda kv: kv[1])[0]
    return {
        "seed": seed,
        "cands": cands,
        "means": means,
        "v10_pick": v10_pick,
        "grpo_pick": grpo_pick,
        "agree": v10_pick == grpo_pick,
        "spread": max(means.values()) - min(means.values()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", type=int, default=64)
    ap.add_argument("--worlds", type=int, default=32)
    ap.add_argument("--top-m", type=int, default=4)
    ap.add_argument("--procs", type=int, default=40)
    ap.add_argument("--seed0", type=int, default=710000)
    ap.add_argument("--step-penalty", type=float, default=0.02)
    args = ap.parse_args()

    tasks = [(args.seed0 + i, args.top_m, args.worlds, args.step_penalty)
             for i in range(args.states)]
    with mp.Pool(args.procs) as pool:
        results = [r for r in pool.imap_unordered(eval_state, tasks)
                   if r is not None]
    n = len(results)
    agree = sum(r["agree"] for r in results)
    spreads = [r["spread"] for r in results]
    print(f"有效局面 {n}/{args.states} (每局面 {args.top_m}候选 × "
          f"{args.worlds}共享世界):")
    print(f"  GRPO-argmax 与 v10 一致率: {agree/n:.1%}")
    print(f"  组内优势极差 spread: mean {np.mean(spreads):.3f} "
          f"(塑形奖励尺度)")
    dis = [r for r in results if not r["agree"]]
    for r in dis[:8]:
        print(f"  分歧 seed={r['seed']}: v10选{r['v10_pick']} "
              f"GRPO选{r['grpo_pick']} means="
              f"{ {t: round(v,2) for t,v in r['means'].items()} }")


if __name__ == "__main__":
    main()
