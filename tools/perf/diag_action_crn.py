"""动作级 CRN 的方差诊断: 配对 PPO 的 advantage 噪声能压下去多少。

背景: 条件方差分解(logs/diag_conditional_variance_temp1.json)显示 temp=1 自对弈里
55.5% 的回报方差来自策略自身的动作采样, 而任何基于状态的 critic(盲打或特权)都
消不掉它 —— 特权 critic 天花板只到 44.5%, advantage 噪声最多降到 1/1.34。

`tools/rl_paired_train.py` 的配对基线只在**牌墙**层面用共同随机数(同 seed), 两臂的
动作采样各自独立(而且基线臂默认贪心、不采样)。本脚本回答: 如果两臂的同名座位
共用同一串均匀随机数(按"该座位第 k 次决策"对齐, 经各自策略的逆 CDF 映射),
配对回报差 R_L - R_F 的方差能降多少?

三种口径:
  greedy  基线臂该座位贪心(现行默认 frozen_temp=0)
  indep   基线臂该座位同温独立采样(现行 --frozen-temp 的行为)
  crn     基线臂该座位同温采样, 且与训练臂共用同一串 u(按决策序号对齐)

crn 的性质: 两个策略越接近, 两局的动作序列就越久保持一致, 分歧只发生在策略真正
不同的地方 —— 估计量的方差随"策略差异"缩放, 而不是随"牌局内在随机性"缩放。
自检: 两臂用同一个检查点时, crn 的 sd 必须**恰好为 0**(逐局轨迹完全相同)。

用法:
  python -m tools.perf.diag_action_crn --pairs models/bc_r2_s3.pt models/bc_r2_s3.pt
  python -m tools.perf.diag_action_crn --pairs models/bc_k0_r2.pt models/bc_r2_s3.pt
"""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from backend.game.bot_driver import try_self_gang
from backend.game.engine import Game
from backend.rl import cf_collect
from backend.rl.features_v2 import encode_state
from backend.rl.model import build_model, legal_discard_mask
from tools.rl_paired_train import _claims, _react

STREAM = 200        # 每个 seed 预生成的均匀随机数个数(单座位一局远用不到这么多)


class HVSeat:
    """自家杠/碰杠交给分析器 E 判据, 与部署和评估口径一致。"""

    def __init__(self, game, seat):
        self.game, self.seat = game, seat
        self._hv = _claims(game, seat)

    def decide_gang(self, tile, kind):
        return self._hv.decide_gang(tile, kind)

    def decide_peng(self, tile):
        return self._hv.decide_peng(tile)


def load(path):
    ck = torch.load(path, map_location="cpu", weights_only=True)
    m = build_model(ck["size"], feat_dim=ck.get("feat_dim", 628))
    m.load_state_dict(ck["model"])
    return m.eval()


def sample(logits, mask, temp, u=None):
    """批量采样。temp<=0 贪心; u 给定时走逆 CDF(CRN 用), 否则 multinomial。"""
    lg = logits.masked_fill(~mask, -1e9)
    if temp <= 0:
        return lg.argmax(-1)
    p = F.softmax(lg / max(temp, 1e-6), dim=-1)
    if u is None:
        return torch.multinomial(p, 1).squeeze(-1)
    cum = p.cumsum(-1)
    u = u.clamp(0.0, 1.0 - 1e-7).unsqueeze(-1)
    return (cum <= u).sum(-1).clamp(max=p.shape[1] - 1)


def play_pairs(seeds, seat, learner, frozen, temp, mode, gang_w=0.25,
               bloody=True):
    """每个 seed 跑两局: L 臂(learner 坐 seat)与 F 臂(frozen 坐 seat)。

    两局其余三座都用 frozen 贪心 + 分析器 E 碰杠, 所以两局之间唯一的随机来源是
    牌墙(同 seed 已共享)与 seat 号座位的动作采样。返回每 seed 的配对回报差与
    两臂动作序列的分歧统计。
    """
    n = len(seeds)
    games, is_l = [], []
    for sd in seeds:
        games.append(Game(seed=sd, human_seat=-1, bloody=bloody))
        is_l.append(True)
        games.append(Game(seed=sd, human_seat=-1, bloody=bloody))
        is_l.append(False)
    uniforms = [torch.from_numpy(
        np.random.default_rng(1000003 * (sd % 99991) + 7).random(STREAM)
        .astype(np.float32)) for sd in seeds]
    cnt = [0] * len(games)          # 每局 seat 号座位已做的决策数
    acts = [[] for _ in range(len(games))]

    it = 0
    while it < 900:
        it += 1
        for g in games:
            if g.phase == "react_wait":
                _react(g)
            while g.phase == "discard_wait" and try_self_gang(g, HVSeat(g, g.turn)):
                pass
        idx = [i for i, g in enumerate(games) if g.phase == "discard_wait"]
        if not idx:
            break
        turns = {i: games[i].turn for i in idx}
        for group in ([i for i in idx if turns[i] != seat],
                      [i for i in idx if turns[i] == seat]):
            if not group:
                continue
            hero = turns[group[0]] == seat
            gs = [games[i] for i in group]
            ss = [turns[i] for i in group]
            feats = np.stack([encode_state(g, s) for g, s in zip(gs, ss)])
            masks = torch.stack([legal_discard_mask(g.players[s].hand_counts)
                                 for g, s in zip(gs, ss)])
            with torch.no_grad():
                a = torch.zeros(len(group), dtype=torch.long)
                if not hero:
                    q, _ = frozen(torch.from_numpy(feats))
                    a = sample(q, masks, 0.0)
                else:
                    # 两臂网络不同, 只能分别前向; CRN 的随机数按"该座位第 k 次
                    # 决策"对齐, 两臂各自的计数器独立推进但取同一个 u[k]
                    for tag in (True, False):
                        sel = [k for k, i in enumerate(group) if is_l[i] == tag]
                        if not sel:
                            continue
                        model = learner if tag else frozen
                        q, _ = model(torch.from_numpy(feats[sel]))
                        if mode == "crn":
                            # 两臂都按 temp 采样, 且共用同一个 u -> 逆 CDF 配对
                            t = temp
                            u = torch.stack([uniforms[group[k] // 2]
                                             [cnt[group[k]]] for k in sel])
                        elif tag:
                            t, u = temp, None      # 训练臂永远按 temp 采样
                        elif mode == "greedy":
                            t, u = 0.0, None
                        else:                      # indep: 同温但各自随机
                            t, u = temp, None
                        picked = sample(q, masks[sel], t, u)
                        for j, k in enumerate(sel):
                            a[k] = int(picked[j])
                    for k, i in enumerate(group):
                        cnt[i] += 1
                        if cnt[i] > STREAM:
                            raise RuntimeError("uniform stream exhausted")
                        acts[i].append(int(a[k]))
            for k, i in enumerate(group):
                games[i].action_discard(ss[k], int(a[k]))
    for g in games:
        if g.phase != "game_over":
            raise RuntimeError("rollout exceeded action limit")

    rows = []
    for si in range(n):
        gl, gf = games[si * 2], games[si * 2 + 1]
        al, af = acts[si * 2], acts[si * 2 + 1]
        common = min(len(al), len(af))
        first_div = next((k for k in range(common) if al[k] != af[k]), None)
        rows.append({
            "seed": seeds[si],
            "r_l": cf_collect.default_reward(gl, seat, gang_w),
            "r_f": cf_collect.default_reward(gf, seat, gang_w),
            "decisions_l": len(al), "decisions_f": len(af),
            "divergent_actions": sum(1 for k in range(common)
                                     if al[k] != af[k]),
            "first_divergence": first_div,
        })
        rows[-1]["adv"] = rows[-1]["r_l"] - rows[-1]["r_f"]
    return rows


def perturb(model, eps, seed=0):
    """对全部参数加相对幅度 eps 的高斯噪声 —— 制造一个"距离可控"的近邻策略。

    用来扫描 CRN 的收益随策略距离怎么变: 训练器每轮的 KL 预算是 0.02, 只有
    在这个量级上 CRN 有效才值得改训练器。
    """
    import copy
    m = copy.deepcopy(model)
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in m.parameters():
            p.add_(eps * float(p.std()) * torch.randn(p.shape, generator=g))
    return m.eval()


def policy_distance(learner, frozen, seeds, seat, n_states=400, temp=1.0):
    """在一批真实决策状态上量 KL(learner||frozen) 与两边策略的熵。"""
    states = []
    for sd in seeds:
        g = Game(seed=sd, human_seat=-1, bloody=True)
        bots = {s: HVSeat(g, s) for s in range(4)}
        guard = 0
        while g.phase != "game_over" and guard < 900 and len(states) < n_states:
            guard += 1
            if g.phase == "discard_wait":
                s = g.turn
                if try_self_gang(g, bots[s]):
                    continue
                if s == seat:
                    states.append((encode_state(g, s),
                                   legal_discard_mask(g.players[s].hand_counts)))
                g.action_discard(s, frozen_choose(frozen, g, s))
            else:
                _react(g)
        if len(states) >= n_states:
            break
    x = torch.from_numpy(np.stack([s[0] for s in states]))
    m = torch.stack([s[1] for s in states])
    with torch.no_grad():
        ql, _ = learner(x)
        qf, _ = frozen(x)
        pl = F.log_softmax(ql.masked_fill(~m, -1e9) / temp, -1)
        pf = F.log_softmax(qf.masked_fill(~m, -1e9) / temp, -1)
        kl = float(((pl.exp() * (pl - pf)).sum(-1)).mean())
        ent_l = float(-(pl.exp() * pl).sum(-1).mean())
        ent_f = float(-(pf.exp() * pf).sum(-1).mean())
        agree = float((pl.argmax(-1) == pf.argmax(-1)).float().mean())
    return {"states": len(states), "kl_learner_frozen": kl,
            "entropy_learner": ent_l, "entropy_frozen": ent_f,
            "argmax_agreement": agree}


def frozen_choose(model, g, s):
    x = torch.from_numpy(encode_state(g, s)).unsqueeze(0)
    m = legal_discard_mask(g.players[s].hand_counts).unsqueeze(0)
    with torch.no_grad():
        q, _ = model(x)
    return int(q.masked_fill(~m, -1e9)[0].argmax().item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", nargs=2, required=True,
                    metavar=("LEARNER", "FROZEN"),
                    help="两个检查点; 传同一个用于自检(crn 的 sd 必须为 0)")
    ap.add_argument("--perturb", type=float, default=0.0,
                    help=">0 时不用 --pairs 的第一个检查点, 而是对第二个加相对"
                         "幅度 EPS 的高斯噪声, 制造距离可控的近邻策略")
    ap.add_argument("--seeds", type=int, default=300)
    ap.add_argument("--seed0", type=int, default=207100000)
    ap.add_argument("--seat", type=int, default=0)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--first-win", action="store_true")
    ap.add_argument("--out", default="logs/diag_action_crn.json")
    args = ap.parse_args()

    torch.set_num_threads(1)
    frozen = load(args.pairs[1])
    if args.perturb > 0:
        learner = perturb(frozen, args.perturb, seed=17)
        learn_name = f"{args.pairs[1]} + 噪声 eps={args.perturb}"
    else:
        learner = load(args.pairs[0])
        learn_name = args.pairs[0]
    seeds = list(range(args.seed0, args.seed0 + args.seeds))
    same = args.perturb == 0.0 and args.pairs[0] == args.pairs[1]
    dist = policy_distance(learner, frozen, seeds[:40], args.seat,
                           temp=args.temp)
    print(f"learner={learn_name}\nfrozen ={args.pairs[1]}"
          f"{'   (同一检查点 -> crn 的 sd 应为 0)' if same else ''}")
    print(f"策略距离: KL(L||F)={dist['kl_learner_frozen']:.4f}  "
          f"熵 L/F={dist['entropy_learner']:.3f}/{dist['entropy_frozen']:.3f}  "
          f"argmax 一致率={dist['argmax_agreement']:.1%}")
    print(f"{args.seeds} seed × 2 臂, 座位 {args.seat}, temp={args.temp}, "
          f"{'首胡' if args.first_win else '血战'}\n")

    res = {}
    for mode in ("greedy", "indep", "crn"):
        rows = play_pairs(seeds, args.seat, learner, frozen, args.temp, mode,
                          bloody=not args.first_win)
        adv = np.array([r["adv"] for r in rows], dtype=float)
        sd = float(adv.std(ddof=1))
        res[mode] = {
            "mean": float(adv.mean()), "sd": sd,
            "se": sd / np.sqrt(len(adv)), "n": len(adv),
            "decisions_mean": float(np.mean([r["decisions_l"] for r in rows])),
            "divergent_actions_mean":
                float(np.mean([r["divergent_actions"] for r in rows])),
            "seeds_needed_for_0.05": (sd / 0.05) ** 2,
            "rows": rows[:10],
        }
        print(f"  {mode:7s} adv 均 {adv.mean():+.4f}  sd {sd:.4f}  "
              f"两臂动作分歧 {res[mode]['divergent_actions_mean']:.2f} 次/局  "
              f"-> 检出 0.05 分/局 需 {res[mode]['seeds_needed_for_0.05']:.0f} seed")

    base = res["greedy"]["sd"]
    print("\n相对 greedy 口径:")
    for mode in ("indep", "crn"):
        sd = res[mode]["sd"]
        ratio = (base / sd) if sd > 1e-12 else float("inf")
        print(f"  {mode:7s} sd 压缩 {ratio:.2f}x  (方差压缩 {ratio ** 2:.1f}x)")
    if same and res["crn"]["sd"] > 1e-9:
        print("  ⚠ 自检失败: 同一检查点下 crn 的 sd 不为 0, 两臂随机数没对齐")
    elif same:
        print("  自检通过: 同一检查点下 crn 两臂逐局轨迹完全一致")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump({"args": vars(args), "learner": learn_name,
                   "policy_distance": dist, "results": res}, fh,
                  ensure_ascii=False, indent=1)
    print(f"\n已写 {args.out}")


if __name__ == "__main__":
    main()
