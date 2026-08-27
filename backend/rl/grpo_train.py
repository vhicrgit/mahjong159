"""安康159 - GRPO 式组内比较训练 (想法2实现)

流程(每轮迭代):
1. 状态采样: v10 自对弈, 随机座位/巡目快照 Game + 特征
2. 当前策略 NN 对每局面出 top-m 候选弃牌(并入 v10 选择保底)
3. n 个共享世界(共同随机数): 候选在同一组世界里由 v1 推演到底,
   塑形奖励(159期望+步数惩罚) → 每候选均分
4. 组内标准化优势 A = (r - mean) / (std + eps), 策略梯度
   loss = -Σ A·logπ(a|s) + β·KL(π ‖ π_ref), π_ref 冻结于初始化

防退化设计(对应本项目 PPO/AWR 全退化的历史教训):
- 共享世界消运气噪声; 塑形奖励消翻牌噪声
- KL 锚定参考策略; 小 lr; 优势裁剪; sd 下限缩幅(低区分度组防噪声放大)
- 每 eval_every 轮评估(vs 3×v1), 只保留最优 checkpoint

用法:
  python -m backend.rl.grpo_train --init models/dqn_shaped_best.pt \
      --iters 60 --states-per-iter 192 --worlds 16 --procs 24
"""

import argparse
import copy
import multiprocessing as mp
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from ..game.engine import Game
from ..ai.bot_v1 import Bot as BotV1
from ..ai.bot_v10 import Bot as BotV10
from ..rules.ting import discard_options
from .model import build_model, N_ACTIONS
from .gen_offline import _shaped_scores
from .world_grpo import sample_world, rollout_candidate, _v10_scores


def _get_encoder(feat_version: int):
    if feat_version == 3:
        from .features_v3 import encode_state as enc, FEAT_DIM as d
    else:
        from .features_v2 import encode_state as enc, FEAT_DIM as d
    return enc, d

G = {}  # worker 全局: {model_sd, size, feat_dim} 不需要, rollout 纯 CPU 规则


def _collect_worker(args):
    """v10 自对弈一局, 返回至多 snaps_per_game 个 (快照, seat) 决策点。"""
    seed, snaps_per_game = args
    rng = random.Random(seed ^ 0xC0FFEE)
    g = Game(seed=seed, human_seat=-1)
    bots = {i: BotV10(g, i) for i in range(4)}
    take_turns = set(rng.sample(range(2, 14), snaps_per_game))
    turn_count = 0
    snaps = []
    guard = 0
    while g.phase != "game_over" and guard < 500:
        guard += 1
        if g.phase == "discard_wait":
            turn_count += 1
            seat = g.turn
            if turn_count in take_turns:
                opts = discard_options(list(g.players[seat].hand_counts))
                if len(opts) >= 3:
                    g.log = []
                    snaps.append((copy.deepcopy(g), seat))
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
    return snaps


def _rollout_worker(task):
    """task = (snap, seat, tile, worlds_seeds, step_penalty)"""
    snap, seat, tile, world_seed, n_worlds, step_penalty = task
    rng = random.Random(world_seed)
    vals = []
    for _ in range(n_worlds):
        hands, wall = sample_world(snap, rng, hero_seat=seat)
        vals.append(rollout_candidate(snap, tile, hands, wall,
                                      step_penalty, hero_seat=seat))
    return float(np.mean(vals))


class GRPOTrainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available()
                                   else "cpu")
        self.encode, self.feat_dim = _get_encoder(args.feat_version)
        self.model = build_model(args.size, feat_dim=self.feat_dim).to(self.device)
        if args.init:
            ckpt = torch.load(args.init, map_location=self.device,
                              weights_only=True)
            sd = ckpt["model"] if "model" in ckpt else ckpt
            if sd["input_proj.weight"].shape[1] == self.feat_dim:
                self.model.load_state_dict(sd)
                print(f"初始化: {args.init}")
            else:
                print(f"初始化跳过: 维度不符 "
                      f"({sd['input_proj.weight'].shape[1]} != {self.feat_dim})")
        self.ref = copy.deepcopy(self.model)
        for p in self.ref.parameters():
            p.requires_grad_(False)
        self.ref.eval()
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=args.lr,
                                     weight_decay=0.01)

    def _policy_probs(self, feats: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(feats).to(self.device)
        with torch.no_grad():
            logits, _ = self.model(x)
        # 合法掩码从特征恢复(与 dqn_offline 同法)
        oh = x[:, :112].reshape(len(x), 28, 4)
        m = (oh[:, :, 1:] > 0.5).any(dim=-1)
        logits = logits.masked_fill(~m, -1e9)
        return torch.softmax(logits.float(), dim=-1).cpu().numpy()

    def _rollout_nn(self, snaps, cand_lists):
        """NN 自对弈推演: 所有 (候选, 世界) 注入成 Game, 单批次向量化跑完。
        共享世界: 同一局面的候选复用同一组 (hands, wall)。"""
        games, meta = [], []
        for si, (snap, seat) in enumerate(snaps):
            rng = random.Random(self.args.seed0 + 991 + si)
            worlds = [sample_world(snap, rng, hero_seat=seat)
                      for _ in range(self.args.worlds)]
            for tile in cand_lists[si]:
                for hands, wall in worlds:
                    g = copy.deepcopy(snap)
                    for s2, h in hands.items():
                        g.players[s2].hand = list(h)
                    g.wall = list(wall)
                    g.action_discard(seat, tile)
                    games.append(g)
                    meta.append((si, tile, seat))
        from .vec_selfplay import VectorizedSelfPlay
        eng = VectorizedSelfPlay(self.model, len(games), self.device,
                                 games=games, model_seats=[0, 1, 2, 3],
                                 feat_version=self.args.feat_version,
                                 record=False)
        eng.run(temperature=0.0)
        r_map = {}
        buf = {}
        for (si, tile, seat), g in zip(meta, eng.games):
            r = float(_shaped_scores(g, self.args.step_penalty)[seat])
            buf.setdefault(si, {}).setdefault(tile, []).append(r)
        for si, d in buf.items():
            r_map[si] = {t: float(np.mean(v)) for t, v in d.items()}
        return r_map

    def run(self):
        args = self.args
        log_f = open(args.log, "a", buffering=1)
        # 基线: 先评估初始模型, 作为"不劣于init"的护栏
        from .grp_ppo_train import evaluate_vec
        self.model.eval()
        wr0, avg0 = evaluate_vec(self.model, self.device,
                                 args.eval_games, feat_version=args.feat_version)
        self._best = avg0
        torch.save({"model": self.model.state_dict(), "size": args.size,
                    "feat_dim": self.feat_dim,
                    "feat_version": args.feat_version},
                   args.out.replace(".pt", "_best.pt"))
        line = f"[it 0] init baseline: 胜率 {wr0:.1%}, 场均 {avg0:+.2f}"
        print(line); log_f.write(line + "\n")
        for it in range(1, args.iters + 1):
            t0 = time.time()
            # 1) 状态采样
            n_games = max(8, args.states_per_iter // args.snaps_per_game)
            with mp.Pool(args.collect_procs) as pool:
                snaps_nested = pool.map(_collect_worker, [
                    (args.seed0 + it * 100003 + gi, args.snaps_per_game)
                    for gi in range(n_games)])
            snaps = [sn for sub in snaps_nested for sn in sub]
            if not snaps:
                print("iter", it, "无有效快照, 跳过"); continue
            # 2) 特征 + 当前策略候选
            feats = np.stack([self.encode(g, seat) for g, seat in snaps])
            probs = self._policy_probs(feats)
            # 3) 组评估任务
            tasks = []
            meta = []  # (snap_idx, tile)
            for si, (g, seat) in enumerate(snaps):
                legal = [t for t in range(N_ACTIONS) if probs[si, t] > 0]
                top = sorted(legal, key=lambda t: -probs[si, t])[:args.top_m]
                v10_pick = max(_v10_scores(g, seat).items(),
                               key=lambda kv: kv[1])[0]
                if v10_pick not in top:
                    top = top[:args.top_m - 1] + [v10_pick]
                for tile in top:
                    tasks.append((g, seat, tile,
                                  args.seed0 + it * 7919 + si, args.worlds,
                                  args.step_penalty))
                    meta.append((si, tile))
            if args.rollout in ("nn", "jax"):
                cand_lists = {}
                for (si, tile) in meta:
                    cand_lists.setdefault(si, []).append(tile)
                cand_lists = [cand_lists[si] for si in range(len(snaps))]
            if args.rollout == "nn":
                self.model.eval()
                r_map = self._rollout_nn(snaps, cand_lists)
            elif args.rollout == "jax":
                from backend.jax159.rollout import (build_world_states,
                                                    rollout_jax)
                from backend.jax159.jax_net import JaxNet
                self.model.eval()
                sd = {k: v.detach().cpu().numpy()
                      for k, v in self.model.state_dict().items()}
                net = JaxNet.from_dict(sd)
                sts, meta2, _ = build_world_states(
                    snaps, cand_lists, args.worlds,
                    args.seed0 + it * 7919)
                r_map = rollout_jax(sts, meta2, net, args.step_penalty)
            else:
                with mp.Pool(args.procs) as pool:
                    rets = pool.map(_rollout_worker, tasks, chunksize=1)
                r_map = {}
                for (si, tile), r in zip(meta, rets):
                    r_map.setdefault(si, {})[tile] = r
            A_list, feat_rows, act_rows = [], [], []
            spreads = []
            for si, tile_r in r_map.items():
                vals = np.array(list(tile_r.values()))
                tiles = list(tile_r.keys())
                mu, sd = vals.mean(), vals.std()
                # sd 下限: 低区分度组(spread≈0)的回报差在 16 世界均值
                # 估计误差(~0.9)内, 归一化会放大成虚高优势, 按下限压幅
                sd_eff = max(sd, args.min_sd)
                spreads.append(vals.max() - vals.min())
                for t, v in zip(tiles, vals):
                    a = (v - mu) / (sd_eff + 1e-6)
                    a = float(np.clip(a, -args.adv_clip, args.adv_clip))
                    A_list.append(a)
                    feat_rows.append(feats[si])
                    act_rows.append(t)
            x = torch.from_numpy(np.stack(feat_rows)).to(self.device)
            acts = torch.from_numpy(np.asarray(act_rows)).to(self.device)
            adv = torch.from_numpy(np.asarray(A_list,
                                              dtype=np.float32)).to(self.device)
            self.model.train()
            for ep in range(args.inner_epochs):
                logits, _ = self.model(x)
                oh = x[:, :112].reshape(len(x), 28, 4)
                m = (oh[:, :, 1:] > 0.5).any(dim=-1)
                logits = logits.masked_fill(~m, -1e9)
                logp = F.log_softmax(logits.float(), dim=-1)
                pi = logp.exp()
                with torch.no_grad():
                    ref_logits, _ = self.ref(x)
                    ref_logits = ref_logits.masked_fill(~m, -1e9)
                    ref_logp = F.log_softmax(ref_logits.float(), dim=-1)
                    ref_p = ref_logp.exp()
                kl = (pi * (logp - ref_logp)).sum(-1).mean()
                pg = -(adv * logp.gather(1, acts.unsqueeze(1)).squeeze(1)).mean()
                loss = pg + args.kl_beta * kl
                self.opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.opt.step()
            dt = time.time() - t0
            line = (f"[it {it}] snaps={len(snaps)} rollouts={len(tasks)} "
                    f"spread={np.mean(spreads):+.3f} pg={pg.item():.4f} "
                    f"kl={kl.item():.5f} {dt:.0f}s")
            print(line); log_f.write(line + "\n")
            # 5) 评估 + 保存
            if it % args.eval_every == 0 or it == args.iters:
                from .grp_ppo_train import evaluate_vec
                self.model.eval()
                wr, avg = evaluate_vec(self.model, self.device,
                                       args.eval_games, feat_version=args.feat_version)
                tag = ""
                score_key = avg
                if not hasattr(self, "_best") or score_key > self._best:
                    self._best = score_key
                    tag = " ★"
                    torch.save({"model": self.model.state_dict(),
                                "size": args.size, "feat_dim": self.feat_dim,
                                "feat_version": args.feat_version},
                               args.out.replace(".pt", "_best.pt"))
                eline = f"[it {it}] eval: 胜率 {wr:.1%}, 场均 {avg:+.2f}{tag}"
                print(eline); log_f.write(eline + "\n")
                torch.save({"model": self.model.state_dict(),
                            "size": args.size, "feat_dim": self.feat_dim,
                            "feat_version": args.feat_version}, args.out)
        log_f.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", type=str, default="")
    ap.add_argument("--size", type=str, default="small")
    ap.add_argument("--iters", type=int, default=60)
    ap.add_argument("--states-per-iter", type=int, default=192)
    ap.add_argument("--snaps-per-game", type=int, default=4)
    ap.add_argument("--worlds", type=int, default=16)
    ap.add_argument("--top-m", type=int, default=4)
    ap.add_argument("--worlds-per-task-note", action="store_true")
    ap.add_argument("--procs", type=int, default=24)
    ap.add_argument("--collect-procs", type=int, default=12)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--kl-beta", type=float, default=0.02)
    ap.add_argument("--adv-clip", type=float, default=3.0)
    ap.add_argument("--min-sd", type=float, default=1.0,
                    help="优势归一化 sd 下限(塑形奖励单位): 低区分度组不再放大噪声")
    ap.add_argument("--inner-epochs", type=int, default=2)
    ap.add_argument("--feat-version", type=int, default=3, choices=[2,3])
    ap.add_argument("--rollout", type=str, default="v1",
                    choices=["v1", "nn", "jax"],
                    help="推演策略: v1规则 / nn当前策略 / jax环境+nn(最快)")
    ap.add_argument("--step-penalty", type=float, default=0.02)
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--eval-games", type=int, default=400)
    ap.add_argument("--seed0", type=int, default=720000)
    ap.add_argument("--out", type=str, default="models/grpo_v1.pt")
    ap.add_argument("--log", type=str, default="logs/grpo_train.log")
    args = ap.parse_args()
    GRPOTrainer(args).run()


if __name__ == "__main__":
    main()
