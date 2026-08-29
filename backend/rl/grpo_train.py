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
from ..ai.bot_v31 import Bot as BotV31
from ..ai.bot_cheat import Bot as BotCheat
from ..ai.bot_oracle import Bot as BotOracle
from ..ai.bot_native import NativeV1, NativeV10, NativeV31
from ..ai.bot_cheat_native import NativeCheatFull
from ..rules.ting import discard_options
from ..rules.tiles import tile_name
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
    """v10 自对弈一局, 返回至多 snaps_per_game 个 (快照, seat) 决策点。
    用 NativeV10(C 实现, 与 BotV10 逐位一致, 见 test_parity_collect.py)。"""
    seed, snaps_per_game = args
    rng = random.Random(seed ^ 0xC0FFEE)
    g = Game(seed=seed, human_seat=-1)
    bots = {i: NativeV10(g, i) for i in range(4)}
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


# Bot 注册表: key -> {name, cls, desc, kwargs(传给 Bot.__init__)}
_BOT_REGISTRY = {
    "v1":         {"name": "菜鸟", "cls": BotV1,    "kwargs": {},
                   "desc": "纯向听+进张, 基础规则Bot"},
    "v10":        {"name": "中鸟", "cls": BotV10,   "kwargs": {},
                   "desc": "v10 规则, 含触发性向听"},
    "v31":        {"name": "老鸟", "cls": BotV31,   "kwargs": {},
                   "desc": "v10 + 副露感知碰牌(碰后还要打一张才比向听)。"
                           "v31n 是它的 C 实现, 逐位一致且快 ~600x"},
    "cheat":      {"name": "挂哥", "cls": BotCheat, "kwargs": {"wall_lookahead": 32,
                                                                "see_opponents": False},
                   "desc": "可见牌墙(32 张), 不看对手手牌"},
    "cheat_full": {"name": "神挂", "cls": BotCheat, "kwargs": {"wall_lookahead": -1,
                                                                "see_opponents": True,
                                                                "rollout": True},
                   "desc": "全信息: 墙+对手手牌+rollout 推演"},
    "oracle":     {"name": "先知", "cls": BotOracle, "kwargs": {},
                   "desc": "Beam Search, 已知自己摸牌序列"},
    "cheat_full_jax": {"name": "神挂JAX", "cls": BotCheat, "kwargs": {},
                   "desc": "GPU 加速版: 用 rollout_jax(NN)替代 beam 搜索, 后续接入墙特征"},
    # 原生(C)版: 与对应的纯 Python Bot 逐位同口径, 由
    # backend/ai/test_parity_native.py 把关(逐决策 + 整局日志/得分全等)。
    # 单核 ~500x, 128 世界推演靠它才跑得起来。
    "v1n":        {"name": "菜鸟C", "cls": NativeV1,  "kwargs": {},
                   "desc": "v1 的 C 实现, 与 v1 逐位一致"},
    "v10n":       {"name": "中鸟C", "cls": NativeV10, "kwargs": {},
                   "desc": "v10 的 C 实现, 与 v10 逐位一致"},
    "v31n":       {"name": "老鸟C", "cls": NativeV31, "kwargs": {},
                   "desc": "bot_v31 的 C 实现(含副露感知碰牌), 与 bot_v31 逐位一致"},
    "cheat_fulln": {"name": "神挂C", "cls": NativeCheatFull, "kwargs": {},
                   "desc": "cheat_full 的 C 实现, 与 cheat_full 逐位一致(~250x)。"
                           "仍比 v31n 贵 ~170x/局, 成本由 CHEAT_DEPTH/"
                           "CHEAT_ROOT_WIDTH 主导"},
}


def _pool_map_robust(worker, tasks, procs, spawn, timeout=3600, tag=""):
    """pool.map + 超时自愈。任务由种子驱动、完全确定, 超时杀池重试安全。
    排障背景: spawn 池曾出现 worker 全员 idle、主进程永远等结果的死锁,
    一次卡死让 run 空转 9 小时。"""
    for attempt in (1, 2):
        ctx = mp.get_context("spawn") if spawn else mp
        pool = ctx.Pool(procs)
        try:
            result = pool.map_async(worker, tasks, chunksize=1).get(
                timeout=timeout)
        except mp.TimeoutError:
            print(f"[警告] {tag} 推演池 {timeout}s 无结果, 终止重试"
                  f"(第{attempt}次)", flush=True)
            pool.terminate()
            pool.join()
            continue
        pool.close()
        pool.join()
        return result
    raise RuntimeError(f"{tag} 推演池两次超时, 放弃本轮迭代")


def _rollout_worker_rules(task):
    """通用规则推演(4 家同策略)。task = (snap, seat, tile, world_seed, n_worlds, step_penalty, bot_type)"""
    snap, seat, tile, world_seed, n_worlds, step_penalty, bot_type = task
    rng = random.Random(world_seed)
    cfg = _BOT_REGISTRY[bot_type]
    bot_cls = cfg["cls"]
    bot_kwargs = cfg["kwargs"]
    vals = []
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
            elif g.phase == "react_wait":
                s = list(g.pending_actions.keys())[0]
                b = bots[s]
                if g.pending_actions[s].get("gang") and b.decide_gang(g.last_discard, "ming"):
                    g.action_gang(s)
                elif g.pending_actions[s].get("peng") and b.decide_peng(g.last_discard):
                    g.action_peng(s)
                else:
                    g.action_pass(s)
        vals.append(float(_shaped_scores(g, step_penalty)[seat]))
    return float(np.mean(vals))


def _rollout_worker_mixed(task):
    """混合推演: hero 用 hero_type bot, 对手用 opp_type bot。
    task = (snap, seat, tile, world_seed, n_worlds, step_penalty, hero_type, opp_type)"""
    snap, seat, tile, world_seed, n_worlds, step_penalty, hero_type, opp_type = task
    rng = random.Random(world_seed)
    hero_cfg = _BOT_REGISTRY[hero_type]
    opp_cfg = _BOT_REGISTRY[opp_type]
    vals = []
    for _ in range(n_worlds):
        hands, wall = sample_world(snap, rng, hero_seat=seat)
        g = copy.deepcopy(snap)
        for s2, h in hands.items():
            g.players[s2].hand = list(h)
        g.wall = list(wall)
        hero_bot = hero_cfg["cls"](g, seat, **hero_cfg["kwargs"])
        opp_bots = {}
        for i in range(4):
            if i != seat:
                opp_bots[i] = opp_cfg["cls"](g, i, **opp_cfg["kwargs"])
        g.action_discard(seat, tile)
        guard = 0
        while g.phase != "game_over" and guard < 500:
            guard += 1
            if g.phase == "discard_wait":
                if g.turn == seat:
                    g.action_discard(seat, hero_bot.choose_discard())
                else:
                    g.action_discard(g.turn, opp_bots[g.turn].choose_discard())
            elif g.phase == "react_wait":
                s = list(g.pending_actions.keys())[0]
                b = opp_bots.get(s, hero_bot)
                if g.pending_actions[s].get("gang") and b.decide_gang(g.last_discard, "ming"):
                    g.action_gang(s)
                elif g.pending_actions[s].get("peng") and b.decide_peng(g.last_discard):
                    g.action_peng(s)
                else:
                    g.action_pass(s)
        vals.append(float(_shaped_scores(g, step_penalty)[seat]))
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
        # 预加载 init 模型(用于 opp_model=init 的对手推演)
        self._init_net = None
        if args.init and args.opp_model == "init":
            ckpt = torch.load(args.init, map_location=self.device,
                              weights_only=True)
            sd2 = ckpt["model"] if "model" in ckpt else ckpt
            if sd2["input_proj.weight"].shape[1] == self.feat_dim:
                from backend.jax159.jax_net import JaxNet
                self._init_net = JaxNet.from_dict(
                    {k: v.detach().cpu().numpy() for k, v in sd2.items()})
                print(f"init 对手模型已加载: {args.init}")

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
            snaps_nested = _pool_map_robust(_collect_worker, [
                (args.seed0 + it * 100003 + gi, args.snaps_per_game)
                for gi in range(n_games)], args.collect_procs,
                spawn=False, tag=f"it{it} 采样")
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
                # argmax(_v10_scores) 与 NativeV10.choose_discard 同一张
                # (两者都按 tile 升序取严格最大, 见 test_parity_collect.py),
                # 但原来的纯 Python 版在主进程里每个快照要跑约 1 秒。
                v10_pick = NativeV10(g, seat).choose_discard()
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
            mode = getattr(args, "rollout_mode", None)
            opp_model = getattr(args, "opp_model", "self")
            # mode 显式指定优先; 否则回退 --rollout
            if mode is not None:
                rollout_mode = mode
            else:
                rl = getattr(args, "rollout", "nn")
                rollout_mode = rl if rl in ("v1", "nn", "jax") else "nn"

            if rollout_mode in ("nn", "jax", "cheat_full_jax", "v10_jax"):
                # JAX + NN 推演 (cheat_full_jax 暂为 NN 推演的别名, 后续加入墙特征)
                cand_lists = {}
                for (si, tile) in meta:
                    cand_lists.setdefault(si, []).append(tile)
                cand_lists = [cand_lists[si] for si in range(len(snaps))]
                from backend.jax159.fast_inject import build_world_states
                from backend.jax159.jax_net import JaxNet
                self.model.eval()
                sd = {k: v.detach().cpu().numpy()
                      for k, v in self.model.state_dict().items()}
                net = JaxNet.from_dict(sd)
                t0 = time.time()
                sts, meta2, _ = build_world_states(
                    snaps, cand_lists, args.worlds,
                    args.seed0 + it * 7919)
                t1 = time.time()
                if opp_model in ("v1", "v10", "cheat"):
                    # 规则对手: 退化为纯规则推演(混合模式待后续实现)
                    bot_type = opp_model
                    tasks_r = [(snaps[si][0], snaps[si][1], tile,
                                args.seed0 + it * 7919 + si, args.worlds,
                                args.step_penalty, bot_type)
                               for (si, tile) in meta]
                    with mp.Pool(args.procs) as pool:
                        rets = pool.map(_rollout_worker_rules, tasks_r, chunksize=1)
                    r_map = {}
                    for (si, tile), r in zip(meta, rets):
                        r_map.setdefault(si, {})[tile] = r
                else:
                    net_opp = self._init_net if opp_model == "init" else None
                    if args.rollout_gpus > 1:
                        from backend.jax159.parallel_rollout import (
                            start_workers, rollout_parallel)
                        if not hasattr(self, "_rw"):
                            self._rw = start_workers(args.rollout_gpus, "")
                        r_map = rollout_parallel(sts, meta2, net.params,
                                                 args.step_penalty,
                                                 self._rw[0], self._rw[1],
                                                 args.rollout_gpus)
                    else:
                        if rollout_mode == "v10_jax":
                            from backend.jax159.v10_rollout import rollout_v10_jax
                            r_map = rollout_v10_jax(sts, meta2, args.step_penalty)
                        else:
                            from backend.jax159.rollout import rollout_jax
                            r_map = rollout_jax(sts, meta2, net, args.step_penalty,
                                                net_opp=net_opp)
                t2 = time.time()
                log_f.write(f"  [timing] inject={t1-t0:.1f}s rollout={t2-t1:.1f}s\n")
                log_f.flush()
            elif rollout_mode in tuple(_BOT_REGISTRY.keys()):
                # 规则推演(4 家同策略) 或 混合推演(hero≠opp)
                opp_is_rule = opp_model in tuple(_BOT_REGISTRY.keys())
                if opp_is_rule and opp_model != rollout_mode:
                    # 混合模式: hero=rollout_mode, 对手=opp_model
                    bot_type = rollout_mode
                    tasks_r = [(snaps[si][0], snaps[si][1], tile,
                                args.seed0 + it * 7919 + si, args.worlds,
                                args.step_penalty, rollout_mode, opp_model)
                               for (si, tile) in meta]
                    rets = _pool_map_robust(_rollout_worker_mixed, tasks_r,
                                            args.procs, spawn=True,
                                            tag=f"it{it} 混合推演")
                else:
                    # 纯规则推演(4 家同策略)
                    bot_type = rollout_mode
                    tasks_r = [(snaps[si][0], snaps[si][1], tile,
                                args.seed0 + it * 7919 + si, args.worlds,
                                args.step_penalty, bot_type)
                               for (si, tile) in meta]
                    rets = _pool_map_robust(_rollout_worker_rules, tasks_r,
                                            args.procs, spawn=False,
                                            tag=f"it{it} 规则推演")
                r_map = {}
                for (si, tile), r in zip(meta, rets):
                    r_map.setdefault(si, {})[tile] = r
            else:
                # 旧 v1 规则推演(向后兼容)
                rets = _pool_map_robust(_rollout_worker, tasks,
                                        args.procs, spawn=True,
                                        tag=f"it{it} 旧v1推演")
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
                if args.dump_decisions and si < args.dump_decisions:
                    g, seat = snaps[si]
                    hand = " ".join(tile_name(t) for t in sorted(g.players[seat].hand))
                    melds = g.players[seat].melds
                    dump = (f"[it {it} 决策点#{si} 座{seat} 墙剩{len(g.wall)} "
                            f"手牌[{hand}] 副露{melds if melds else '无'}]")
                    for t in sorted(tiles, key=lambda x: -r_map[si][x]):
                        v = r_map[si][t]
                        a = (v - mu) / (sd_eff + 1e-6)
                        a = float(np.clip(a, -args.adv_clip, args.adv_clip))
                        dump += (f"\n    {tile_name(t)}: 回报{v:+.3f} "
                                 f"sd_eff={sd_eff:.3f} 优势{a:+.3f}")
                    log_f.write(dump + "\n")
                    log_f.flush()
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
        # 清理多卡 worker(否则常驻进程泄漏显存, 后续训练 OOM)
        if hasattr(self, "_rw"):
            from backend.jax159.parallel_rollout import stop_workers
            stop_workers(self._rw[0], self._rw[2])
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
    ap.add_argument("--dump-decisions", type=int, default=0,
                    help="每迭代打印前 N 个决策点的候选/回报/优势明细")
    ap.add_argument("--inner-epochs", type=int, default=2)
    ap.add_argument("--feat-version", type=int, default=3, choices=[2,3])
    ap.add_argument("--rollout", type=str, default="nn",
                    choices=["v1", "nn", "jax"],
                    help="弃用: 请用 --rollout-mode")
    ap.add_argument("--rollout-mode", type=str, default=None,
                    choices=["nn", "v1", "v10", "v31", "v1n", "v10n", "v31n",
                             "v10_jax", "cheat", "cheat_full", "cheat_fulln",
                             "cheat_full_jax", "oracle"],
                    help="推演模式: nn=JAX+NN, 其余=对应规则bot推演"
                         "(带 n 后缀=C 原生实现, 与同名 Python bot 逐位一致但快 ~500x)")
    ap.add_argument("--opp-model", type=str, default="self",
                    choices=["self", "init", "v1", "v10", "v31",
                             "v1n", "v10n", "v31n", "cheat", "cheat_full",
                             "cheat_fulln", "cheat_full_jax", "oracle"],
                    help="对手模型(nn模式): self=当前策略, init=固定初始模型, 规则bot名=规则对手(需--opp-model)")
    ap.add_argument("--rollout-gpus", type=int, default=1,
                    help="jax rollout 使用的 GPU 数(多卡并行推演)")
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
