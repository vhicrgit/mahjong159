"""Actor-Critic 自对弈训练(A2C), 价值网络用牌型分析器 E 预训练热启动。

架构(backend/rl/model.py MahjongNet): 残差躯干 + q_head(28 出牌 logits,
当策略头用) + value_head(期望调整得分)。预训练(tools/rl_value_pretrain.py)
把躯干+value 头学到 E(期望巡数)上; 本阶段 value 头转向调整得分, q 头从零学起。

奖励口径(按需求): 终局得分为基础, 但**移除"放杠赔 3 分"** —— 别人杠我方弃牌
涉及对对手手牌的猜测, 噪声大, 不罚; 自己杠的收入(明杠+3/暗补杠各+1×3)保留。
实现: 调整后得分 = score_delta + 3 × (该座位作为明杠放杠者的次数)。

引擎: VectorizedSelfPlay 四座位全模型(自我对弈), 碰/杠走 v1 规则(与训练环境一致)。

每 --eval-every 轮做配对评估: 座0 NN(贪心) vs 3×v31n, 同种子配对报胜率。

用法:
  python -m tools.rl_ac_train --iters 300 --games 128 --size small
"""

import argparse
import math
import multiprocessing as mp
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from backend.ai.bot_native import NativeV31
from backend.game.engine import Game
from backend.rl.features_v2 import encode_state  # noqa: F401  (vec 内部用)
from backend.rl.model import build_model
from backend.rl.vec_selfplay import VectorizedSelfPlay


# vec_selfplay 逐决策用纯 Python shanten 算"规则分差"(regret), 本训练用不到,
# 而它占了采集的大头(~14 次 Python 向听/决策)。monkeypatch 成常数, 省掉。
import backend.rl.vec_selfplay as _vs
vs_rule_score_orig = _vs._rule_score
_vs._rule_score = lambda *a, **k: 0.0


class ACVectorizedSelfPlay(VectorizedSelfPlay):
    """_collect_results 增加 gang_records 以便训练侧做奖励调整。
    碰/杠反应用 v31n(与数据来源、评估口径一致; 默认是 v1)。"""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.bots = [{s: NativeV31(g, s) for s in range(4)}
                     for g in self.games]

    def _collect_results(self):
        results = super()._collect_results()
        for i, g in enumerate(self.games):
            results[i]["gang_records"] = g.gang_records
        return results


def adjusted_scores(res):
    """按需求口径: 放杠(明杠的弃牌方)不扣分, 其余不变。"""
    adj = list(res["scores"])
    for rec in res["gang_records"]:
        if rec["kind"] == "ming":
            adj[rec["from"]] += 3
    return adj


def eval_vs_v31(seed0, n_games, model, procs=1):
    """座0 NN(贪心) vs 3×v31n, 同种子。串行(in-process)评估:
    进程池 fork 会在 torch 线程锁上卡死(实测), 而 16-96 局的串行
    评估只要几秒, 没必要并行。"""
    from backend.ai.bot_v1 import Bot as V1
    from backend.rl.features_v2 import encode_state as enc
    from backend.rl.model import legal_discard_mask

    class G:
        def __init__(self, game, seat):
            self.game, self.seat = game, seat
            self._v1 = V1(game, seat)

        def choose_discard(self):
            feat = enc(self.game, self.seat)
            x = torch.from_numpy(feat).unsqueeze(0)
            mask = legal_discard_mask(
                self.game.players[self.seat].hand_counts).unsqueeze(0)
            with torch.no_grad():
                q = model.q(x, mask)[0]
            return int(q.argmax().item())

        def decide_peng(self, tile):
            from backend.native import native
            return native.decide_peng(31, self.game.players[self.seat].hand_counts, tile)

        def decide_gang(self, tile, kind):
            from backend.native import native
            return native.decide_gang(31, self.game.players[self.seat].hand_counts, tile, kind)

    wins, scores = [], []
    for i in range(n_games):
        g = Game(seed=seed0 + i, human_seat=-1)
        bots = {s_: NativeV31(g, s_) for s_ in range(1, 4)}
        bots[0] = G(g, 0)
        guard = 0
        while g.phase != "game_over" and guard < 500:
            guard += 1
            if g.phase == "discard_wait":
                g.action_discard(g.turn, bots[g.turn].choose_discard())
            else:
                s_ = list(g.pending_actions.keys())[0]
                pend = g.pending_actions[s_]
                b = bots[s_]
                if pend.get("gang") and b.decide_gang(g.last_discard, "ming"):
                    g.action_gang(s_)
                elif pend.get("peng") and b.decide_peng(g.last_discard):
                    g.action_peng(s_)
                else:
                    g.action_pass(s_)
        wins.append(1 if g.winner == 0 else 0)
        scores.append(g.players[0].score_delta)
    return float(np.mean(wins)), float(np.mean(scores))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--games", type=int, default=128)
    ap.add_argument("--size", type=str, default="small")
    ap.add_argument("--pretrain", type=str, default="models/hv_value_pretrained.pt")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--ent", type=float, default=0.01)
    ap.add_argument("--eval-every", type=int, default=20)
    ap.add_argument("--eval-games", type=int, default=96)
    ap.add_argument("--seed0", type=int, default=2000000)
    ap.add_argument("--out", type=str, default="models/acnn_latest.pt")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if os.path.exists(args.pretrain):
        # 本地产出物, 且内容仅含张量与基本类型, 用 weights_only 加载
        ckpt = torch.load(args.pretrain, map_location="cpu",
                          weights_only=True)
        if ckpt.get("size") and ckpt["size"] != args.size:
            print(f"预训练模型规格 {ckpt['size']} 覆盖 --size {args.size}")
            args.size = ckpt["size"]
    model = build_model(args.size).to(device)
    if os.path.exists(args.pretrain):
        missing, unexpected = model.load_state_dict(ckpt["model"],
                                                    strict=False)
        print(f"热启动预训练权重 (val MAE {ckpt.get('val_mae_turns', '?'):.3f} 巡)")
    else:
        print("未找到预训练权重, 从零开始(不推荐)")
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_wr = -1.0
    for it in range(1, args.iters + 1):
        t0 = time.time()
        vec = ACVectorizedSelfPlay(model, args.games, device,
                                   seed0=args.seed0 + it * args.games)
        results = vec.run(temperature=args.temp)
        collect_s = time.time() - t0

        # 汇总批次
        feats, masks, tiles, rets = [], [], [], []
        for res in results:
            adj = adjusted_scores(res)
            for (seat, feat, tile, logp, val, regret, mask) in res["records"]:
                feats.append(feat)
                masks.append(mask)
                tiles.append(tile)
                rets.append(float(adj[seat]))
        X = torch.from_numpy(np.stack(feats)).to(device)
        M = torch.from_numpy(np.stack(masks)).to(device)
        A = torch.tensor(tiles, dtype=torch.long, device=device)
        G = torch.tensor(rets, dtype=torch.float32, device=device)
        # 得分量级大(~±20), 价值头回归更稳: 缩小 10 倍
        Gs = G / 10.0

        model.train()
        q, v = model(X)
        logits = q.masked_fill(~M, -1e9)
        logp_all = F.log_softmax(logits, dim=-1)
        probs = logp_all.exp()
        ent = -(probs * logp_all).sum(-1).mean()
        logp_a = logp_all.gather(1, A.unsqueeze(1)).squeeze(1)
        adv = (Gs - v.detach())
        adv = (adv - adv.mean()) / (adv.std() + 1e-6)
        loss_pi = -(logp_a * adv).mean()
        loss_v = F.mse_loss(v, Gs)
        loss = loss_pi + 0.5 * loss_v - args.ent * ent
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        n_rec = len(rets)
        gang_free = np.mean([np.mean(adjusted_scores(r)) for r in results])
        # 价值头质量: 预测 vs 实际回报的 MAE(换算回分)
        v_mae = (v - Gs).abs().mean().item() * 10.0
        msg = (f"iter {it:4d}  样本{n_rec}  loss {loss.item():+.3f} "
               f"(pi {loss_pi.item():+.3f} v {loss_v.item():.3f} "
               f"ent {ent.item():.2f} vMAE {v_mae:.2f}分)  采集{collect_s:.0f}s")
        if it % args.eval_every == 0 or it == args.iters:
            model.eval()
            sd = {k: t.cpu() for k, t in model.state_dict().items()}
            eval_model = build_model(args.size)
            eval_model.load_state_dict(sd)
            eval_model.eval()
            wr, sc = eval_vs_v31(args.seed0 + 900000 + it, args.eval_games,
                                 eval_model)
            msg += f"  | 评估 vs 3×v31n: 胜率 {wr:.1%} 场均 {sc:+.2f}"
            model.train()
            os.makedirs(os.path.dirname(args.out), exist_ok=True)
            torch.save({"size": args.size, "model": sd, "iter": it,
                        "eval_wr": wr}, args.out)
            if wr > best_wr:
                best_wr = wr
                torch.save({"size": args.size, "model": sd, "iter": it,
                            "eval_wr": wr},
                               args.out.replace(".pt", "_best.pt"))
        print(msg, flush=True)

    print(f"完成。最佳评估胜率 {best_wr:.1%}, 模型在 {args.out}")


if __name__ == "__main__":
    main()
