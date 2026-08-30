"""血战到底 + 名次奖励 + PPO 的自对弈训练。

相比 tools/rl_ac_train.py 修掉的四件事(每一条都有实测依据):

1. **奖励换成血战到底的名次奖励**。停在首胡时 81% 的局里还有别家已听牌,
   谁先胡基本是摸牌顺序抽签。改成打到 3 家胡完给 +3/+1/-1/-3, 实测
   R²(巡5状态E -> 奖励) 0.0823 -> 0.1617, 等效样本量 2.15x
   (tools/perf/diag_bloody.py, 1500 局)。

2. **critic 头重新初始化**。旧流程用"回归 E(期望巡数)"的预训练头热启动 RL,
   而 E 越小越好、得分越大越好 —— 实测 corr(v, E)=+0.956, 那个头与真正的
   价值目标反相关且偏置 +4.7 分, 前 20~30 iter 都在拆它, 期间 advantage
   符号是错的。躯干保留, value 头从零开始。

3. **PPO 多步更新**。旧流程每采集 128 局只做 1 步梯度, 200 iter 总共
   200 步; 采集 9~14s 而更新几毫秒(GPU 占用为 0 就是这么来的)。

4. **评估改 CRN 配对 + 轮坐四座位**。旧流程 96 局绝对胜率, 标准误 4.4%,
   看不见 2~3% 的真实进步, 而且"10 次取最大"有选择偏差。配对实测降方差
   1/(1-ρ): 相邻检查点 20~37x。

用法:
  python -m tools.rl_bloody_train --iters 200 --games 192 --size small
"""

import argparse
import copy
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from backend.ai.bot_native import NativeV31
from backend.rl import eval_crn
from backend.rl.features_v2 import encode_state
from backend.rl.model import build_model, legal_discard_mask
from backend.rl.vec_selfplay import VectorizedSelfPlay
import backend.rl.vec_selfplay as _vs

# regret 只在老的 reward shaping 里用, 本训练用不到, 而它是纯 Python 向听,
# 会吃掉采集的大头。置零。
_vs._rule_score = lambda *a, **k: 0.0


class BloodySelfPlay(VectorizedSelfPlay):
    """四座位全模型的血战到底自对弈; 碰/杠用 v31 规则(与评估口径一致)。"""

    def __init__(self, *a, **kw):
        kw.setdefault("bloody", True)
        super().__init__(*a, **kw)
        self.bots = [{s: NativeV31(g, s) for s in range(4)}
                     for g in self.games]


def gang_own(res, seat):
    """自己杠的收入(明杠+3, 暗杠/补杠向当时在场的每人各收1); 放杠不罚。"""
    v = 0.0
    for rec in res["gang_records"]:
        if rec["seat"] != seat:
            continue
        if rec["kind"] == "ming":
            v += 3.0
        else:
            v += float(len([x for x in rec.get("active", [0, 1, 2, 3])
                            if x != seat]))
    return v


def rewards_of(res, gang_w):
    """每座位的训练奖励 = 名次奖励 + gang_w × 自己杠分。"""
    rr = res["rank_rewards"]
    return [rr[s] + gang_w * gang_own(res, s) for s in range(4)]


class NNSeat:
    """给评估用: NN 贪心弃牌 + v31 规则碰杠。"""

    def __init__(self, game, seat, model):
        self.game, self.seat, self.model = game, seat, model
        self._v31 = NativeV31(game, seat)

    def choose_discard(self):
        x = torch.from_numpy(encode_state(self.game, self.seat)).unsqueeze(0)
        m = legal_discard_mask(
            self.game.players[self.seat].hand_counts).unsqueeze(0)
        with torch.no_grad():
            q = self.model.q(x, m)[0]
        return int(q.argmax().item())

    def decide_peng(self, tile):
        return self._v31.decide_peng(tile)

    def decide_gang(self, tile, kind):
        return self._v31.decide_gang(tile, kind)


def make_nn_factory(model):
    return lambda g, s: NNSeat(g, s, model)


def cpu_copy(model, size):
    m = build_model(size)
    m.load_state_dict({k: v.detach().cpu()
                       for k, v in model.state_dict().items()})
    m.eval()
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--games", type=int, default=192)
    ap.add_argument("--size", default="small")
    ap.add_argument("--pretrain", default="models/hv_value_pretrained.pt",
                    help="躯干+q头的热启动权重; none 表示从零开始")
    ap.add_argument("--keep-value-head", action="store_true",
                    help="保留预训练的 value 头(默认重新初始化, 见文件头 #2)")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--ent", type=float, default=0.01)
    ap.add_argument("--kl-ref", type=float, default=0.0,
                   help="锚定到参考策略的 KL 惩罚权重。熵奖励把策略推向均匀, "
                        "而参考策略是已知不错的 BC 起点, 用它当锚更合理")
    ap.add_argument("--ref", default="",
                   help="参考策略检查点; 留空则用 --pretrain 那个")
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--epochs", type=int, default=4, help="每批 PPO 轮数")
    ap.add_argument("--minibatch", type=int, default=1024)
    ap.add_argument("--vcoef", type=float, default=0.5)
    ap.add_argument("--kl-target", type=float, default=0.02,
                    help="批内平均 KL 超过它就停掉剩余 epoch(防策略崩)")
    ap.add_argument("--gang-w", type=float, default=0.25)
    ap.add_argument("--shuffle-adv", action="store_true",
                    help="对照实验: 批内打乱 advantage。若退化速度与正常一致, "
                         "说明 advantage 里没有可用信号, 问题是信噪比不是超参")
    ap.add_argument("--rscale", type=float, default=3.0, help="奖励缩放除数")
    ap.add_argument("--no-bucket-center", action="store_true",
                    help="关掉按弃牌序号分桶去均值。实测 corr(弃牌序号,奖励)"
                         "=-0.31(胡家早下场只留少量正奖励决策, 没胡的人留下"
                         "大量负奖励决策), 不去掉会无差别压制所有后期动作")
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--eval-seeds", type=int, default=200)
    ap.add_argument("--seed0", type=int, default=30000000)
    ap.add_argument("--out", default="models/bloody_latest.pt")
    ap.add_argument("--hist", default="logs/bloody_hist.json")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    size = args.size
    ckpt = None
    if args.pretrain != "none" and os.path.exists(args.pretrain):
        ckpt = torch.load(args.pretrain, map_location="cpu",
                          weights_only=True)
        if ckpt.get("size"):
            size = ckpt["size"]
    model = build_model(size).to(device)
    if ckpt is not None:
        fresh_v = {k: v for k, v in build_model(size).state_dict().items()
                   if k.startswith("value_head")}
        sd = dict(ckpt["model"])
        if not args.keep_value_head:
            # 预训练 value 头回归 E(越小越好), 与得分/名次反相关, 是负资产
            sd.update({k: v for k, v in fresh_v.items() if k in sd})
        model.load_state_dict(sd, strict=False)
        print(f"热启动 {args.pretrain} (size={size}, "
              f"value 头{'保留' if args.keep_value_head else '重新初始化'})")
    else:
        print(f"从零开始 (size={size})")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    ref_model = None
    if args.kl_ref > 0:
        rp = args.ref or args.pretrain
        rck = torch.load(rp, map_location="cpu", weights_only=True)
        ref_model = build_model(rck.get("size", size)).to(device)
        ref_model.load_state_dict(rck["model"], strict=False)
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad_(False)
        print(f"KL 锚定参考策略 {rp} (权重 {args.kl_ref})")

    hist = []
    best = None
    best_key = -1e9
    eval_seeds = list(range(args.seed0 + 900000,
                            args.seed0 + 900000 + args.eval_seeds))

    for it in range(1, args.iters + 1):
        t0 = time.time()
        vec = BloodySelfPlay(model, args.games, device,
                             seed0=args.seed0 + it * args.games)
        results = vec.run(temperature=args.temp)
        t_col = time.time() - t0

        feats, masks, acts, oldlp, rets, didx = [], [], [], [], [], []
        rr_all, rank_all = [], []
        for res in results:
            rw = rewards_of(res, args.gang_w)
            rr_all += res["rank_rewards"]
            rank_all += res["ranks"]
            cnt = {}
            for (seat, feat, tile, lp, val, regret, mask) in res["records"]:
                cnt[seat] = cnt.get(seat, 0) + 1
                feats.append(feat)
                masks.append(mask)
                acts.append(tile)
                oldlp.append(lp)
                rets.append(rw[seat] / args.rscale)
                didx.append(cnt[seat])
        n = len(rets)
        if n == 0:
            print(f"iter {it}: 无样本, 跳过")
            continue
        X = torch.from_numpy(np.stack(feats)).to(device)
        M = torch.from_numpy(np.stack(masks)).to(device)
        A = torch.tensor(acts, dtype=torch.long, device=device)
        LP0 = torch.tensor(oldlp, dtype=torch.float32, device=device)
        R = torch.tensor(rets, dtype=torch.float32, device=device)

        # advantage 用采集时的 critic(冻结), 每个 epoch 内不再变
        model.eval()
        with torch.no_grad():
            _, v0 = model(X)
        adv0 = R - v0
        if not args.no_bucket_center:
            # 按弃牌序号分桶去均值: 消掉"晚巡=坏"这个与动作质量无关的混淆
            D = torch.tensor(didx, dtype=torch.long, device=device)
            D = D.clamp(max=20)
            for b in D.unique():
                m_b = D == b
                if m_b.sum() >= 8:
                    adv0 = adv0.masked_scatter(
                        m_b, adv0[m_b] - adv0[m_b].mean())
        dcorr = float(np.corrcoef(np.asarray(didx, float),
                                  adv0.detach().cpu().numpy())[0, 1])
        adv0 = (adv0 - adv0.mean()) / (adv0.std() + 1e-6)
        if args.shuffle_adv:
            adv0 = adv0[torch.randperm(n, device=device)]
        LPREF = None
        if ref_model is not None:
            with torch.no_grad():
                qr, _ = ref_model(X)
                LPREF = F.log_softmax(qr.masked_fill(~M, -1e9), dim=-1)

        model.train()
        stats = {"pi": 0.0, "v": 0.0, "ent": 0.0, "kl": 0.0, "clip": 0.0,
                 "klref": 0.0}
        nstep = 0
        n_ep = 0
        stop = False
        for _ in range(args.epochs):
            if stop:
                break
            n_ep += 1
            perm = torch.randperm(n, device=device)
            for k in range(0, n, args.minibatch):
                idx = perm[k:k + args.minibatch]
                q, v = model(X[idx])
                logits = q.masked_fill(~M[idx], -1e9)
                logp_all = F.log_softmax(logits, dim=-1)
                probs = logp_all.exp()
                ent = -(probs * logp_all).sum(-1).mean()
                lp = logp_all.gather(1, A[idx].unsqueeze(1)).squeeze(1)
                ratio = (lp - LP0[idx]).exp()
                a = adv0[idx]
                l1 = ratio * a
                l2 = ratio.clamp(1 - args.clip, 1 + args.clip) * a
                loss_pi = -torch.min(l1, l2).mean()
                loss_v = F.mse_loss(v, R[idx])
                loss = loss_pi + args.vcoef * loss_v - args.ent * ent
                klref = torch.zeros((), device=device)
                if LPREF is not None:
                    klref = (probs * (logp_all - LPREF[idx])).sum(-1).mean()
                    loss = loss + args.kl_ref * klref
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                kl = (LP0[idx] - lp).mean().item()
                stats["pi"] += loss_pi.item()
                stats["v"] += loss_v.item()
                stats["ent"] += ent.item()
                stats["kl"] += kl
                stats["clip"] += ((ratio - 1).abs() > args.clip
                                  ).float().mean().item()
                stats["klref"] += float(klref)
                nstep += 1
                if stats["kl"] / nstep > args.kl_target:
                    stop = True        # 策略跑太远, 本批不再更新
                    break
        for k in stats:
            stats[k] /= max(1, nstep)

        # critic 质量: 用更新后的 v 对同批样本的 R² (旧流程这里是 -0.01)
        model.eval()
        with torch.no_grad():
            _, v1 = model(X)
        resid = (R - v1)
        r2 = 1.0 - (resid.var() / R.var()).item()

        msg = (f"iter {it:4d} 样本{n:6d} 步{nstep:3d}/{n_ep}ep "
               f"pi {stats['pi']:+.4f} v {stats['v']:.4f} "
               f"ent {stats['ent']:.2f} kl {stats['kl']:+.4f} "
               f"clip {stats['clip']:.2f} klref {stats['klref']:.3f} "
               f"vR² {r2:+.3f} dcorr {dcorr:+.3f} "
               f"名次奖励均{np.mean(rr_all):+.2f} 采集{t_col:.1f}s")

        if it % args.eval_every == 0 or it == args.iters:
            m_cpu = cpu_copy(model, size)
            t1 = time.time()
            ev = eval_crn.paired_vs_v31(make_nn_factory(m_cpu), eval_seeds)
            key = ev["rank"]["mean"]
            msg += (f"\n   [评估 {args.eval_seeds}seed×4座位 vs v31n] "
                    f"名次差 {ev['rank']['mean']:+.3f}±{ev['rank']['se']:.3f} "
                    f"(t={ev['rank']['t']:+.1f})  "
                    f"得分差 {ev['score']['mean']:+.3f}±"
                    f"{ev['score']['se']:.3f}  用时{time.time() - t1:.0f}s")
            hist.append({"iter": it, "rank_diff": ev["rank"]["mean"],
                         "rank_se": ev["rank"]["se"],
                         "score_diff": ev["score"]["mean"],
                         "vr2": r2, "ent": stats["ent"]})
            os.makedirs(os.path.dirname(args.out), exist_ok=True)
            torch.save({"size": size, "model": m_cpu.state_dict(),
                        "iter": it, "rank_diff": key}, args.out)
            if key > best_key:
                best_key = key
                best = copy.deepcopy(m_cpu.state_dict())
                torch.save({"size": size, "model": best, "iter": it,
                            "rank_diff": key},
                           args.out.replace(".pt", "_best.pt"))
            os.makedirs(os.path.dirname(args.hist), exist_ok=True)
            with open(args.hist, "w") as f:
                json.dump(hist, f, ensure_ascii=False, indent=1)
        print(msg, flush=True)

    print(f"完成。最佳 名次差 {best_key:+.3f} 分/局 vs v31n, "
          f"模型 {args.out.replace('.pt', '_best.pt')}")


if __name__ == "__main__":
    main()
