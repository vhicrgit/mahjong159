"""同墙反事实采集 —— 把逐决策的 advantage 从"整局回报"换成"换一张牌的差"。

为什么必须这么做(全部实测):
  R - v(s) 这种 advantage 里, 一个座位一局内的 ~10 次决策共享同一个回报,
  逐决策信噪比只有 0.063; critic 修好后 vR² 也只有 0.27, 因为 80%+ 的回报
  方差来自未来摸牌顺序, 状态里根本看不见。结果是 PPO 在一个不错的 BC 初始化
  上做随机游走, 单调退化(-0.408 -> -0.861 分/局 vs v31n), 且退化幅度随步长走。

反事实的做法: 在采样到的决策点复制局面, 换成另一张牌重放到终局, **牌墙完全
相同**。两条支线只差这一张牌, 实测 ρ≈0.97 -> 方差降 1/(1-ρ) ≈ 37x。
主轨迹本身就是"选中那张"的支线, 所以每个样本只需要一次额外重放。

  adv(a vs a') = R_主 - R_替

这是 a 相对 a' 的优势的无偏估计(共同牌墙只降方差不引偏差)。
"""

import copy

import numpy as np
import torch
import torch.nn.functional as F

from ..ai.bot_native import NativeV31
from ..game.engine import Game
from .features_v2 import encode_state
from .model import legal_discard_mask


def _react(g):
    """把 react_wait 阶段走完(碰/杠用 v31 规则, 与评估口径一致)。"""
    while g.phase == "react_wait":
        s = list(g.pending_actions.keys())[0]
        b = NativeV31(g, s)
        hc = g.players[s].hand_counts
        if g.pending_actions[s].get("gang") and \
                b.decide_gang(g.last_discard, "ming"):
            g.action_gang(s)
        elif g.pending_actions[s].get("peng") and b.decide_peng(g.last_discard):
            g.action_peng(s)
        else:
            g.action_pass(s)


def _policy_batch(model, device, games, seats, temp):
    """一次前向决定一批局面的弃牌。返回 (tiles, logps, feats, masks, probs)。"""
    feats = np.stack([encode_state(g, s) for g, s in zip(games, seats)])
    masks = torch.stack([legal_discard_mask(g.players[s].hand_counts)
                         for g, s in zip(games, seats)])
    x = torch.from_numpy(feats).to(device)
    m = masks.to(device)
    with torch.no_grad():
        q, _ = model(x)
        logits = q.masked_fill(~m, -1e9)
        if temp != 1.0:
            logits = logits / max(temp, 1e-6)
        logp = F.log_softmax(logits, dim=-1)
        p = logp.exp()
        a = torch.multinomial(p, 1).squeeze(-1)
        lp = logp.gather(1, a.unsqueeze(1)).squeeze(1)
    return (a.cpu().numpy(), lp.cpu().numpy(), feats, masks.numpy(),
            p.cpu().numpy())


def run_games(model, device, games, temp, snap_p=0.0, rng=None, max_iter=900):
    """把一批局面推到终局。snap_p>0 时按概率在决策点留快照。

    返回 (games, snaps); snaps 每项 = dict(gi=主线局号, game=局面副本, seat,
    tile=主线选的, alt=按 π 采样的另一张, feat, mask, logp)。
    """
    snaps = []
    it = 0
    while it < max_iter:
        it += 1
        for g in games:
            if g.phase == "react_wait":
                _react(g)
        idx = [i for i, g in enumerate(games) if g.phase == "discard_wait"]
        if not idx:
            break
        gs = [games[i] for i in idx]
        ss = [games[i].turn for i in idx]
        tiles, lps, feats, masks, probs = _policy_batch(
            model, device, gs, ss, temp)
        for k, i in enumerate(idx):
            g, s, t = games[i], ss[k], int(tiles[k])
            if snap_p > 0 and rng.random() < snap_p:
                # 从 π 里采一个不同的替代动作; 只有一张可打就没有反事实
                p = probs[k].copy()
                p[t] = 0.0
                tot = p.sum()
                if tot > 1e-9:
                    alt = int(rng.choice(len(p), p=p / tot))
                    snaps.append({"gi": i, "game": copy.deepcopy(g),
                                  "seat": s, "tile": t, "alt": alt,
                                  "feat": feats[k], "mask": masks[k],
                                  "logp": float(lps[k])})
            g.action_discard(s, t)
    for g in games:
        if g.phase == "react_wait":
            _react(g)
    return games, snaps


def collect_counterfactual(model, device, n_games, seed0, temp=1.0,
                           snap_p=0.08, seed=0, gang_w=0.25):
    """主线 n_games 局 + 每个快照一次同墙重放, 产出反事实 advantage 样本。

    返回 dict(feats, masks, acts, logps, adv) 或 None(没采到快照)。
    adv = R_主 - R_替, 已按 rscale 之外的原始尺度给出。
    """
    rng = np.random.default_rng(seed)
    main = [Game(seed=seed0 + i, human_seat=-1, bloody=True)
            for i in range(n_games)]
    main, snaps = run_games(model, device, main, temp, snap_p, rng)
    if not snaps:
        return None

    # 每个快照: 换成 alt 那张牌, 用同一副牌墙重放到终局
    alt_games = []
    for sn in snaps:
        g = sn["game"]
        g.action_discard(sn["seat"], sn["alt"])
        alt_games.append(g)
    alt_games, _ = run_games(model, device, alt_games, temp)

    feats, masks, acts, logps, adv = [], [], [], [], []
    for sn, ga in zip(snaps, alt_games):
        s = sn["seat"]
        feats.append(sn["feat"])
        masks.append(sn["mask"])
        acts.append(sn["tile"])
        logps.append(sn["logp"])
        adv.append(default_reward(main[sn["gi"]], s, gang_w)
                   - default_reward(ga, s, gang_w))
    return {"feats": np.stack(feats), "masks": np.stack(masks),
            "acts": np.array(acts), "logps": np.array(logps, dtype=np.float32),
            "adv": np.array(adv, dtype=np.float32),
            "n_main_decisions": sum(len(p.discards) for g in main
                                    for p in g.players)}


def default_reward(g, seat, gang_w=0.25):
    """名次奖励 + gang_w × 自己杠分(放杠不罚)。实测 gang_w=0.25 几乎不抬噪声。"""
    r = g.rank_rewards()[seat]
    v = 0.0
    for rec in g.gang_records:
        if rec["seat"] != seat:
            continue
        if rec["kind"] == "ming":
            v += 3.0
        else:
            v += float(len([x for x in rec.get("active", [0, 1, 2, 3])
                            if x != seat]))
    return r + gang_w * v
