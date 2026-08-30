"""牌运 vs 打法: 终局得分的方差里, 有多少在发牌那一刻就定了。

回答三件事:
  1. 开局手牌质量(向听/进张)能解释多少终局得分方差 -> critic 在发牌时的天花板
  2. 中盘状态 E 能解释多少 -> critic 在对局中的天花板
  3. 每次弃牌的实际损失(E_选中 - E_最优, 巡)与终局得分的相关 -> ΔE 密集奖励的信噪比

四座位都用同一个 NN(与训练采集口径一致, temp=1), 奖励用"放杠不罚"的调整得分。

用法: python -m tools.perf.diag_luck_vs_skill --games 80
"""

import argparse

import numpy as np
import torch

from backend.analysis import hv_native
from backend.game.engine import Game
from backend.native import native
from backend.rl.features_v2 import encode_state
from backend.rl.model import build_model, legal_discard_mask
from backend.rules.tiles import TILE_COUNT


def visible_counts(game, seat):
    vis = [0] * TILE_COUNT
    for q in game.players:
        for t in q.discards:
            vis[t] += 1
        for m in q.melds:
            vis[m["tile"]] += 3 if m["type"] == "peng" else 4
    for t, n in enumerate(game.players[seat].hand_counts):
        vis[t] += n
    return vis


def e_table(game, seat, kai):
    """当前 14 张手牌的整张弃牌表 {tile: E}; 非 3n+2 返回 None。"""
    hand = list(game.players[seat].hand_counts)
    if sum(hand) % 3 != 2:
        return None
    vis = visible_counts(game, seat)
    es = {}
    for t in range(TILE_COUNT):
        if hand[t]:
            hv_native.set_hand(hand, vis, 1.0, kai > 0, 2, kai, 6)
            es[t] = hv_native.e_after_discard(t)
    return es


def adjusted(game, seat):
    """放杠(明杠的弃牌方)不扣分。"""
    s = game.players[seat].score_delta
    for rec in game.gang_records:
        if rec["kind"] == "ming" and rec["from"] == seat:
            s += 3
    return float(s)


def deal_quality(hand):
    """开局手牌质量 (向听, 进张)。庄家 14 张时取最优弃牌后的 13 张口径,
    这样四家可比。unseen = 4 - 自己持有(发牌时别的都还没露)。"""
    unseen = [4 - n for n in hand]
    if sum(hand) % 3 == 2:
        best_sh, best_uk = 99, 0
        for t in range(TILE_COUNT):
            if not hand[t]:
                continue
            h = list(hand)
            h[t] -= 1
            sh = native.shanten(h)
            uk = native.waits_ukeire(h, unseen)
            if (sh, -uk) < (best_sh, -best_uk):
                best_sh, best_uk = sh, uk
        return best_sh, best_uk
    return native.shanten(hand), native.waits_ukeire(hand, unseen)


def r2(x, y):
    """一元线性回归的 R²(等于相关系数平方)。"""
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 10 or x[ok].std() == 0:
        return float("nan")
    return float(np.corrcoef(x[ok], y[ok])[0, 1] ** 2)


def r2_multi(X, y):
    """多元最小二乘的 R²。"""
    X, y = np.asarray(X, float), np.asarray(y, float)
    ok = np.isfinite(X).all(1) & np.isfinite(y)
    X, y = X[ok], y[ok]
    A = np.column_stack([X, np.ones(len(X))])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ beta
    return float(1 - resid.var() / y.var())


def report(rows, mid_turn):
    G = np.array([r["G"] for r in rows])
    sh0 = np.array([r["sh0"] for r in rows], float)
    uk0 = np.array([r["uk0"] for r in rows], float)
    midE = np.array([r["midE"] for r in rows])
    loss = np.array([r["loss"] for r in rows])
    win = np.array([r["win"] for r in rows])
    S = G.reshape(-1, 4)

    print(f"\n单元 {len(G)} (局 {len(S)})   得分 均 {G.mean():+.2f} "
          f"标准差 {G.std():.2f}   胜率 {win.mean():.1%}")
    print(f"共同分量 C=四家均值: 标准差 {S.mean(1).std():.3f}  "
          f"占方差 {S.mean(1).var() / G.var():.2%}  <- 配对能消掉的全部")

    print("\n=== 终局得分方差的可解释部分 (R²) ===")
    print(f"  开局向听 sh0                : {r2(sh0, G):.3f}")
    print(f"  开局进张 uk0                : {r2(uk0, G):.3f}")
    print(f"  开局 sh0+uk0 (多元)         : "
          f"{r2_multi(np.column_stack([sh0, uk0]), G):.3f}")
    print(f"  巡{mid_turn} 状态 E                 : {r2(midE, G):.3f}")
    print(f"  巡{mid_turn} E + 开局 (多元)        : "
          f"{r2_multi(np.column_stack([midE, sh0, uk0]), G):.3f}")
    print(f"  平均出牌损失 (打法)         : {r2(loss, G):.3f}")
    ok = np.isfinite(loss)
    print(f"\n出牌损失: 均 {loss[ok].mean():.3f} 巡  标准差 "
          f"{loss[ok].std():.3f}  与得分相关 "
          f"{np.corrcoef(loss[ok], G[ok])[0, 1]:+.3f}")
    print(f"  与是否胡牌相关 {np.corrcoef(loss[ok], win[ok])[0, 1]:+.3f}")


KEYS = ("G", "sh0", "uk0", "midE", "loss", "win")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=80)
    ap.add_argument("--model", default="models/acnn_v2.pt")
    ap.add_argument("--seed0", type=int, default=8000000)
    ap.add_argument("--kai", type=int, default=0, help="E 的换型档位")
    ap.add_argument("--mid-turn", type=int, default=5,
                    help="在第几次弃牌时取中盘状态 E")
    ap.add_argument("--loss-stride", type=int, default=2,
                    help="每隔几次弃牌算一次出牌损失(整表 E 较贵)")
    ap.add_argument("--save", help="把逐单元结果存成 npz")
    ap.add_argument("--load", nargs="+", help="读若干 npz 汇总后直接出报告")
    args = ap.parse_args()

    if args.load:
        rows = []
        for p in args.load:
            d = np.load(p)
            rows += [dict(zip(KEYS, vals))
                     for vals in zip(*[d[k] for k in KEYS])]
        print(f"汇总 {len(args.load)} 份数据")
        report(rows, args.mid_turn)
        return

    ck = torch.load(args.model, map_location="cpu", weights_only=True)
    model = build_model(ck["size"])
    model.load_state_dict(ck["model"])
    model.eval()
    print(f"模型 {args.model} (size={ck['size']}, iter={ck.get('iter')})  "
          f"后端 {hv_native.backend_kind()}  kai={args.kai}")

    rows = []          # 每个 (局, 座位) 一行
    for gi in range(args.games):
        g = Game(seed=args.seed0 + gi, human_seat=-1)
        # 开局手牌质量
        deal = {}
        for s in range(4):
            deal[s] = deal_quality(list(g.players[s].hand_counts))
        midE = {s: float("nan") for s in range(4)}
        loss_sum = {s: 0.0 for s in range(4)}
        loss_n = {s: 0 for s in range(4)}
        ndisc = {s: 0 for s in range(4)}

        guard = 0
        while g.phase != "game_over" and guard < 500:
            guard += 1
            if g.phase == "discard_wait":
                s = g.turn
                x = torch.from_numpy(encode_state(g, s)).unsqueeze(0)
                mask = legal_discard_mask(
                    g.players[s].hand_counts).unsqueeze(0)
                with torch.no_grad():
                    q = model.q(x, mask)[0]
                    pr = torch.softmax(q, -1)
                tile = int(torch.multinomial(pr, 1).item())
                ndisc[s] += 1
                need_mid = ndisc[s] == args.mid_turn
                need_loss = ndisc[s] % args.loss_stride == 0
                if need_mid or need_loss:
                    es = e_table(g, s, args.kai)
                    if es:
                        if need_mid:
                            midE[s] = min(es.values())
                        if need_loss and tile in es:
                            loss_sum[s] += es[tile] - min(es.values())
                            loss_n[s] += 1
                g.action_discard(s, tile)
            else:
                s = list(g.pending_actions.keys())[0]
                pend = g.pending_actions[s]
                hc = g.players[s].hand_counts
                if pend.get("gang") and native.decide_gang(
                        31, hc, g.last_discard, "ming"):
                    g.action_gang(s)
                elif pend.get("peng") and native.decide_peng(
                        31, hc, g.last_discard):
                    g.action_peng(s)
                else:
                    g.action_pass(s)

        for s in range(4):
            rows.append({
                "g": gi, "seat": s, "G": adjusted(g, s),
                "sh0": deal[s][0], "uk0": deal[s][1], "midE": midE[s],
                "loss": loss_sum[s] / loss_n[s] if loss_n[s] else float("nan"),
                "win": 1.0 if g.winner == s else 0.0,
            })
        if (gi + 1) % 20 == 0:
            print(f"  ...{gi + 1}/{args.games} 局")

    if args.save:
        np.savez(args.save,
                 **{k: np.array([r[k] for r in rows], float) for k in KEYS})
        print(f"已存 {args.save}")
    report(rows, args.mid_turn)


if __name__ == "__main__":
    main()
