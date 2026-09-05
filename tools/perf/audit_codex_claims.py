"""Codex 2026-09-05 诊断意见的逐条本机验证(只读, 不改训练/评估代码)。

Codex 检查的是 fe68da8(2026-08-30)快照; 其引用的 grpo_train.py / mj159.c /
bot_v1.py / ppo.py / env159.py / paired_eval.py 自那以后一行未改, 所以静态结论
仍适用于当前 HEAD。本脚本复现其中四条可在本机判定的断言:

  1. C 原生进张原语在四副露后错误处理单钓: tiles_info() 走 shanten_code(),
     而 need==0 特判只加在外层 mj_shanten() —— 单钓会被算成两面/嵌张进张。
     影响 native.choose_discard_v10 => NativeV10/NativeV31(我们的基准对手)。
  2. features_v2 走的 native.shanten/discard_shanten/waits_ukeire 是否受 1 影响
     (它们经 mj_shanten / mj_is_win, 预期不受影响)。
  3. bot_v1.choose_discard 的 visible 计数把 `if q.seat == self.seat` 写在循环外,
     只有座位3 把自己手牌计入 visible(NativeV1 已注明是刻意复刻的既有行为)。
  4. public_state() 是否泄露对手摸牌与当前行动家的杠选项(部署侧信息边界)。

另外把 bot_v29 的 1-(1-W/T)^k 与无放回精确式 1-C(T-W,k)/C(T,k) 的偏差量化存档
(该文件属已证伪归档实验, 只记录不修改)。

用法:
  python -m tools.perf.audit_codex_claims --games 200 \
      --out logs/audit_codex_20260905.json
"""

import argparse
import hashlib
import json
import os
import random
import subprocess
from math import comb

from backend.ai.bot_native import NativeV31
from backend.ai.bot_v1 import Bot as V1Bot
from backend.ai.bot_v31 import Bot as PyV31
from backend.ai.bot_v31 import _second_step_m, _sh, _tiles_m, _ukeire_m
from backend.game.bot_driver import try_self_gang
from backend.game.engine import Game
from backend.native import native
from backend.rules.ting import waiting_tiles
from backend.rules.win import shanten as py_shanten

RED = 27
CONT_MAX = 2


def _py_candidates_v31(hand, n_melds, unseen):
    """Python 参考(bot_v31 的副露感知打分, risk 权重为 0 时与 C 同口径)。"""
    hand = tuple(hand)
    unseen = tuple(unseen)
    out = []
    min_sh = min(_sh(_minus(hand, t), n_melds) for t in range(28) if hand[t] > 0)
    for t in range(28):
        if hand[t] <= 0:
            continue
        h = _minus(hand, t)
        s = _sh(h, n_melds)
        if s > min_sh:
            u, cont, score = 0, 0.0, -10.0 * 100.0 - 100.0 * s
        else:
            u = _ukeire_m(h, n_melds, unseen)
            cont = _second_step_m(h, n_melds, unseen) if s <= CONT_MAX else 0.0
            score = 1.0 * u + 0.5 * cont
        out.append({"tile": t, "shanten": s, "ukeire": u, "cont": cont,
                    "score": score, "tiles": list(_tiles_m(h, n_melds)[1])})
    return out


def _c_candidates(hand, unseen):
    return native.score_discards_v10(hand, unseen, [0] * 28, 0.0,
                                     100.0, 1.0, 0.5, 0.0, CONT_MAX)


def _minus(hand, t):
    h = list(hand)
    h[t] -= 1
    return tuple(h)


def check_native_four_melds(rng, n_random=300):
    """断言1: 四副露(暗牌2张)时 C 的进张集与 Python 参考是否一致。"""
    # Codex 的构造例: 暗牌 一条+九条, 一条已绝张、九条只剩1张
    hand = [0] * 28
    hand[0] = 1
    hand[8] = 1
    unseen = [4] * 28
    unseen[0] = 0     # 一条绝张 => 留一条是死听
    unseen[8] = 1     # 九条剩1张 => 留九条有1张真实进张
    py = _py_candidates_v31(hand, 4, unseen)
    c = _c_candidates(hand, unseen)
    constructed = {
        "hand": {0: hand[0], 8: hand[8]},
        "unseen": {0: unseen[0], 8: unseen[8]},
        "python": [{k: v for k, v in d.items()} for d in py],
        "c": c,
        "python_pick": max(py, key=lambda d: d["score"])["tile"],
        "c_pick": max(c, key=lambda d: d["score"])["tile"],
    }

    mismatch_tiles = mismatch_pick = 0
    examples = []
    for _ in range(n_random):
        h = [0] * 28
        # 四副露后暗牌恒为 2 张(1 张 + 摸 1 张)
        a = rng.randrange(28)
        b = rng.randrange(28) if rng.random() < 0.7 else a
        h[a] += 1
        h[b] += 1
        u = [rng.randrange(5) for _ in range(28)]
        u[a] = min(u[a], 4 - h[a])
        u[b] = min(u[b], 4 - h[b])
        py = _py_candidates_v31(h, 4, u)
        c = _c_candidates(h, u)
        py_by_tile = {d["tile"]: d for d in py}
        bad = [d["tile"] for d in c
               if d["ukeire"] != py_by_tile[d["tile"]]["ukeire"]]
        py_pick = max(py, key=lambda d: d["score"])["tile"]
        c_pick = max(c, key=lambda d: d["score"])["tile"]
        if bad:
            mismatch_tiles += 1
        if py_pick != c_pick:
            mismatch_pick += 1
        if bad and len(examples) < 5:
            examples.append({
                "hand": {t: h[t] for t in range(28) if h[t]},
                "unseen": {t: u[t] for t in range(28) if u[t]},
                "c_ukeire": {d["tile"]: d["ukeire"] for d in c},
                "py_ukeire": {t: py_by_tile[t]["ukeire"] for t in py_by_tile},
                "py_tiles": {t: py_by_tile[t]["tiles"] for t in py_by_tile},
            })
    return {
        "constructed_example": constructed,
        "random_four_meld_hands": n_random,
        "ukeire_mismatch_hands": mismatch_tiles,
        "ukeire_mismatch_rate": mismatch_tiles / n_random,
        "pick_mismatch_hands": mismatch_pick,
        "pick_mismatch_rate": mismatch_pick / n_random,
        "examples": examples,
    }


def check_features_path(rng, n=200):
    """断言2: features_v2 用到的三个 C 原语在四副露下是否与 Python 参考一致。"""
    bad = {"shanten": 0, "discard_shanten": 0, "waits_ukeire": 0}
    for _ in range(n):
        h = [0] * 28
        total = rng.choice([1, 2])
        for _ in range(total):
            t = rng.randrange(28)
            while h[t] >= 4:
                t = rng.randrange(28)
            h[t] += 1
        if native.shanten(h) != py_shanten(h):
            bad["shanten"] += 1
        py_ds = sorted((t, py_shanten(_minus(h, t))) for t in range(28) if h[t] > 0)
        c_ds = sorted(native.discard_shanten(h))
        if py_ds != c_ds:
            bad["discard_shanten"] += 1
        if total == 1 and py_shanten(h) == 0:
            u = [rng.randrange(5) for _ in range(28)]
            u[h.index(1)] = min(u[h.index(1)], 3)
            py_u = sum(u[w] for w in waiting_tiles(list(h)))
            if native.waits_ukeire(h, u) != py_u:
                bad["waits_ukeire"] += 1
    return {"samples": n, "mismatches": bad,
            "verdict": "OK" if not any(bad.values()) else "MISMATCH"}


def check_v1_seat_dependence(rng, n=400):
    """断言3: bot_v1 的 visible 计数是否依赖绝对座位(循环作用域 bug)。"""
    fixed_discards = [1, 5, 9, 13, 17, 21, 25, RED]
    disagree = 0
    examples = []
    for _ in range(n):
        hand = []
        counts = [0] * 28
        while len(hand) < 14:
            t = rng.randrange(28)
            if counts[t] < 4:
                counts[t] += 1
                hand.append(t)
        hand.sort()

        def build(hero_seat):
            g = Game(seed=0, human_seat=-1)
            for s in range(4):
                p = g.players[s]
                p.melds = []
                p.discards = list(fixed_discards)
                if s == hero_seat:
                    p.hand = list(hand)
                else:
                    p.hand = [2, 6, 10, 14, 18, 22, 26, 3, 7, 11]
            g.phase = "discard_wait"
            g.turn = hero_seat
            return g

        t0 = V1Bot(build(0), 0).choose_discard()
        t3 = V1Bot(build(3), 3).choose_discard()

        # 意图正确的 visible(把手牌计入自己视角) —— 判定谁偏离了意图
        visible = [0] * 28
        for t in fixed_discards:
            visible[t] += 4          # 四家同样的弃牌堆
        for t in hand:
            visible[t] += 1
        intent = _v1_pick_with_visible(counts, visible)

        if t0 != t3:
            disagree += 1
            if len(examples) < 5:
                examples.append({"hand": {t: counts[t] for t in range(28) if counts[t]},
                                 "seat0_pick": t0, "seat3_pick": t3,
                                 "intended_pick": intent})
    return {"hands": n, "seat0_vs_seat3_disagree": disagree,
            "disagree_rate": disagree / n, "examples": examples,
            "note": "bot_native.NativeV1 注释已声明刻意复刻此行为(改动会变策略)"}


def _v1_pick_with_visible(counts, visible):
    """按 bot_v1 的公式, 但 visible 用正确口径(自己手牌计入)重算一遍。"""
    from backend.rules.ting import discard_options
    opts = discard_options(list(counts))
    best_tile, best_score = None, -1e9
    for o in opts:
        t = o["tile"]
        wr = int(sum(max(0, 4 - visible[w] - counts[w]) for w in o["waits"]))
        risk = 0.0
        if t != RED:
            remain = 4 - visible[t] - counts[t]
            risk = {3: 0.4, 2: 0.2, 1: 0.05, 0: 0.0}.get(max(0, remain), 0.4)
        score = -100 * o["shanten"] + 3 * wr - 25 * risk
        if score > best_score:
            best_score, best_tile = score, t
    return best_tile


def check_public_state_leak():
    """断言4: public_state(0) 是否泄露对手摸牌 / 非自家杠选项。"""
    g = Game(seed=7, human_seat=-1)
    g.phase = "discard_wait"
    g.turn = 2
    g.last_drawn = {"seat": 2, "tile": 13}
    # 给座位2 造一个暗杠选项
    g.players[2].hand = sorted(g.players[2].hand[:9] + [5, 5, 5, 5])
    st = g.public_state(0)
    return {
        "for_seat": 0,
        "acting_seat": g.turn,
        "last_drawn_exposed": st["last_drawn"],
        "gang_options_exposed": st["gang_options"],
        "opponent_hands_hidden": all(p["hand"] is None for p in st["players"]
                                     if p["seat"] != 0),
        "log_tail_sample": st["log"][-3:],
        "leak": bool(st["last_drawn"] and st["last_drawn"].get("seat") != 0)
                or bool(st["gang_options"]),
    }


def check_binomial_approx():
    """归档实验 bot_v29 的近似式与无放回精确式的偏差(只记录)。"""
    rows = []
    for T, W in ((40, 8), (30, 4), (20, 2), (12, 1)):
        for k in (1, 3, 5, 8):
            if k > T:
                continue
            approx = 1.0 - (1.0 - W / T) ** k
            exact = 1.0 - comb(T - W, k) / comb(T, k)
            rows.append({"T": T, "W": W, "k": k, "approx": round(approx, 5),
                         "exact": round(exact, 5),
                         "abs_err": round(abs(approx - exact), 5)})
    return {"formula_in_code": "backend/ai/bot_v29.py:94  1.0 - (1.0 - w/total) ** k",
            "exact_without_replacement": "1 - C(T-W, k) / C(T, k)",
            "status": "bot_v28/v29/v30 属已证伪归档实验, 不在现役链路",
            "table": rows}


def check_four_meld_frequency(n_games, seed0):
    """四副露决策点在真实对局中的出现频率, 以及 C 与 Python v31 的分歧数。"""
    total_decisions = 0
    by_melds = {}
    disagree = 0
    examples = []
    for i in range(n_games):
        g = Game(seed=seed0 + i, human_seat=-1)
        bots = {s: NativeV31(g, s) for s in range(4)}
        guard = 0
        while g.phase != "game_over" and guard < 900:
            guard += 1
            if g.phase == "discard_wait":
                s = g.turn
                if try_self_gang(g, bots[s]):
                    continue
                n_melds = len(g.players[s].melds)
                tile = bots[s].choose_discard()
                total_decisions += 1
                by_melds[n_melds] = by_melds.get(n_melds, 0) + 1
                if n_melds >= 4:
                    py_tile = PyV31(g, s).choose_discard()
                    if py_tile != tile:
                        disagree += 1
                        if len(examples) < 5:
                            examples.append({
                                "seed": seed0 + i, "seat": s,
                                "hand": {t: c for t, c in
                                         enumerate(g.players[s].hand_counts) if c},
                                "melds": [m["tile"] for m in g.players[s].melds],
                                "c_pick": tile, "python_pick": py_tile,
                            })
                g.action_discard(s, tile)
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
            raise RuntimeError("game exceeded action limit")
    four = by_melds.get(4, 0)
    return {"games": n_games, "seed0": seed0,
            "discard_decisions": total_decisions,
            "decisions_by_meld_count": by_melds,
            "four_meld_decisions": four,
            "four_meld_rate": four / max(1, total_decisions),
            "c_vs_python_disagree": disagree,
            "disagree_rate_within_four_meld": disagree / max(1, four),
            "examples": examples}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--seed0", type=int, default=206090500)
    ap.add_argument("--random-hands", type=int, default=300)
    ap.add_argument("--seed", type=int, default=20260905)
    ap.add_argument("--out", default="logs/audit_codex_20260905.json")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                          text=True).stdout.strip()
    src = {}
    for p in ("backend/native/mj159.c", "backend/ai/bot_v1.py",
              "backend/ai/bot_native.py", "backend/game/engine.py"):
        with open(p, "rb") as f:
            src[p] = hashlib.sha256(f.read()).hexdigest()[:16]

    out = {"git_head": head, "source_hashes": src, "args": vars(args)}

    print("== 1. C 原生四副露进张 ==", flush=True)
    out["native_four_meld_ukeire"] = check_native_four_melds(rng, args.random_hands)
    r = out["native_four_meld_ukeire"]
    print(f"  构造例: Python 打 {r['constructed_example']['python_pick']}, "
          f"C 打 {r['constructed_example']['c_pick']}")
    print(f"  随机 {r['random_four_meld_hands']} 手: 进张不一致 "
          f"{r['ukeire_mismatch_rate']:.1%}, 选牌不一致 {r['pick_mismatch_rate']:.1%}",
          flush=True)

    print("== 2. features 路径是否受牵连 ==", flush=True)
    out["features_path"] = check_features_path(rng)
    print(f"  {out['features_path']}", flush=True)

    print("== 3. bot_v1 座位依赖 ==", flush=True)
    out["v1_seat_dependence"] = check_v1_seat_dependence(rng)
    print(f"  座位0 vs 座位3 分歧 {out['v1_seat_dependence']['disagree_rate']:.1%}",
          flush=True)

    print("== 4. public_state 信息边界 ==", flush=True)
    out["public_state"] = check_public_state_leak()
    print(f"  leak={out['public_state']['leak']} "
          f"last_drawn={out['public_state']['last_drawn_exposed']} "
          f"gang_options={out['public_state']['gang_options_exposed']}", flush=True)

    print("== 5. bot_v29 近似式偏差(归档) ==", flush=True)
    out["binomial_approx"] = check_binomial_approx()

    print(f"== 6. 四副露决策点频率({args.games} 局) ==", flush=True)
    out["four_meld_frequency"] = check_four_meld_frequency(args.games, args.seed0)
    f = out["four_meld_frequency"]
    print(f"  弃牌决策 {f['discard_decisions']}, 四副露 {f['four_meld_decisions']} "
          f"({f['four_meld_rate']:.3%}), C 与 Python 分歧 {f['c_vs_python_disagree']}"
          f" ({f['disagree_rate_within_four_meld']:.1%})")
    print(f"  按副露数分布: {f['decisions_by_meld_count']}", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1, default=str)
    print(f"已写 {args.out}")


if __name__ == "__main__":
    main()
