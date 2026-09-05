"""Mortal(日麻开源AI) -> 安康159 的知识蒸馏管线。

思路(用户提出): 不要求规则一致 —— 让 Mortal 在它自己的规则下自对弈, 从牌谱
里蒸馏可迁移的牌型知识。日麻的三门数牌(27种)与我们完全同构, 映射:
  1s-9s(索) -> 条0-8   1p-9p(筒) -> 饼9-17   1m-9m(万) -> 万18-26
字牌(1z-7z)与赤宝(0m/0p/0s)无对应 -> 过滤掉含字牌的决策点。

两段:
  1. compare: 解析 mjai 牌谱 -> 还原每个弃牌决策的公开局面 -> 我们的 628 维
     特征 -> 对比"分析器 E 的最优弃牌"与"Mortal 实际弃牌", 量化知识差。
     先测再蒸: 若一致率高, Mortal 没有超出 E 的东西; 分歧集中处才是可蒸的。
  2. distill: 把 Mortal 的映射弃牌当硬标签训练(单独或混合 E 软标签)。

已知的迁移风险(必须过滤):
  - 断幺倾向: Mortal 系统性避 1/9(做断幺九), 我们无番种, 这是伪知识;
    对比报告里单独统计"分歧是否集中在 1/9 vs 中张"来识别。
  - 吃/立直/振听逻辑: 我们没有, 状态里的副露只取碰/杠。
  - 场风/自风牌: 字牌, 已过滤。

用法:
  python -m tools.distill_mortal compare --logs "third_party/logs/*.json"
  python -m tools.distill_mortal distill --logs ... --out models/mortal_bc.npz
"""

import argparse
import glob
import gzip
import json
import os

import numpy as np

TILE_MAP = {}          # mjai pai 字符串 -> 我们的 tile id
for i in range(1, 10):
    TILE_MAP[f"{i}s"] = i - 1          # 索 -> 条
    TILE_MAP[f"{i}p"] = 8 + i          # 筒 -> 饼
    TILE_MAP[f"{i}m"] = 17 + i         # 万 -> 万
# 赤宝牌: mjai 记法 0m/0p/0s 或大写 M/P/S, 本质是 5m/5p/5s -> 映射为对应五
# (日麻每门 5 有 5 份拷贝, 我们只有 4 份; analyzer 的 unseen=max(0,4-v) 会
#  clamp, 形状知识蒸馏可接受该形变)
TILE_MAP["0s"] = TILE_MAP["S"] = 4
TILE_MAP["0p"] = TILE_MAP["P"] = 13
TILE_MAP["0m"] = TILE_MAP["M"] = 22
HONORS = {f"{i}z" for i in range(1, 8)}


class FakePlayer:
    def __init__(self):
        self.hand = []                 # tile id 列表(28 制)
        self.melds = []                # {'type':'peng'/'gang','tile':t}
        self.discards = []

    @property
    def hand_counts(self):
        c = [0] * 28
        for t in self.hand:
            c[t] += 1
        return c


class FakeGame:
    """最小 Game 替身: 只实现 features.py/features_v2.py 用到的接口。"""

    def __init__(self, dealer=0, wall_rem=70):
        self.players = [FakePlayer() for _ in range(4)]
        self.dealer = dealer
        self._wall = wall_rem

    def wall_remaining(self):
        return self._wall


def parse_log(path):
    """解析一局 mjai 牌谱, 产出弃牌决策点:
    dict(seat, hand28(14张), melds/discards 局面, wall_rem, chosen, honors_n)
    """
    g = FakeGame()
    out = []
    started = False
    ndraw = 0
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ev = json.loads(line)
            ty = ev.get("type")
            if ty == "start_game":
                continue
            if ty == "start_kyoku":
                # 新一局: tehais 是四家起手 13 张
                g = FakeGame()
                ndraw = 0
                for s_ in range(4):
                    g.players[s_].hand = [TILE_MAP.get(pai, -1)
                                          for pai in ev["tehais"][s_]]
                g.dealer = ev.get("oya", 0)
                g._wall = 70
                started = True
                continue
            if not started:
                continue
            if ty in ("end_kyoku", "ryuukyoku", "hora", "end_game"):
                started = False
                continue
            if ty == "tsumo":
                pai = ev["pai"]
                if pai in TILE_MAP:
                    g.players[ev["actor"]].hand.append(TILE_MAP[pai])
                else:
                    g.players[ev["actor"]].hand.append(-1)   # 字牌占位
                g._wall = max(0, g._wall - 1)
                ndraw += 1
            elif ty == "dahai":
                seat = ev["actor"]
                pai = ev["pai"]
                hand = g.players[seat].hand
                honors = sum(1 for t in hand if t == -1)
                if pai in TILE_MAP and sum(len(p.hand) for p in g.players):
                    tid = TILE_MAP[pai]
                    if tid in hand:
                        out.append({
                            "seat": seat,
                            "hand": list(hand),
                            "melds": [list(m) for m in
                                      (p.melds for p in g.players)],
                            "discards": [list(p.discards) for p in g.players],
                            "wall_rem": g._wall,
                            "chosen": tid,
                            "honors_n": honors,
                        })
                # 落实弃牌
                real = TILE_MAP.get(pai, -1)
                if real in hand:
                    hand.remove(real)
                elif -1 in hand:
                    hand.remove(-1)
                g.players[seat].discards.append(real if real >= 0 else -1)
            elif ty in ("pon", "minkan", "ankan", "kakan"):
                seat = ev["actor"]
                pai = ev.get("pai")
                tid = TILE_MAP.get(pai, -1)
                n_rm = 2 if ty == "pon" else (3 if ty == "minkan" else
                                              (4 if ty == "ankan" else 1))
                if tid >= 0:
                    kind = "peng" if ty == "pon" else "gang"
                    g.players[seat].melds.append(
                        {"type": kind, "tile": tid})
                    for _ in range(n_rm):
                        if tid in g.players[seat].hand:
                            g.players[seat].hand.remove(tid)
                else:
                    # 字牌副露: 不进 melds(我们没有对应牌), 只维护占位计数
                    for _ in range(n_rm):
                        if -1 in g.players[seat].hand:
                            g.players[seat].hand.remove(-1)
                if ty == "pon" and "target" in ev:
                    tg = ev["target"]
                    ds = g.players[tg].discards
                    if ds and ds[-1] == tid:
                        ds.pop()
            elif ty == "reach":
                pass    # 立直对我们无对应, 忽略
    return out


def build_fake_game(rec):
    """决策点 -> FakeGame(把 hand 里 -1 剔除, 特征只用数牌信息)。"""
    g = FakeGame(wall_rem=rec["wall_rem"])
    for s in range(4):
        g.players[s].hand = [t for t in rec["hand"] if t >= 0] if s == rec["seat"] \
            else []                        # 别家手牌不可见, 特征只用计数公开段
        g.players[s].melds = rec["melds"][s]
        g.players[s].discards = [t for t in rec["discards"][s] if t >= 0]
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["compare", "distill"])
    ap.add_argument("--logs", required=True)
    ap.add_argument("--max-honors", type=int, default=3,
                    help="决策点手里允许的字牌数(日麻手牌几乎总带字牌, "
                         "0 会滤掉几乎所有决策点)")
    ap.add_argument("--out", default="models/mortal_bc.npz")
    args = ap.parse_args()

    paths = []
    for p in args.logs.split():
        paths += sorted(glob.glob(p))
    recs = []
    for p in paths:
        for r in parse_log(p):
            if r["honors_n"] <= args.max_honors and len(r["hand"]) >= 13:
                recs.append(r)
    print(f"牌谱 {len(paths)} 局 -> 纯数牌决策点 {len(recs)}")
    if not recs:
        return

    if args.cmd == "compare":
        compare(recs)
    else:
        distill(recs, args.out)


def compare(recs):
    from backend.analysis import hv_native
    from backend.rl.features_v2 import encode_state
    agree = 0
    dE = []
    term_mismatch = 0      # 分歧且 Mortal 打的是中张而 E 要打幺九(断幺嫌疑)
    n = 0
    for r in recs[:5000]:
        hand = [0] * 28
        for t in r["hand"]:
            if t >= 0:
                hand[t] += 1
        if sum(hand) % 3 != 2:
            continue
        g = build_fake_game(r)
        seat = r["seat"]
        # 分析器最优
        vis = [0] * 28
        for t in hand:
            vis[t] += 1 if t >= 0 else 0
        for s in range(4):
            for t in r["discards"][s]:
                if t >= 0:
                    vis[t] += 1
            for m in r["melds"][s]:
                if m["tile"] >= 0:
                    vis[m["tile"]] += 3 if m["type"] == "peng" else 4
        hv_native.set_hand(hand, vis, 1.0, False, 2, 0, 6)
        es = {t: hv_native.e_after_discard(t) for t in range(28) if hand[t]}
        if not es:
            continue
        e_best = min(es, key=es.get)
        n += 1
        if e_best == r["chosen"]:
            agree += 1
        else:
            de = es[r["chosen"]] - es[e_best]
            dE.append(de)
            def is_term(t): return t % 9 in (0, 8)
            if is_term(e_best) and not is_term(r["chosen"]):
                term_mismatch += 1
    if n == 0:
        print("无可比决策(牌谱太少或全被过滤)")
        return
    print(f"可比决策 {n}")
    print(f"E最优 与 Mortal 实际弃牌 一致率: {agree / n:.1%}")
    if dE:
        dE = np.array(dE)
        print(f"分歧时 Mortal 选择的 E 代价: 均 {dE.mean():.3f} 巡  "
              f"中位 {np.median(dE):.3f}")
        print(f"分歧中'E 要打幺九而 Mortal 留幺九打中张'(断幺伪知识嫌疑): "
              f"{term_mismatch}/{len(dE)} ({term_mismatch / len(dE):.1%})")


def distill(recs, out):
    from backend.rl.features_v2 import encode_state
    feats, bests = [], []
    for r in recs:
        g = build_fake_game(r)
        feats.append(encode_state(g, r["seat"]))
        bests.append(r["chosen"])
    np.savez_compressed(out, feats=np.stack(feats),
                        bests=np.array(bests, dtype=np.int8),
                        labels=np.zeros(len(bests), dtype=np.float32))
    print(f"已存 {out}  {len(bests)} 条 (628维, Mortal 硬标签)")


if __name__ == "__main__":
    main()
