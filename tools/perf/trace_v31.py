"""4×v31 自对弈的逐决策对局记录(座0 主视角)。

用 NativeV31(与 bot_v31 逐位一致的 C 实现)。每个座0 决策点记录:
  - 出牌: 手牌、每个候选的 (向听/进张/两步推演值/总分), 标出实际选择
  - 碰杠: 谁打了什么、能否碰/杠、碰杠前后向听、决定及理由
其他家的碰/杠也记一行(影响座0 的进张池)。输出到文件, 供人工审局。

用法: python -m tools.perf.trace_v31 --games 5 --seed0 961000 --out logs/v31_trace.txt
"""

import argparse

from backend.ai.bot_native import NativeV31
from backend.game.engine import Game
from backend.native import native
from backend.rules.tiles import tile_name

RED = 27


def hand_str(g, seat):
    p = g.players[seat]
    parts = [tile_name(t) for t in sorted(p.hand)]
    if p.melds:
        parts.append("| 副露:" + " ".join(
            ("碰" if m["type"] == "peng" else ("明杠" if m["kind"] == "ming"
                                               else ("暗杠" if m["kind"] == "an"
                                                     else "补杠")))
            + tile_name(m["tile"]) for m in p.melds))
    return " ".join(parts)


def meld_str(m):
    kind = "碰" if m["type"] == "peng" else \
        {"ming": "明杠", "an": "暗杠", "bu": "补杠"}.get(m["kind"], "?")
    return f"{kind}{tile_name(m['tile'])}"


class Tracer:
    def __init__(self, out, hero=0):
        self.out = out
        self.hero = hero
        self.turn = 0

    def w(self, s=""):
        self.out.write(s + "\n")

    def unseen(self, g):
        visible = [0] * 28
        for q in g.players:
            for t in q.discards:
                visible[t] += 1
            for m in q.melds:
                visible[m["tile"]] += 3 if m["type"] == "peng" else 4
        for t, n in enumerate(g.players[self.hero].hand_counts):
            visible[t] += n
        return [max(0, 4 - visible[t]) for t in range(28)]

    def penged_by_others(self, g):
        m = [0] * 28
        for q in g.players:
            if q.seat == self.hero:
                continue
            for mm in q.melds:
                if mm["type"] == "peng":
                    m[mm["tile"]] = 1
        return m

    def hero_discard(self, g, bot):
        self.turn += 1
        p = g.players[self.hero]
        unseen = self.unseen(g)
        penged = self.penged_by_others(g)
        eg = max(0.0, min(1.0, (60 - g.wall_remaining()) / 60.0))
        rows = native.score_discards_v10(p.hand_counts, unseen, penged, eg)
        choice = bot.choose_discard()
        best = max(rows, key=lambda r: r["score"])["tile"]
        assert choice == best, f"trace 与 bot 不一致: {best} vs {choice}"
        sh_now = native.shanten(
            [max(0, c - (1 if t == choice else 0))
             for t, c in enumerate(p.hand_counts)])
        self.w(f"第{self.turn}巡 摸 {tile_name(g.last_drawn['tile']) if g.last_drawn else '?'} "
               f"| 墙剩 {g.wall_remaining()} | 打后向听 {sh_now}")
        self.w(f"  手牌: {hand_str(g, self.hero)}")
        for r in sorted(rows, key=lambda r: -r["score"]):
            mark = " ← 选" if r["tile"] == choice else ""
            if r["shanten"] > min(x["shanten"] for x in rows):
                self.w(f"    打 {tile_name(r['tile']):4s} 向听{r['shanten']}"
                       f"  (淘汰: 非最小向听){mark}")
            else:
                self.w(f"    打 {tile_name(r['tile']):4s} 向听{r['shanten']} "
                       f"进张{r['ukeire']:2d}  两步{r['cont']:5.2f}  "
                       f"总分{r['score']:7.2f}  剩{unseen[r['tile']]}张{mark}")
        return choice

    def hero_react(self, g, bot):
        tile = g.last_discard
        who = g.last_discarder
        pend = g.pending_actions[self.hero]
        p = g.players[self.hero]
        counts = list(p.hand_counts)
        before = native.shanten(counts)
        line = f"  碰杠机会: 座{who}打{tile_name(tile)}"
        if pend.get("gang"):
            c = list(counts)
            c[tile] -= 3
            after = native.shanten(c)
            ok = bot.decide_gang(tile, "ming")
            line += f" | 可明杠: 向听{before}→{after} => {'杠!' if ok else '不杠'}"
        if pend.get("peng"):
            c = list(counts)
            c[tile] -= 2
            after11 = native.shanten(c)
            ok = bot.decide_peng(tile)
            line += f" | 可碰: 向听{before}→{after11} => {'碰!' if ok else '不碰'}"
        self.w(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=5)
    ap.add_argument("--seed0", type=int, default=961000)
    ap.add_argument("--hero", type=int, default=0)
    ap.add_argument("--out", type=str, default="logs/v31_trace.txt")
    args = ap.parse_args()

    with open(args.out, "w", encoding="utf-8") as out:
        for gi in range(args.games):
            seed = args.seed0 + gi
            g = Game(seed=seed, human_seat=-1)
            hero = args.hero
            bots = {i: NativeV31(g, i) for i in range(4)}
            tr = Tracer(out, hero)
            tr.w("=" * 70)
            tr.w(f"局 seed={seed}  主视角=座{hero}(庄家={'是' if g.dealer == hero else '否'})")
            tr.w(f"起始手牌: {hand_str(g, hero)}")
            tr.w("-" * 70)
            guard = 0
            while g.phase != "game_over" and guard < 500:
                guard += 1
                if g.phase == "discard_wait":
                    seat = g.turn
                    if seat == hero:
                        tile = tr.hero_discard(g, bots[seat])
                    else:
                        tile = bots[seat].choose_discard()
                    g.action_discard(seat, tile)
                    # 对手对该出牌/我们出牌的即时反应在 react 分支记录
                    if g.phase == "react_wait" and seat != hero:
                        if hero in g.pending_actions:
                            tr.hero_react(g, bots[hero])
                else:
                    s = list(g.pending_actions.keys())[0]
                    pend = g.pending_actions[s]
                    b = bots[s]
                    tile = g.last_discard
                    if pend.get("gang") and b.decide_gang(tile, "ming"):
                        g.action_gang(s)
                        if s != hero:
                            tr.w(f"  座{s} 明杠{tile_name(tile)}(别人打出的)")
                    elif pend.get("peng") and b.decide_peng(tile):
                        g.action_peng(s)
                        if s != hero:
                            tr.w(f"  座{s} 碰{tile_name(tile)}")
                    else:
                        g.action_pass(s)
            tr.w("-" * 70)
            if g.winner is not None:
                tr.w(f"结局: 座{g.winner} {g.win_kind} 自摸和牌, "
                     f"翻159: {[tile_name(t) for t in g.fan_159]} n={g.n_159}")
            else:
                tr.w("结局: 黄庄")
            tr.w(f"得分: " + "  ".join(
                f"座{p.seat} {p.score_delta:+d}" for p in g.players))
            tr.w(f"杠记录: " + (", ".join(
                f"座{r['seat']}{r['kind']}杠{tile_name(r['tile'])}"
                for r in g.gang_records) or "无"))
            tr.w("")
    print(f"已写入 {args.out}")


if __name__ == "__main__":
    main()
