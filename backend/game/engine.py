"""安康159 - 游戏引擎状态机

流程: 发牌(各13张,庄14) -> 庄家出牌 -> (其他家可碰/杠) -> 下家摸牌 -> 出牌 ...
黄庄: 轮到摸牌时牌堆剩余<=6 立即黄庄
胡牌: 仅自摸(含杠上花); 胡后从牌堆顺序翻6张, 统计1/5/9数量n, 输家各赔 n+1
杠分(非黄庄必结): 放杠赔3分; 暗杠/补杠其他三家各赔1分
"""

import random
from ..rules.tiles import (
    build_wall, counts_from_tiles, tiles_from_counts,
    tile_name, is_159,
)
from ..rules.win import is_win
from ..rules.ting import waiting_tiles

RED = 27


class Player:
    def __init__(self, seat: int, is_bot: bool = True):
        self.seat = seat            # 座位 0-3, 0为玩家(人类)
        self.is_bot = is_bot
        self.hand: list[int] = []   # 手牌(不含副露)
        self.melds: list[dict] = [] # 副露: {'type': 'peng'/'gang', 'tile': t, 'kind': 'ming'/'an'/'bu'}
        self.discards: list[int] = []  # 弃牌
        self.score_delta = 0        # 本局得分变化

    @property
    def hand_counts(self):
        return counts_from_tiles(self.hand)

    def total_hand_counts(self):
        """手牌+副露的计数(副露每张已固定,不可用于胡牌,只影响计数)"""
        return self.hand_counts


class Game:
    def __init__(self, seed: int | None = None, human_seat: int = 0):
        self.rng = random.Random(seed)
        self.wall: list[int] = build_wall()
        self.rng.shuffle(self.wall)
        self.dealer = 0  # 庄家座位(首局随机由调用方设定, 这里默认0)
        self.players = [Player(i, is_bot=(i != human_seat)) for i in range(4)]
        self.human_seat = human_seat
        self.turn = self.dealer     # 当前行动玩家
        self.phase = "init"         # init/discard_wait/draw/game_over
        self.last_discard: int | None = None
        self.last_discarder: int | None = None
        self.pending_actions: dict = {}   # seat -> {'peng':bool,'gang':bool}
        self.winner: int | None = None
        self.win_tile: int | None = None
        self.win_kind: str | None = None   # 'zimo' / 'gangshang' / 'tianhu'
        self.fan_159: list[int] = []       # 翻出的6张159牌
        self.n_159 = 0
        self.huangzhuang = False
        self.gang_records: list[dict] = []  # 杠分结算记录
        self.log: list[str] = []
        self.last_drawn: dict | None = None  # {"seat", "tile"} 刚摸的牌
        self.last_action: str = ""           # 最近动作描述
        self._deal()

    # ---------- 初始化 ----------
    def _deal(self):
        for p in self.players:
            p.hand = sorted(self.wall[:13])
            self.wall = self.wall[13:]
        # 庄家多摸一张
        last = self.wall.pop(0)
        self.players[self.dealer].hand.append(last)
        self.players[self.dealer].hand.sort()
        self.turn = self.dealer
        self.phase = "discard_wait"  # 庄家直接出牌
        self.log.append(f"发牌完成, 庄家: 座位{self.dealer}")
        # 天胡: 庄家开局 14 张即成胡牌形态。is_win 本身判定正确, 但原先只在
        # _next_draw / _draw_after_gang 里调用, 发牌阶段一次都不查, 导致天胡
        # 被漏判, 庄家明明已经胡了还被要求继续出牌。
        # (非庄家的"地胡"由 _next_draw 的自摸检查天然覆盖, 无需额外处理)
        # 与 mobile/js/engine.js 的 _deal 保持一致, 修改时必须同步两边。
        if is_win(self.players[self.dealer].hand_counts):
            self.last_drawn = {"seat": self.dealer, "tile": last}
            self.log.append(f"座位{self.dealer} 天胡")
            self._hu(self.dealer, last, "tianhu")

    # ---------- 工具 ----------
    def wall_remaining(self) -> int:
        return len(self.wall)

    def check_huangzhuang_before_draw(self) -> bool:
        """轮到抓牌时牌堆<=6 -> 黄庄"""
        if len(self.wall) <= 6:
            self.huangzhuang = True
            self.phase = "game_over"
            self.log.append("牌堆剩余<=6, 黄庄")
            return True
        return False

    # ---------- 行动 ----------
    def action_discard(self, seat: int, tile: int) -> dict:
        """出牌。返回后续状态(是否触发碰/杠询问)"""
        assert self.phase == "discard_wait" and self.turn == seat, "非法时机"
        p = self.players[seat]
        assert tile in p.hand, "手里没有这张牌"
        p.hand.remove(tile)
        p.discards.append(tile)
        self.last_discard = tile
        self.last_discarder = seat
        self.last_drawn = None  # 出牌后清除摸牌标记
        self.last_action = f"座位{seat} 打出 {tile_name(tile)}"
        self.log.append(f"座位{seat} 打出 {tile_name(tile)}")

        # 检查其他家碰/杠(不能吃, 不能点炮胡, 红中不能被碰杠)
        self.pending_actions = {}
        if tile != RED:
            for other in range(4):
                if other == seat:
                    continue
                op = self.players[other]
                cnt = op.hand.count(tile)
                can_peng = cnt >= 2
                can_gang = cnt >= 3
                if can_peng or can_gang:
                    self.pending_actions[other] = {"peng": can_peng, "gang": can_gang}
        if self.pending_actions:
            self.phase = "react_wait"
            return {"event": "react", "pending": list(self.pending_actions.keys())}
        # 无人响应 -> 下家摸牌
        return self._next_draw()

    def action_pass(self, seat: int) -> dict:
        """玩家放弃碰/杠"""
        assert self.phase == "react_wait", "非法时机"
        self.pending_actions.pop(seat, None)
        if not self.pending_actions:
            return self._next_draw()
        return {"event": "react", "pending": list(self.pending_actions.keys())}

    def action_peng(self, seat: int) -> dict:
        """碰"""
        assert self.phase == "react_wait" and seat in self.pending_actions, "不能碰"
        assert self.pending_actions[seat]["peng"], "不能碰"
        tile = self.last_discard
        p = self.players[seat]
        assert p.hand.count(tile) >= 2
        p.hand.remove(tile)
        p.hand.remove(tile)
        p.melds.append({"type": "peng", "tile": tile, "wr": len(self.wall)})
        # 弃牌堆中移除被碰的牌(标记)
        if self.players[self.last_discarder].discards and \
           self.players[self.last_discarder].discards[-1] == tile:
            self.players[self.last_discarder].discards.pop()
        self.pending_actions = {}
        self.turn = seat
        self.phase = "discard_wait"   # 碰后出牌
        self.log.append(f"座位{seat} 碰 {tile_name(tile)}")
        return {"event": "peng", "seat": seat, "tile": tile}

    def action_gang(self, seat: int, tile: int | None = None) -> dict:
        """杠。三种:
        - 明杠(点杠): react_wait 阶段, 别人打出的牌, 自己手有3张
        - 暗杠: discard_wait 阶段, 自己手有4张, tile 指定
        - 补杠: discard_wait 阶段, 已碰, 又摸到第4张
        """
        p = self.players[seat]
        if self.phase == "react_wait" and seat in self.pending_actions and \
           self.pending_actions[seat].get("gang"):
            # 明杠
            t = self.last_discard
            assert p.hand.count(t) >= 3
            for _ in range(3):
                p.hand.remove(t)
            p.melds.append({"type": "gang", "tile": t, "kind": "ming",
                            "wr": len(self.wall)})
            if self.players[self.last_discarder].discards and \
               self.players[self.last_discarder].discards[-1] == t:
                self.players[self.last_discarder].discards.pop()
            self.gang_records.append({"seat": seat, "kind": "ming", "tile": t,
                                      "from": self.last_discarder})
            self.log.append(f"座位{seat} 明杠 {tile_name(t)} (座位{self.last_discarder}放杠)")
        elif self.phase == "discard_wait" and self.turn == seat:
            assert tile is not None, "暗杠/补杠需指定牌"
            t = tile
            if t == RED:
                raise ValueError("红中不能杠")
            if p.hand.count(t) == 4:
                # 暗杠
                for _ in range(4):
                    p.hand.remove(t)
                p.melds.append({"type": "gang", "tile": t, "kind": "an",
                                "wr": len(self.wall)})
                self.gang_records.append({"seat": seat, "kind": "an", "tile": t})
                self.log.append(f"座位{seat} 暗杠 {tile_name(t)}")
            elif p.hand.count(t) == 1 and any(
                    m["type"] == "peng" and m["tile"] == t for m in p.melds):
                # 补杠
                p.hand.remove(t)
                for m in p.melds:
                    if m["type"] == "peng" and m["tile"] == t:
                        m["type"] = "gang"
                        m["kind"] = "bu"
                        break
                self.gang_records.append({"seat": seat, "kind": "bu", "tile": t})
                self.log.append(f"座位{seat} 补杠 {tile_name(t)}")
            else:
                raise ValueError("不满足杠的条件")
        else:
            raise ValueError("非法杠时机")

        self.pending_actions = {}
        self.turn = seat
        # 杠后补牌(从牌堆尾部)
        return self._draw_after_gang(seat)

    def _draw_after_gang(self, seat: int) -> dict:
        """杠后从牌堆尾补一张; 若补牌后自摸则胡(杠上花)"""
        if len(self.wall) <= 6:
            # 杠补牌不触发黄庄判定, 但补完后若不足6张且杠上花, 按规则需能翻6张
            # 这里默认: 杠补牌照常(黄庄判定只在正常轮抓时)
            pass
        if not self.wall:
            self.huangzhuang = True
            self.phase = "game_over"
            return {"event": "huangzhuang"}
        tile = self.wall.pop()  # 从尾部补牌
        self.players[seat].hand.append(tile)
        self.players[seat].hand.sort()
        self.last_drawn = {"seat": seat, "tile": tile}
        self.last_action = f"座位{seat} 杠后补牌"
        self.log.append(f"座位{seat} 杠后补牌 {tile_name(tile)}")
        # 检查杠上花
        counts = self.players[seat].hand_counts
        if is_win(counts):
            return self._hu(seat, tile, "gangshang")
        self.phase = "discard_wait"
        self.turn = seat
        return {"event": "gang_draw", "seat": seat, "tile": tile}

    def _next_draw(self) -> dict:
        """下家摸牌"""
        self.turn = (self.last_discarder + 1) % 4
        if self.check_huangzhuang_before_draw():
            return {"event": "huangzhuang"}
        tile = self.wall.pop(0)
        p = self.players[self.turn]
        p.hand.append(tile)
        p.hand.sort()
        self.last_drawn = {"seat": self.turn, "tile": tile}
        self.last_action = f"座位{self.turn} 摸牌"
        self.log.append(f"座位{self.turn} 摸牌 {tile_name(tile)}")
        # 检查自摸
        counts = p.hand_counts
        if is_win(counts):
            return self._hu(self.turn, tile, "zimo")
        # 检查是否可暗杠/补杠(提示, 由玩家决定)
        self.phase = "discard_wait"
        gang_options = self._gang_options(self.turn)
        return {"event": "draw", "seat": self.turn, "tile": tile,
                "gang_options": gang_options}

    def _gang_options(self, seat: int) -> list[int]:
        """返回当前可杠的牌(暗杠/补杠)"""
        p = self.players[seat]
        opts = []
        counts = p.hand_counts
        for t in range(27):
            if counts[t] == 4:
                opts.append(t)  # 暗杠
        for m in p.melds:
            if m["type"] == "peng" and counts[m["tile"]] >= 1:
                opts.append(m["tile"])  # 补杠
        return sorted(set(opts))

    def _hu(self, seat: int, tile: int, kind: str) -> dict:
        """胡牌结算"""
        self.winner = seat
        self.win_tile = tile
        self.win_kind = kind
        # 翻159: 从牌堆顺序抓6张
        self.fan_159 = []
        n = 0
        if len(self.wall) >= 6:
            self.fan_159 = self.wall[:6]
            n = sum(1 for t in self.fan_159 if is_159(t))
        # 若不足6张(杠上花边缘情况), 按0张算
        self.n_159 = n
        self._settle()
        self.phase = "game_over"
        self.log.append(f"座位{seat} 胡牌({kind}), 翻159: "
                        f"{[tile_name(t) for t in self.fan_159]}, n={n}")
        return {"event": "hu", "seat": seat, "kind": kind,
                "fan_159": self.fan_159, "n_159": n}

    def _settle(self):
        """结算: 杠分 + 胡牌159分"""
        for p in self.players:
            p.score_delta = 0
        # 杠分
        for rec in self.gang_records:
            s = rec["seat"]
            if rec["kind"] == "ming":
                self.players[rec["from"]].score_delta -= 3
                self.players[s].score_delta += 3
            else:  # an / bu
                for other in range(4):
                    if other != s:
                        self.players[other].score_delta -= 1
                        self.players[s].score_delta += 1
        # 胡牌159分
        if self.winner is not None:
            per_loser = self.n_159 + 1
            for other in range(4):
                if other != self.winner:
                    self.players[other].score_delta -= per_loser
                    self.players[self.winner].score_delta += per_loser

    # ---------- 状态导出 ----------
    def public_state(self, for_seat: int) -> dict:
        """从某座位视角的公开状态(隐藏其他家手牌)"""
        ps = []
        for p in self.players:
            ps.append({
                "seat": p.seat,
                "hand": sorted(p.hand) if p.seat == for_seat else None,
                "hand_count": len(p.hand),
                "melds": p.melds,
                "discards": p.discards,
                "score_delta": p.score_delta,
                "is_dealer": p.seat == self.dealer,
            })
        return {
            "players": ps,
            "dealer": self.dealer,
            "turn": self.turn,
            "phase": self.phase,
            "wall_remaining": len(self.wall),
            "last_discard": self.last_discard,
            "last_discarder": self.last_discarder,
            "pending_actions": self.pending_actions,
            "winner": self.winner,
            "win_kind": self.win_kind,
            "fan_159": self.fan_159,
            "n_159": self.n_159,
            "huangzhuang": self.huangzhuang,
            "gang_records": self.gang_records,
            "last_drawn": self.last_drawn,
            "last_action": self.last_action,
            "gang_options": (self._gang_options(self.turn)
                             if self.phase == "discard_wait" else []),
            "log": self.log[-30:],
        }
