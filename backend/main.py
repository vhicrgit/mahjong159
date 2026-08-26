"""安康159 - FastAPI 后端

提供:
- POST /api/new_game    开新局
- GET  /api/state       获取当前状态(玩家视角)
- POST /api/discard     出牌
- POST /api/peng        碰
- POST /api/gang        杠
- POST /api/pass        过
- GET  /api/analyze     实时分析
AI 回合自动推进。
"""

import uuid
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import os

from .game.engine import Game
from .ai.bot import Bot
from .ai import roster
from .analysis.analyzer import Analyzer
from .rules.tiles import tile_name

app = FastAPI(title="安康159麻将")

# 单局会话(单用户单局, 足够)
_session = {}


def current_game() -> Game:
    g = _session.get("game")
    if g is None:
        raise HTTPException(400, "没有进行中的对局, 请先开新局")
    return g


def _make_ai_bot(g: Game, seat: int):
    """按档位名构造单个 Bot(三席统一档位时使用)"""
    kind = _session.get("bot_kind") or os.environ.get("MAHJONG_BOT", "v31")
    param = int(_session.get("bot_param")
                or os.environ.get("MAHJONG_BOT_PARAM", "0"))
    if kind in ("v31", "normal"):
        return Bot(g, seat)
    if kind == "v10":
        from .ai.bot_v10 import Bot as B
        return B(g, seat)
    if kind == "v1":
        from .ai.bot_v1 import Bot as B
        return B(g, seat)
    if kind == "target":
        from .ai.bot_target import Bot as B
        return B(g, seat)
    if kind == "cheat_wall":
        from .ai.bot_cheat import Bot as B
        return B(g, seat, wall_lookahead=param or 32,
                 see_opponents=False, beam=12, rollout=False)
    if kind == "cheat_opp":
        from .ai.bot_cheat import Bot as B
        return B(g, seat, wall_lookahead=param or 32,
                 see_opponents=True, beam=12, rollout=False)
    if kind == "cheat_full":
        from .ai.bot_cheat import Bot as B
        return B(g, seat, wall_lookahead=-1, see_opponents=True,
                 beam=param or 4, rollout=True)
    return Bot(g, seat)


def _gang_kind(g: Game, seat: int, tile: int) -> str:
    if g.players[seat].hand.count(tile) == 4:
        return "an"
    return "bu"


def _build_ai_bots(g: Game) -> dict:
    """构建 AI 阵容

    默认走 roster 固定阵容(三席强度不同: 菜鸟v1 / 老鸟v31 / 挂哥cheat_wall),
    与手机版保持一致。若显式指定了档位(会话参数 bot_kind 或环境变量
    MAHJONG_BOT), 则三席统一使用该档位, 便于横向评测单一 Bot 强度。
    """
    if not (_session.get("bot_kind") or os.environ.get("MAHJONG_BOT")):
        return roster.build_bots(g, g.human_seat)
    return {i: _make_ai_bot(g, i) for i in range(4) if i != g.human_seat}


def state_with_names(g: Game) -> dict:
    """公开状态 + 每个座位的 AI 名字/说明(供前端显示对手身份)"""
    st = g.public_state(0)
    for p in st["players"]:
        p["name"] = roster.seat_name(p["seat"])
        p["bot_desc"] = roster.seat_desc(p["seat"])
    return st


def _record_decision(g: Game, actual_tile: int):
    """记录我的出牌决策点(用于赛后检讨)
    出牌前调用: 记录手牌快照 + 分析器推荐 + 实际选择"""
    try:
        az = Analyzer(g, g.human_seat)
        hand = az.analyze_hand()
        opts = az.analyze_discards()
        if not opts:
            return
        best = opts[0]
        p = g.players[g.human_seat]
        _session.setdefault("review_log", []).append({
            "step": len(_session.get("review_log", [])) + 1,
            "hand": sorted(p.hand),
            "actual": actual_tile,
            "recommended": best["tile"],
            "match": best["tile"] == actual_tile,
            "shanten_before": hand["shanten"],
            "shanten_after": best["shanten"],
            "expected_fan159": hand["expected_fan159"],
            "options": [
                {"tile": o["tile"], "name": o["name"],
                 "gang_risk": o["gang_risk"], "wait_remain": o["wait_remain"],
                 "shanten": o["shanten"]}
                for o in opts[:4]
            ],
        })
    except Exception:
        pass  # 记录失败不影响游戏


def run_bots_until_human(g: Game):
    """自动推进 AI 回合, 直到轮到人类或游戏结束"""
    bots = _build_ai_bots(g)
    guard = 0
    while g.phase != "game_over" and guard < 300:
        guard += 1
        if g.phase == "discard_wait":
            if g.turn == g.human_seat:
                break
            gang_taken = False
            for tile in g._gang_options(g.turn):
                if bots[g.turn].decide_gang(tile, _gang_kind(g, g.turn, tile)):
                    r = g.action_gang(g.turn, tile)
                    gang_taken = True
                    break
            if gang_taken:
                continue
            r = g.action_discard(g.turn, bots[g.turn].choose_discard())
        elif g.phase == "react_wait":
            # 人类优先响应
            if g.human_seat in g.pending_actions:
                break
            s = list(g.pending_actions.keys())[0]
            b = bots[s]
            if g.pending_actions[s].get("gang") and \
                    b.decide_gang(g.last_discard, "ming"):
                r = g.action_gang(s)
            elif g.pending_actions[s].get("peng") and \
                    b.decide_peng(g.last_discard):
                r = g.action_peng(s)
            else:
                r = g.action_pass(s)
        else:
            break


class NewGameReq(BaseModel):
    dealer: int = 0
    bot_kind: str | None = None
    bot_param: int | None = None


class DiscardReq(BaseModel):
    tile: int


class GangReq(BaseModel):
    tile: int | None = None


@app.post("/api/new_game")
def new_game(req: NewGameReq | None = None):
    g = Game(human_seat=0)
    if req:
        g.dealer = req.dealer
        _session["bot_kind"] = req.bot_kind
        _session["bot_param"] = req.bot_param
    else:
        _session["bot_kind"] = None
        _session["bot_param"] = None
    _session["game"] = g
    _session["review_log"] = []
    # 若庄家不是人类, 先推进
    if g.dealer != g.human_seat:
        run_bots_until_human(g)
    return state_with_names(g)


@app.get("/api/state")
def get_state():
    g = current_game()
    return state_with_names(g)


@app.post("/api/discard")
def discard(req: DiscardReq):
    g = current_game()
    _record_decision(g, req.tile)   # 先记录(出牌前分析)
    try:
        g.action_discard(g.human_seat, req.tile)
    except (AssertionError, ValueError) as e:
        raise HTTPException(400, str(e))
    run_bots_until_human(g)
    return state_with_names(g)


@app.post("/api/peng")
def peng():
    g = current_game()
    try:
        g.action_peng(g.human_seat)
    except (AssertionError, ValueError) as e:
        raise HTTPException(400, str(e))
    run_bots_until_human(g)
    return state_with_names(g)


@app.post("/api/gang")
def gang(req: GangReq):
    g = current_game()
    try:
        g.action_gang(g.human_seat, req.tile)
    except (AssertionError, ValueError) as e:
        raise HTTPException(400, str(e))
    run_bots_until_human(g)
    return state_with_names(g)


@app.post("/api/pass")
def pass_action():
    g = current_game()
    try:
        g.action_pass(g.human_seat)
    except (AssertionError, ValueError) as e:
        raise HTTPException(400, str(e))
    run_bots_until_human(g)
    return state_with_names(g)


@app.get("/api/analyze")
def analyze():
    g = current_game()
    az = Analyzer(g, g.human_seat)
    result = {"hand": az.analyze_hand()}
    # 14张状态(轮到我出牌)时给出打出建议
    if g.phase == "discard_wait" and g.turn == g.human_seat:
        result["discards"] = az.analyze_discards()
    return result


@app.get("/api/review")
def review():
    """赛后检讨: 返回我每一手的决策记录与推荐对比"""
    log = _session.get("review_log", [])
    if not log:
        return {"steps": [], "total": 0, "match_rate": 0.0}
    matched = sum(1 for s in log if s["match"])
    return {
        "steps": log,
        "total": len(log),
        "matched": matched,
        "match_rate": round(matched / len(log), 3),
    }


# 静态前端
frontend_dir = os.path.join(os.path.dirname(__file__), "..", "frontend")
frontend_dir = os.path.abspath(frontend_dir)


@app.get("/")
def index():
    return FileResponse(os.path.join(frontend_dir, "index.html"))


app.mount("/static", StaticFiles(directory=frontend_dir), name="static")
