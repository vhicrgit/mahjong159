/* 安康159 - 游戏引擎(JS版)
 * 翻译自 backend/game/engine.py
 * 玩家座位0为人类, 1-3为AI
 */

class Player {
  constructor(seat) {
    this.seat = seat;
    this.hand = [];
    this.melds = [];
    this.discards = [];
    this.score_delta = 0;
  }
  handCounts() { return countsFromTiles(this.hand); }
}

class Game {
  constructor(humanSeat = 0) {
    this.wall = buildWall();
    // 洗牌
    for (let i = this.wall.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [this.wall[i], this.wall[j]] = [this.wall[j], this.wall[i]];
    }
    this.dealer = 0;
    this.players = [0, 1, 2, 3].map(i => new Player(i));
    this.humanSeat = humanSeat;
    this.turn = this.dealer;
    this.phase = "init";  // init/discard_wait/react_wait/game_over
    this.lastDiscard = null;
    this.lastDiscarder = null;
    this.pendingActions = {};
    this.winner = null;
    this.winTile = null;
    this.winKind = null;
    this.fan159 = [];
    this.n159 = 0;
    this.huangzhuang = false;
    this.gangRecords = [];
    this.log = [];
    this.reviewLog = [];  // 复盘记录
    this.lastDrawn = null;  // {seat, tile} 刚摸的牌
    this.lastAction = "";  // 最近动作描述(用于状态栏)
    this._deal();
  }

  _deal() {
    for (const p of this.players) {
      p.hand = this.wall.splice(0, 13).sort((a, b) => a - b);
    }
    const d = this.players[this.dealer];
    d.hand.push(this.wall.shift());
    d.hand.sort((a, b) => a - b);
    this.turn = this.dealer;
    this.phase = "discard_wait";
    this.log.push(`发牌完成, 庄家: 座位${this.dealer}`);
  }

  wallRemaining() { return this.wall.length; }

  checkHuangzhuang() {
    if (this.wall.length <= 6) {
      this.huangzhuang = true;
      this.phase = "game_over";
      this.log.push("牌堆剩余<=6, 黄庄");
      return true;
    }
    return false;
  }

  discard(seat, tile) {
    if (this.phase !== "discard_wait" || this.turn !== seat) throw new Error("非法时机");
    const p = this.players[seat];
    const idx = p.hand.indexOf(tile);
    if (idx < 0) throw new Error("手里没有这张牌");
    p.hand.splice(idx, 1);
    p.discards.push(tile);
    this.lastDiscard = tile;
    this.lastDiscarder = seat;
    this.lastDrawn = null;  // 出牌后清除摸牌标记
    this.lastAction = `座位${seat} 打出 ${tileName(tile)}`;
    this.log.push(`座位${seat} 打出 ${tileName(tile)}`);

    this.pendingActions = {};
    if (tile !== RED) {
      for (let o = 0; o < 4; o++) {
        if (o === seat) continue;
        const cnt = this.players[o].hand.filter(t => t === tile).length;
        const canPeng = cnt >= 2, canGang = cnt >= 3;
        if (canPeng || canGang) this.pendingActions[o] = { peng: canPeng, gang: canGang };
      }
    }
    if (Object.keys(this.pendingActions).length > 0) {
      this.phase = "react_wait";
      return { event: "react", pending: Object.keys(this.pendingActions).map(Number) };
    }
    return this._nextDraw();
  }

  pass(seat) {
    if (this.phase !== "react_wait") throw new Error("非法时机");
    delete this.pendingActions[seat];
    if (Object.keys(this.pendingActions).length === 0) return this._nextDraw();
    return { event: "react", pending: Object.keys(this.pendingActions).map(Number) };
  }

  peng(seat) {
    if (this.phase !== "react_wait" || !this.pendingActions[seat]) throw new Error("不能碰");
    const tile = this.lastDiscard;
    const p = this.players[seat];
    for (let i = 0; i < 2; i++) p.hand.splice(p.hand.indexOf(tile), 1);
    p.melds.push({ type: "peng", tile });
    const dd = this.players[this.lastDiscarder].discards;
    if (dd.length && dd[dd.length - 1] === tile) dd.pop();
    this.pendingActions = {};
    this.turn = seat;
    this.phase = "discard_wait";
    this.log.push(`座位${seat} 碰 ${tileName(tile)}`);
    return { event: "peng", seat, tile };
  }

  gang(seat, tile = null) {
    const p = this.players[seat];
    if (this.phase === "react_wait" && this.pendingActions[seat] && this.pendingActions[seat].gang) {
      // 明杠
      const t = this.lastDiscard;
      for (let i = 0; i < 3; i++) p.hand.splice(p.hand.indexOf(t), 1);
      p.melds.push({ type: "gang", tile: t, kind: "ming" });
      const dd = this.players[this.lastDiscarder].discards;
      if (dd.length && dd[dd.length - 1] === t) dd.pop();
      this.gangRecords.push({ seat, kind: "ming", tile: t, from: this.lastDiscarder });
      this.log.push(`座位${seat} 明杠 ${tileName(t)}`);
      tile = t;
    } else if (this.phase === "discard_wait" && this.turn === seat) {
      if (tile === null) throw new Error("需指定杠牌");
      if (tile === RED) throw new Error("红中不能杠");
      const cnt = p.hand.filter(t => t === tile).length;
      if (cnt === 4) {
        for (let i = 0; i < 4; i++) p.hand.splice(p.hand.indexOf(tile), 1);
        p.melds.push({ type: "gang", tile, kind: "an" });
        this.gangRecords.push({ seat, kind: "an", tile });
        this.log.push(`座位${seat} 暗杠 ${tileName(tile)}`);
      } else if (cnt === 1 && p.melds.some(m => m.type === "peng" && m.tile === tile)) {
        p.hand.splice(p.hand.indexOf(tile), 1);
        for (const m of p.melds) {
          if (m.type === "peng" && m.tile === tile) { m.type = "gang"; m.kind = "bu"; break; }
        }
        this.gangRecords.push({ seat, kind: "bu", tile });
        this.log.push(`座位${seat} 补杠 ${tileName(tile)}`);
      } else {
        throw new Error("不满足杠的条件");
      }
    } else {
      throw new Error("非法杠时机");
    }
    this.pendingActions = {};
    this.turn = seat;
    return this._drawAfterGang(seat);
  }

  _drawAfterGang(seat) {
    if (this.wall.length === 0) {
      this.huangzhuang = true;
      this.phase = "game_over";
      return { event: "huangzhuang" };
    }
    const tile = this.wall.pop();  // 尾部补牌
    const p = this.players[seat];
    p.hand.push(tile);
    p.hand.sort((a, b) => a - b);
    this.lastDrawn = { seat, tile };
    this.lastAction = `座位${seat} 杠后补牌`;
    this.log.push(`座位${seat} 杠后补牌 ${tileName(tile)}`);
    if (isWin(p.handCounts())) return this._hu(seat, tile, "gangshang");
    this.phase = "discard_wait";
    this.turn = seat;
    return { event: "gang_draw", seat, tile };
  }

  _nextDraw() {
    this.turn = (this.lastDiscarder + 1) % 4;
    if (this.checkHuangzhuang()) return { event: "huangzhuang" };
    const tile = this.wall.shift();
    const p = this.players[this.turn];
    p.hand.push(tile);
    p.hand.sort((a, b) => a - b);
    this.lastDrawn = { seat: this.turn, tile };
    this.lastAction = `座位${this.turn} 摸牌`;
    this.log.push(`座位${this.turn} 摸牌 ${tileName(tile)}`);
    if (isWin(p.handCounts())) return this._hu(this.turn, tile, "zimo");
    this.phase = "discard_wait";
    return { event: "draw", seat: this.turn, tile, gang_options: this.gangOptions(this.turn) };
  }

  gangOptions(seat) {
    const p = this.players[seat];
    const counts = p.handCounts();
    const opts = [];
    for (let t = 0; t < 27; t++) if (counts[t] === 4) opts.push(t);
    for (const m of p.melds) {
      if (m.type === "peng" && counts[m.tile] >= 1) opts.push(m.tile);
    }
    return [...new Set(opts)].sort((a, b) => a - b);
  }

  _hu(seat, tile, kind) {
    this.winner = seat;
    this.winTile = tile;
    this.winKind = kind;
    this.fan159 = [];
    let n = 0;
    if (this.wall.length >= 6) {
      this.fan159 = this.wall.slice(0, 6);
      n = this.fan159.filter(is159).length;
    }
    this.n159 = n;
    this._settle();
    this.phase = "game_over";
    this.log.push(`座位${seat} 胡牌(${kind}), n159=${n}`);
    return { event: "hu", seat, kind, fan_159: this.fan159, n_159: n };
  }

  _settle() {
    for (const p of this.players) p.score_delta = 0;
    for (const rec of this.gangRecords) {
      const s = rec.seat;
      if (rec.kind === "ming") {
        this.players[rec.from].score_delta -= 3;
        this.players[s].score_delta += 3;
      } else {
        for (let o = 0; o < 4; o++) {
          if (o !== s) {
            this.players[o].score_delta -= 1;
            this.players[s].score_delta += 1;
          }
        }
      }
    }
    if (this.winner !== null) {
      const per = this.n159 + 1;
      for (let o = 0; o < 4; o++) {
        if (o !== this.winner) {
          this.players[o].score_delta -= per;
          this.players[this.winner].score_delta += per;
        }
      }
    }
  }
}
