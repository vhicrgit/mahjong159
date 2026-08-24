# 安康159麻将

一个基于 Web 的安康地方玩法"159门清麻将"人机对战游戏,内置规则引擎、AI 对手、实时牌局分析器和赛后检讨功能,并提供完整的深度强化学习训练管线(可自训练更强的神经网络 AI)。

## 玩法规则(安康159)

- 牌组:条、饼、万 1~9 各 4 张 + 红中 4 张,共 112 张
- 胡牌:只能自摸(含杠上花),不能点炮;可碰、可杠,不能吃
- 杠分独立结算(非黄庄必结):放杠者赔杠牌人 3 分;暗杠/补杠时其他三家各赔 1 分
- 黄庄:轮到某家抓牌时,牌堆剩余 <= 6 张,立即黄庄(本局作废,杠分不结)
- 159 结算:胡牌后从牌堆按序翻 6 张,统计其中 1/5/9 的张数 n(不分花色),输家每人赔赢家 n+1 分
- 红中(癞子):和牌时可当任意一张牌;不能用于碰/杠;平时可当普通牌打出
- 听牌后可以换口
- 无番种,胜负只看胡牌底分 + 159 翻牌

## 功能特性

- 完整规则引擎:胡牌判断(支持红中癞子组合优化)、向听数/听口分析、碰杠胡、黄庄判定、159 结算
- AI 对手:规则引擎(牌效+放杠防守)或自训练的神经网络模型
- 实时分析(Akagi 式):出牌推荐高亮、向听数、听口、放杠风险、159 预期收益、对手威胁度
- 赛后检讨(mjai 式):逐手回放你的出牌 vs AI 推荐,标出不一致的打法并给出当时候选与风险
- 强化学习管线:BC 热身 + AWR + 在线迭代自对弈,模型规模可配置(tiny/small/base),支持多卡/GPU

## 项目架构

```
mahjong159/
├── backend/
│   ├── main.py                 # FastAPI 接口(对局、分析、复盘)
│   ├── rules/                  # 规则库(纯算法)
│   │   ├── tiles.py            #   牌定义(112张)
│   │   ├── win.py              #   胡牌判断+向听数(支持红中, DFS+缓存)
│   │   └── ting.py             #   听口/进张分析
│   ├── game/
│   │   └── engine.py           # 游戏引擎状态机(摸打出碰杠胡/黄庄/结算)
│   ├── analysis/
│   │   └── analyzer.py         # 实时分析器(放杠风险/收益预估/威胁度)
│   ├── ai/
│   │   └── bot.py              # 规则AI对手
│   └── rl/                     # 强化学习(模仿 Mortal 架构)
│       ├── features.py         #   局面特征编码(512维)
│       ├── model.py            #   ResNet+policy/value双头(大小可配)
│       ├── selfplay.py         #   自对弈数据生成
│       ├── train.py            #   BC热身+AWR
│       ├── iter_train.py       #   在线迭代训练(数据->训练->评估)
│       ├── net_bot.py          #   神经网络Bot
│       └── evaluate.py         #   模型 vs 规则Bot 评估
├── frontend/                   # 前端(原生 HTML/CSS/JS, 零构建)
│   ├── index.html
│   ├── css/style.css
│   └── js/game.js
└── README.md
```

技术栈:Python 3.10+ / FastAPI / PyTorch / 原生 HTML+CSS+JS

## 快速开始

```bash
cd mahjong159

# 创建虚拟环境并安装依赖
python3.12 -m venv .venv
.venv/bin/pip install fastapi uvicorn torch numpy

# 启动服务
.venv/bin/python -m uvicorn backend.main:app --host 127.0.0.1 --port 8159
```

浏览器打开 http://127.0.0.1:8159 即可游玩。

## 使用说明

- 你固定坐庄下(座位 0),点两次手牌打出(第一次选中,第二次确认)
- 有碰/杠机会时中间弹出按钮;摸牌后可暗杠/补杠时手牌上方出现可选杠牌
- 出牌阶段,分析器最优推荐的牌会在手牌上金色高亮(标"荐")
- 右上角"分析"打开实时分析面板;"复盘"打开赛后检讨面板

## AI 训练

### 快速验证(几分钟)

```bash
# 小规模闭环验证: 300局自对弈 + BC热身 + AWR
.venv/bin/python -m backend.rl.train --games 300 --size tiny --out model_tiny.pt

# 评估模型 vs 规则Bot
.venv/bin/python -m backend.rl.evaluate --model model_tiny.pt --games 200
```

### 在线迭代训练(推荐)

数据 -> 训练 -> 用新模型再采样 -> 再训练,循环提升:

```bash
.venv/bin/python -m backend.rl.iter_train \
    --size tiny --iters 10 --games-per-iter 1000 \
    --out model_iter.pt
```

### 正式训练(GPU/多卡,如 8xH20)

```bash
.venv/bin/python -m backend.rl.iter_train \
    --size base --iters 100 --games-per-iter 5000 \
    --buffer 500000 --out model_base.pt
```

模型规模三档:`tiny`(0.21M 参数,调试用)、`small`、`base`(接近 Mortal 量级,正式训练用)。

### 使用训练好的模型作为 AI 对手

模型保存后,可将游戏中的规则 Bot 替换为 `backend/rl/net_bot.py` 的 NetBot,或在复盘中把推荐源换成神经网络模型。

## 参考

- Mortal (github.com/Equim-chan/Mortal):日麻 AI,本项目 RL 架构参考
- Akagi / mjai:实时对局分析与赛后检讨的产品形态参考
- Suphx (arXiv:2003.13590):麻将深度强化学习奠基论文

## 协议

仅供学习研究使用。
