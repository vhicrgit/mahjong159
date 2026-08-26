# mahjong159 AI 训练 —— 文档索引与当前状态

> 安康159麻将 AI 训练项目。4人麻将变体：只能自摸胡、可碰杠不能吃，
> 159 翻牌结算（胡牌后牌堆顶翻6张，数1/5/9张数 n，胡牌分 = 3×(n+1)）。

最后更新: 2026-08-26（晚，bug审计：分析器副露向听修复 + RL管线钉回v1口径）

---

## 当前最优与关键数字（干净大样本 3000 局）

| 指标 | 值 | 备注 |
|------|-----|------|
| **正确基线**（v1 规则Bot @座位0） | **~27%** | 非 25%；座位效应见下 |
| **神经网络最优 dqn_off_v5**（离线DQN base 6.8M） | **28.0%** | 当前交付模型 |
| v3 MC搜索 / v4 混合搜索 | 27.4% / 27.9% | **与NN无异**（此前31%是小样本假象） |
| large 43M | 28.3% | 参数×6.4 无增益→容量非瓶颈 |
| v10 广义进张规则 | 31-32% | 前最佳普通Bot |
| **v31 副露向听修复（当前最佳, bot.py 默认）** | **33.5% ±1.7%** | v10+2.4点, McNemar p<0.002 两个seed段复验 |
| Oracle 上界（作弊，已知未来摸牌） | 49.8% | cheat_full（全信息）82.5% |

**★ 2026-08-26 重大修正（副露向听bug）**: `shanten()` 公式硬编码13张手牌，
副露后暗牌变短时向听被高估 ~2×副露数 → **v1-v30 所有规则Bot 其实从不碰杠**
（400局169次碰牌机会0次碰）。修复（`shanten_with_melds`）后 v31 = v10+2.2~2.6
点。此前"159碰杠无价值"的结论是测量假象。详见 ceiling_and_bots.md 第十节。

**★ 出牌侧已饱和（同日证实）**: 用启发式偏置引导的"P(k步内自摸)"精确DP
（v28/v29/v30 三种实现，配对评估 500-3000 局）全部与 v10 无差异——同向听
候选内胜率排序≈进张排序（分歧率仅2%，分歧处五五开）。出牌启发式不必再投入。

**★ 核心结论**: 非作弊 NN 卡在 27-28%。根因是 **159 只能自摸胡、无点炮
→ 无防守博弈深度 → 纯牌效竞速 → 盲打策略空间窄**。Oracle 的 50% 几乎全来自
"作弊预知自己何时成牌"（纯运气信息，不可迁移）。规则Bot修复碰杠后达 33.5-36%，
盲打上限相应上修至 **~35-38%**。NN 侧陪练数据缺副露局面（陪练Bot从不碰杠），
换 v31 重生成数据可能抬高 NN 天花板。

**座位效应**: 引擎 dealer=0 硬编码，庄家(座位0)比末位(座位3)高 +6.1 点，
所有单座位评估都在座位0。公平口径应四座位轮转。

**★ small CPU 交付模型（完成）**: `dqn_small_v1_best.pt`，1.25M 参数，
CPU 推理 0.84ms/次。同口径3000局对比：small 26.1% ≈ base v5 26.4%
（差0.3点在噪声内）—— 再次印证容量非瓶颈，small 完全满足用户"纯CPU~1M参数"要求。
（注: 绝对胜率随评估 seed 在 26-28% 波动，同 seed 下 small≈base 是可靠结论。）

---

## 文档导航

| 文档 | 内容 |
|------|------|
| **README.md**（本文） | 索引 + 当前状态快照 |
| `research_report.md` | 业界调研（Mortal/Suphx/LuckyJ/NAGA/Mahjax）与启示 |
| `methods_and_findings.md` | GRP 信用分配 · 三个 RL 退化 bug · 离线 DQN 突破 · 训练提速 |
| `ceiling_and_bots.md` | 正确基线（座位效应）· Oracle 上界 · 规则Bot v1-v4 · PIMC 失败教训 |
| `self_evolution_roadmap.md` | 能否超越 v4 · AlphaZero 式搜索⇄学习迭代循环规划 |

---

## 模型/数据文件（models/）

- `bc_base_50k.pt` — 行为克隆基线（24.5%，弱于规则Bot）
- `dqn_off_v5.pt` — 离线 DQN 最优（28.2%），base 6.8M
- `dqn_large_v1.pt` — large 43M（28.2%，验证容量非瓶颈）
- `grp_v3.pt` — GRP 模型（R²=0.163，势函数奖励用）
- `offline_v2_600k.npz` — Bot v2 自对弈 1700万样本
- `offline_v4_100k.npz` — Bot v4 教师数据（生成中，AlphaZero 第一轮）

## 规则/搜索 Bot（backend/ai/）

- `bot_v1.py` — 原始规则Bot（评估基线，勿改）
- `bot_v2.py` — 阶段切换+防守强化
- `bot_v3.py` — 蒙特卡罗出牌
- `bot_v4.py` — 解析骨架+搜索精修（大样本复测 27.9%）
- `bot_v10.py` — 广义进张规则（31-32%）
- `bot_v28.py`/`bot_v29.py`/`bot_v30.py` — 胜率DP精修实验（全部与 v10 无差异，已证伪）
- `bot_v31.py` — **当前最强普通Bot 33.5%**（副露感知向听，`bot.py` 默认）
- `bot_oracle.py` — 完美信息上界（作弊，测天花板）
- `bot_cheat.py` — 分级作弊Bot（cheat_wall/opp/full, 最强 82.5%）
- `bot_pimc.py` — 纯 PIMC（失败案例，保留作参考）
- `bot_eval.py` — 单座位强度评估（1 测试Bot + 3×v1）
- `bot_battle.py` — Bot 间 2v2 对战（消除座位偏差）
- `tools/paired_eval.py` — 同seed配对评估（McNemar+决策分歧率，高分辨力）

## 核心训练脚本（backend/rl/）

- `gen_offline.py` — 离线数据生成（`--bot-version` 选陪练，160 workers）
- `dqn_offline.py` — 离线 DQN+CQL（bf16/bs16384，`--mid-eval-every`）
- `eval_parallel.py` — 多进程评估
- `grp_train3.py` / `grp.py` — GRP 模型
- `model.py` — 网络（tiny/small/base/large）

---

## 下一步（详见 self_evolution_roadmap.md）

进行中: v4 教师数据 → 蒸馏神经网络（AlphaZero 第一轮）。
之后: NN 引导搜索 → 更强教师 → 再蒸馏，迭代放大。
