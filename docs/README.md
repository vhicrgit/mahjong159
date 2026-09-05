# mahjong159 AI 训练 —— 文档索引与当前状态

> **2026-09-05 独立审计后的更正**：下文保留历史快照，不再把“平台期定量闭合”
> 当作当前结论。已复现硬教师标签零策略梯度/假100%命中、评估遗漏自家杠、
> seed内相关性统计问题。代码修复与新实验见
> [修复和推进记录](recovery_progress_20260905.md)，原始证据见
> [独立审计报告](independent_audit_20260905.md)。新评估协议与旧数字不可直接混用。

> **2026-09-05 晚间第二次更正（外部评审落地）**：完整动作协议（四家统一执行
> 暗杠/补杠）下重测，**"NN ≈ 分析器 ≈ v31n 三线打平"作废**：冠军 NN 首胡
> −0.2037 ± 0.0413（t=−4.94）、血战 −0.2469；分析器 −0.076 ± 0.059；v31n 定义
> 为 0。新增规则 Bot **v32**（听牌后选择性碰牌）4096 种子独立复验
> **+0.0215 ± 0.0077（t=+2.80）**，是目前测得最强的非作弊策略。条件方差分解
> 同时否掉了"特权 critic"路线（temp=1 下 55.5% 的回报方差来自策略自身采样，
> 特权 critic 天花板只到 44.5%，advantage 噪声最多降到 1/1.34）。全部数字、
> 评审断言逐条裁决与代码改动见
> [规则Bot与分析器评审落地记录](rule_bot_and_analyzer_review.md)。

> 安康159麻将 AI 训练项目。4人麻将变体：只能自摸胡、可碰杠不能吃，
> 159 翻牌结算（胡牌后牌堆顶翻6张，数1/5/9张数 n，胡牌分 = 3×(n+1)），
> 红中禁碰杠。

最后更新: 2026-09-05（自进化探索收官：四条路线全部定量证伪/中性，
平台期结论闭合，见 opponent_features_and_expert_iter.md）

---

## 当前状态（2026-09-05）

**上线模型**: `models/acnn_latest_best.pt` = **bc_k0_r2**（DAgger 软标签
模仿牌型分析器 E，碰/杠用分析器 E 判据）。roster `acnn` 档，端到端验证过。

**棋力格局**（完整动作协议：四家统一执行碰/明杠/暗杠/补杠；CRN 配对、轮坐
四座位、主指标**真实净分差 分/局** vs v31n；首胡=线上规则）:

| 选手 | 首胡 | 血战到底(仅训练用) | 种子数 |
|---|---|---|---|
| **v32**（v31 + 听牌后选择性碰牌） | **+0.0215 ± 0.0077（t=+2.80）** | —— | 4096 |
| v31n（基线） | 0 | 0 | —— |
| 牌型分析器本体（学者, kai=0） | −0.076 ± 0.059（t=−1.30） | —— | 1500 |
| **bc_k0_r2（上线 NN）+ E 碰杠** | **−0.2037 ± 0.0413（t=−4.94）** | −0.2469 ± 0.0526 | 3000 |
| bc_r2_s3（q×3 校准版） | −0.2013 ± 0.0412（t=−4.89） | −0.2438 ± 0.0526 | 3000 |
| 旧 RL 产物 acnn_v2（已备份，*旧协议*数字） | -0.238 ± 0.042 | -0.573 ± 0.057 | —— |
| bc_k0_r2 vs acnn_v2（h2h，*旧协议*） | +0.169 ± 0.036 (t=4.7) | +0.438 ± 0.051 (t=8.7) | —— |

即：**排序是 v32 > v31n ≳ 分析器 > NN**，此前"三者统计打平"的说法只在漏掉
自家杠的旧协议下成立。学生（NN）比自己的教师（分析器）低约 0.13 分/局，
最大嫌疑是训练数据来自不执行暗杠/补杠的采集器（协变量漂移，待重训验证）。
NN 原有的价值主张（分析器级棋力 + 快 3 个数量级的推理）中"棋力"这一半
需要重新挣回来；推理速度那一半仍然成立。

**评估口径警告**: 2026-08 的旧数字（"v31=33.5% 单座位胜率"等）是单座位
绝对胜率口径；本轮全部结论用 **CRN 配对得分差**口径（同 seed 同座位差分，
降方差 1/(1-ρ)=8~37x，SE 0.03~0.05 分/局）。两套口径不可直接比较。
检出 0.05 分/局的效应需要 ~3000 配对 seed —— 这是本项目的分辨率墙。
另一条已验证的坑：300 seed 下 t>2 的结果必须换种子复核（v4 特征的
+0.106(t=2.4) 在 800 新 seed 下塌缩到 +0.010）。

## 平台期结论（为什么停在这里）

五条独立路线全部试过、全部定量证伪或中性（细节与全部数字见
`opponent_features_and_expert_iter.md`）:

1. **终局奖励 RL**（PPO/A2C/同墙反事实/配对基线/冠军闸门，6+ 配置）:
   从任何起点都退化或中性。逐决策 corr²(adv,ΔE) 仅 0.002~0.017；
   train/deploy 错配（采样 vs 贪心差 -1.13 分/局）用 q×3 校准修复后
   仍只是从"显著有害"变"统计中性"。
2. **对手模型特征 v4**（粒子滤波 tracker 90 维）: h2h 合并 +0.036±0.023，
   不显著。E 标签不含名次/防守信息，架构"看得见对手但没动机用"。
3. **搜索教师 / expert iteration**（同墙反事实 rollout，N=8→64 两轮）:
   交叉验证整批有真信号 +0.154(t=3.9)，但单状态 SE(差) 0.32 > 真实分差
   中位 0.156，置信标签规模不足，两轮累计 -0.018±0.027。翻盘需 ~650
   rolls/候选 ≈ 100x 算力。
4. **开源 AI 蒸馏（Mortal）**: 管线全通（Rust 编译/arena 自对弈/牌谱映射），
   但分歧结构 100% 由规则差异解释（断幺 18.7% + 点炮 fold：现物偏好
   61.5% vs 基准 33.3%），纯数牌一致率仅 32%、分歧 E 代价中位 1.6 巡
   —— 无净可蒸馏知识。
5. **模仿过拟合**: 对分析器拟合更好（DAgger r3）反而棋力更差 ——
   模仿上限 = 分析器，而分析器 ≈ v31n。

根因（定量闭合）: 回报方差 87~90% 是摸牌顺序运气（任何状态特征不可知，
完美 critic 也只能解释 ~10%）；仅自摸无点炮 → 无防守博弈深度；牌效口径内
E 已是精确最优。~~解析 shaping 剥离 n_159~~ 已收回（2026-09-05 用户指正）:
训练奖励从始至终是固定名次表(+3/+1/-1/-3)+0.25×杠分，n_159 只影响线上
真实比分与评估参考列，从不在训练信号里。

**再往前走只有两条路，都需要产品/资源决策**:
算力（教师 rollout ~650/候选 + 10 万级状态，约 100x 当前投入）或
规则（血战到底上线：信号 2.15x + 决策量 1.73x，用户已明确排除）。

## 基础设施（本轮沉淀，全部可复现）

- **血战到底引擎**（`Game(bloody=True)`，仅训练用不上线）: 胡家下场继续
  打到 3 家胡，名次奖励零和；首胡模式 500 局与改动前逐位一致
  （`perf/test_bloody_parity.py`，从 git HEAD 取旧引擎对拍）
- **CRN 配对评估**（`backend/rl/eval_crn.py` + `perf/eval_ckpt*.py`,
  `eval_h2h*.py`）: 同 seed 差分 + 轮坐四座位；v31n vs v31n 自检差分恰为 0
- **采集提速 9.1x**（426→3920 决策/s）: 真瓶颈是 features_v2 里对每个候选
  重算纯 Python 向听 DFS（占采集 86%），换 mj_discard_shanten 后特征
  1930 点逐位一致；引擎本身 16364 决策/s 从来不是瓶颈
- **DAgger 管线**（`tools/dagger_collect*.py` + `dagger_train.py`）:
  网络自身分布采状态 + 分析器 E 软标签（整条 E 向量, τ=0.3, kai=0）+
  多数据集加权混合；修复模仿协变量漂移（top-1 命中: v31n 分布 91.65%
  vs 网络自身分布 52.65% → 各分布一致收敛到 E 损失 0.20 巡）
- **q 校准**（`models/bc_r2_s3.pt`）: q_head×3，argmax 逐点不变（2130 点
  验证）但采样≈贪心（采样损失 -0.373 → -0.025 分/局）；后续 RL 一律
  从它出发
- **冠军闸门**（`tools/champion_loop.py`）: 候选必须 CRN 配对胜过冠军才
  替换；34 次判定全部正确拒绝退化候选，冠军全程零退化
- **对手模型**（`backend/analysis/opp_model.py`，新增 hero_seat 参数支持
  四座位各自跟踪）+ v4 特征（`backend/rl/features_v4.py`，718 维，
  warm-start 与 r2 输出差 1e-6 的严格扩展）
- **Mortal 环境**（`third_party/` + `tools/mortal_selfplay.py` +
  `distill_mortal.py`）: libriichi 编译产物在 /dev/shm/mortal_target
  （home inode 配额已满 40 万文件；JuiceFS 链接写速 150KB/s 不可用，
  crates.io 走 rsproxy 镜像），权重 mortal-298k（AGPL-3.0）

## 文档导航

| 文档 | 内容 |
|------|------|
| **README.md**（本文） | 索引 + 当前状态快照 |
| `opponent_features_and_expert_iter.md` | **2026-09 收官记录**: 四方向全部实验数字、平台期证据链、Mortal 蒸馏机制分析、工具索引 |
| `independent_audit_20260905.md` | **独立审计**: 硬教师标签零策略梯度/假100%命中、N次rollout≠N个隐藏世界、评估漏自家杠、seed内相关性、"理论上限"推断为何不成立 |
| `recovery_progress_20260905.md` | 审计对应的修复清单 + 两组实验裁决（旧教师数据无可靠收益、多世界教师未过独立检验） |
| `rule_bot_and_analyzer_review.md` | **外部评审落地**: 断言逐条裁决、v32 选择性碰牌(+0.0215)、分析器 E 四处修复、完整协议重基线、条件方差分解 |
| `research_report.md` | 业界调研（Mortal/Suphx/LuckyJ/NAGA/Mahjax）与启示 |
| `methods_and_findings.md` | GRP 信用分配 · 三个 RL 退化 bug · 离线 DQN 突破（2026-08 历史） |
| `ceiling_and_bots.md` | 座位效应基线 · Oracle 上界 · 规则Bot v1-v31 · PIMC 失败教训（历史） |
| `self_evolution_roadmap.md` | 自进化路线规划 + 终局批注（Oracle Guiding 路线已被 2026-09 工作取代） |
| `cheat_full_jax_benchmark.md` | JAX 全信息 bot 基准（历史） |

## 模型文件（models/，不入库）

**现役**:
- `acnn_latest_best.pt` = `bc_k0_r2.pt` — 上线 NN（DAgger r2，628 维，
  E 软标签，碰杠走 E 判据）
- `bc_r2_s3.pt` — r2 的 q×3 校准版（部署行为逐点一致；RL 实验应从它出发）

**实验产物（保留作对照）**:
- `bc_k0_soft.pt`(r1) / `bc_k0_r3.pt`(过拟合分析器，棋力反而更差)
- `bc_v4_r1/r2/ei.pt` — v4 对手特征系列（全部中性）
- `ei64_a.pt` / `ei64_b.pt` — 搜索教师两轮（中性偏负）
- `acnn_rl_v2_backup.pt` — 旧 RL 冠军（被 r2 取代，+0.169/+0.438）
- `hv_value_pretrained_v2.pt` — BC92 起点（49% 那份 `hv_value_pretrained.pt`
  已弃用）；`warm_v4.pt` — v4 warm-start

**数据集**: `k0_nn/k0_v31/k0_nn_fw/k0_nn_r2/k0_nn_r3.npz`（DAgger 各轮，
628维+kai0 E向量）、`v4_nn_*/v4_fw_*.npz`（718维）、`teach64*/teachv4_*`
（教师 target+scores 矩阵）、`hv_value_data_all.npz`（原始 BC 93k，kai1）

## 工具速查（tools/）

- 训练: `dagger_collect.py` / `dagger_collect_v4.py` / `dagger_train.py` /
  `search_teacher.py` / `search_teacher_v4.py` / `filter_teacher.py`
- RL（已证伪，留档可复现）: `rl_bloody_train.py` / `rl_paired_train.py` /
  `rl_cf_train.py` / `champion_loop.py` / `rl_ac_train.py`
- 评估: `perf/eval_ckpt.py`（--claim hv/v31, hv0/hv1 分析器本体）/
  `perf/eval_ckpt_v4.py` / `perf/eval_h2h.py` / `perf/eval_h2h_v4.py`
- 诊断: `perf/diag_crn.py`（降方差律 1/(1-ρ)）/ `diag_bloody.py`（奖励
  R²）/ `diag_luck_vs_skill.py`（牌运分解）/ `diag_cf_snr.py` /
  `diag_ac_signal.py` / `diag_duplicate.py`（复式轮转证伪）/
  `probe_q_scale.py`（采样损失）/ `bench_kai_depth.py`（换型档位代价）
- 回归: `perf/test_bloody_parity.py` / `perf/test_features_parity.py`
- Mortal: `mortal_selfplay.py`（arena 自对弈产谱）/ `distill_mortal.py`
  （解析/映射/compare/distill）

## 历史存档（2026-08，旧口径，仅供相对参考）

- 判胡/向听 4 个长期 bug 修复（红中补低位顺子漏胡 2.7%、8/9 搭子不计、
  DFS 贪心丢最优、对子不当搭子），v31 副露向听修复 → 规则 bot 从
  "从不碰杠"（400 局 169 次机会 0 次碰）到 33.5% 单座位胜率
- 出牌侧饱和: 胜率 DP 精修（v28-v30）与 v10 无差异（分歧率 2%）
- 离线 DQN+CQL 28%、Oracle 上界 49.8%、作弊 full 82.5%、
  large 43M 无增益（容量非瓶颈）
- 详见 ceiling_and_bots.md / methods_and_findings.md
