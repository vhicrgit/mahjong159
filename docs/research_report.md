# 麻将 AI 调研报告

## MahJax 深入评估（2026-08-27，8×H20 可用后）

MahJax（arXiv:2605.20577，2025）：纯 JAX 麻将环境，8×A100 简化规则 2M steps/s、
完整红宝牌规则 1M steps/s。附 BC/PPO/并行 rollout/两种观测编码的完整例子。

### 关键工程手法（已读源码确认）
1. **向听查表**: 每花色 9 种牌按 base-5 编码（Σc_i·5^i ≤ 5^9≈195万码），
   预计算 shanten_cache.npz（压缩后仅 767KB）加载到 GPU，JIT 内纯 gather
2. **条件分支消除**: `discard()` 用"算全部 34 候选 + where 掩码"代替 lax.cond
   （避免 XLA 在 vmap 下每 lane 复制 70MiB 表）
3. 全状态机 jitted，vmap 跨局并行

### 移植 159 的成本分析
| 组件 | 难度 | 说明 |
|------|------|------|
| 牌组 28 种(27序数+红中) | 低 | 无字牌，比日麻 34 种更小 |
| 无吃/只能自摸 | 低 | 状态机比日麻**简单**（无吃优先级/振听/立直） |
| 碰/杠/黄庄/159结算 | 低-中 | 规则直译，159 翻牌是向量化友好操作 |
| **红中癞子的向听表** | **高** | 红中破坏"每花色独立"前提。方案：每花色表改为
  (code × 可用红数0-4) → Pareto (m,t,p) 集合，合并时枚举红中分配
  (5×5×5=125 种) × 3 花色 Pareto 合并 —— 表约 175MB uint8, 可驻留 GPU |
| 观测编码 | 中 | 我们的 v2/v3 特征含向听派生量 → 依赖上表；或先用 v1 裸特征 |

### 结论
- **值得做，但排在算法验证之后**: 我们当前 Python 引擎 + vec_selfplay
  的瓶颈在 CPU 端引擎步进与特征编码；GRPO 式训练是 rollout 密集型的，
  若想法2验证有信号，JAX 移植是 50-100x 的吞吐乘子（1M steps/s 量级）
- 移植工作量估计 1-2 天（引擎+红中向听表生成器+观测+单测对拍 Python 版）
- 环境隔离：建独立 .venv-jax（jax[cuda] wheel 自带 CUDA 库，不动系统 torch/CUDA）

## 各项目核心方法

| 项目 | 算法 | 架构 | 关键技巧 |
|------|------|------|----------|
| **Mortal** | DQN+CQL (离线RL) | 1D ResNet, 192ch×40blocks, ~10M | GRP信用分配, Boltzmann探索(ε=0.005,T=0.05), 1v3评估 |
| **Suphx** | Policy Gradient + 熵正则 | Deep CNN (无pooling), 5个网络分工 | GRP全局奖励预测, Oracle引导(先知→渐隐), 44 GPU分布式 |
| **LuckyJ** | ACH (Actor-Critic Hedge) | NN + 搜索树 |  regrets最小化, 乐观价值估计搜索, 无人类数据纯自对弈 |
| **NAGA** | 纯监督学习 | 4个CNN分别决策 | 不用RL, 最高级对局数据训练, 也达到8段 |
| **Mahjax** | PPO + KL散度 | Transformer encoder | JAX GPU向量化, 2M步/秒(8×A100) |

## 对我们的关键启示

### 为什么我们的 PPO/AWR 不工作

1. **Suphx 发现**: 直接用局终分数做 RL 效果差 → 需要 GRP (Global Reward Prediction) 做信用分配
2. **Mortal 发现**: 在线策略(PPO类)在麻将这种高方差游戏中不稳定 → 用 DQN+CQL (离线值函数方法) 更稳定
3. **LuckyJ 发现**: PPO 自对弈不收敛到纳什均衡 → 需要 ACH (Hedge/regret最小化)

### 可行方案 (按优先级)

**方案A: DQN + CQL (Mortal 路线, 最稳)**
- 不直接学策略, 而是学 Q(s,a) 值函数
- CQL 惩罚分布外动作的Q值, 防止过估计
- MC return (gamma=1) 而不是 TD, 避免bootstrap误差
- GRP 网络做信用分配 (把局终分数分配到每个决策点)
- Boltzmann 探索 (很低温度 0.05)
- 评估用 1v3 轮换 (消除位置偏差)

**方案B: PPO + 正确的奖励整形**
- GRP 网络把局终分数分解到每个决策
- 大批量自对弈 (至少 1000+ 局/iter)
- 很低的学习率 (1e-5 ~ 3e-5)
- 熵正则防止过早收敛

### 推荐: 方案A (Mortal 路线)

原因:
1. Mortal 是唯一一个完全开源、代码可验证的强麻将AI
2. DQN+CQL 在高方差环境中比 PPO 更稳定
3. 我们没有人类数据, 但可以用规则Bot生成数据作为"专家示范"
4. 模型可以很小 (~1M), CPU 可跑

## 具体实施计划

1. **改造模型为 DQN**: 输出 Q(s,a) 而不是 policy logits
2. **实现 GRP**: 一个小的 GRU 网络, 预测最终得分, 用于信用分配
3. **BC 热身**: 用规则Bot数据训练初始 Q 函数
4. **CQL 训练**: 保守 Q 学习, 防止分布外动作过估计
5. **自对弈 + MC return**: 用当前 Q 函数生成数据, MC 回报更新
