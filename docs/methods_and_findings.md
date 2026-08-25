# 方法与工程发现（RL 训练全过程）

日期: 2026-08-24 ~ 2026-08-25
本文合并原 grp_experiment / dqn_breakthrough / training_speedup 三份记录。
覆盖: GRP 信用分配尝试 → 三个 RL 退化 bug → 离线 DQN 突破 → 训练提速。

---

## 一、GRP（Global Reward Prediction）信用分配

动机: 早期所有 RL（AWR/PPO/DQN+CQL）退化，怀疑局终奖励稀疏+噪声大导致
信用分配失败。Suphx/Mortal 都用 GRP（从中间状态预测最终得分）缓解。

### 得分方差结构（50k 局规则Bot自对弈, 141.5万决策点）
决策者本人得分 var=26.5。方差分解:
- **~85% 来自"谁赢"**（可从局面预测）
- **~15% 来自 159 翻牌随机性**（胡牌后牌堆顶翻6张，不可预测）
- 流局率 6.3%

**修正了一个早期误判**: "159 随机性淹没信用分配"不准确——主导噪声其实是
输赢不确定性，而这正是 GRP 能压缩的部分。GRP 理论 R² 上限本应很高。

### GRP 模型迭代
- **v1**（MLP 0.29M）: R²=0.064，训练≈验证 MSE → 欠拟合（非过拟合）
- **v2**（ResNet 2.9M）: R²=0.154
- **v3**（相对目标 + 杠分特征 + 胜者分类头, 3.9M, 200k局）: R²=0.163
  - 关键设计: 输出改为相对位置 [我,下家,对家,上家] 得分（原绝对座位目标
    与相对视角特征错配，信号被 4 路稀释）；加相对杠分 5 维；胜者分类辅助头
  - 结论: **不完美信息下 GRP 可预测性存在硬上限**（对手手牌不可见），
    R² 提不上去，势函数奖励信号偏弱

### 势函数形式的 GRP 奖励（数学正确但实际增益有限）
r_t = GRP(s_{t+1})[me] − GRP(s_t)[me]（中间步）；r_T = 真实得分 − GRP(s_T)[me]
- Ng et al. 1999 势函数整形: Σr_t = 真实得分 − GRP(s_0)，**最优策略不变**
- GRP R²=0.16 时差分信号太弱，PPO 用它仍停在基线

---

## 二、三个 RL 退化 bug（真正的根因）

**历史上 AWR/PPO/DQN+CQL 全军覆没，主因是这三个工程 bug，而非算法或信用分配。**

### bug1: log_prob 分布不匹配（vec_selfplay.py）
采样用温度化分布 p^(1/T)，但记录的 log_prob 取自原始 softmax。
→ PPO ratio 系统性偏大 → 正优势被 clip 截断、负优势过度压制 → 熵膨胀+退化。
修复: log_prob 取自实际采样分布；T≤0 走 argmax 贪心（评估路径）。

### bug2: PPO 更新未施加合法动作掩码（ppo.py）
更新时对原始 logits 做 log_softmax，未 mask 非法动作。
→ 熵奖励把概率质量泄漏到"永不采样=永不惩罚"的非法动作 → 熵单调膨胀 →
合法分布被拉平 → 行为随机化。
修复: 每条记录存 legal mask，更新时施加相同掩码，熵在合法分布上算 + clip 率诊断。

### bug3: value loss 梯度支配共享主干（ppo.py）
得分 scale ±20 的 MSE value loss vs 已归一化到 ±1 的 advantage。
实测: value loss 对共享主干的梯度是 policy loss 的 **26.6 倍**（value_coef=0.5 时仍 13.3x）
→ value 头拖着主干漂移，policy 头被拉离最优。
修复: value target 归一化 + huber loss，value_coef 0.5→0.1，value 预训练同步归一化。

**验证**: 三个 bug 全修后 PPO 不再退化（熵稳定 ~0.03，clip 率 0.01-0.07），
但也只是稳在基线——PPO 路线最终未突破（见下，被离线 DQN 取代）。

**教训: "RL 不 work"先查工程正确性（ratio/掩码/梯度尺度），再怪算法。**

---

## 三、离线 DQN+CQL 突破（Mortal 路线）

修好 bug 后发现历史 DQN 同样被 bug2/bug3 毁掉。用修复版离线 DQN 重跑:

- 数据: 规则Bot自对弈, 含动作标签（`gen_offline.py`）
- 训练（`dqn_offline.py`）: Q(s,a)→归一化 MC return，CQL 只在合法动作上
  logsumexp，target network，bf16
- **结果: 24.5%(BC) → 27.0%(20ep) → 28.1%(60ep)**，稳定超越 BC

原理: Q-greedy 隐式超越行为策略——目标是 MC return（绝对分），
CQL 压低未见动作 Q，学到的 argmax 排序可优于规则Bot 启发式。

### 数据量/超参 sweep 结论
- CQL 权重 0.1-2.0 胜率不敏感（26.7-27.3%）—— 弱教师下模仿强度无所谓
  （注: **CQL 项数学上等价于对教师动作的 BC 交叉熵**，见 ceiling_and_bots.md）
- 数据 200k→600k、模型 base→large(43M) 均无增益 → **瓶颈是教师质量，非容量/数据量**
- v5（Bot v2 数据 600k）: 28.2% —— 换更强数据源收益也微小

**注意**: 以上 28% 曾被判读为"突破基线"，后经天花板分析修正——
真实基线是座位0庄家的 **27.4%**（非 25%），28.2% 实为 +0.8 点（约 1σ）。
详见 `ceiling_and_bots.md`。

### 失败的分支（记录以免重走）
- **迭代自提升**（当前策略自对弈+规则Bot数据混合再训）: 无增益，
  原因是每轮欠训练(12ep) + 掺规则Bot数据稀释 + 探索不足
- **在线 PPO 精调**（从 DQN 种子）: 停在基线，GRP 奖励信号太弱

---

## 四、训练提速（用户反馈 CPU 闲置后完成）

### 根因诊断
1. **`nproc` 误报 23 核**: 环境变量 `OMP_NUM_THREADS=23` 污染 nproc 输出，
   真实可用 **184 核**（用 `len(os.sched_getaffinity(0))` 判断）。
   之前多进程任务只用了 1/8 CPU。
2. **GPU batch 太小**: fp32/bs2048 被 kernel launch 开销限制（98% util 但吞吐低）。
3. **评估单进程**: 2000 局 ~3min（CPU encode 瓶颈）。

### bf16 基准（H20, base 6.77M）
| batch | fp32 | bf16 | 加速 |
|-------|------|------|------|
| 2048 | 42万样本/s | 45万 | 1.1x |
| 8192 | 51万 | 162万 | 3.2x |
| 16384 | 53万 | **200万** | 3.8x |

### 已实施
| 项目 | 之前 | 现在 | 加速 |
|------|------|------|------|
| DQN 训练 | fp32/bs2048 | bf16/bs16384 autocast | 4.8x |
| 数据生成 400k 局 | 12 workers ~80min | 160 workers 13.6min | ~6x |
| 评估 2000 局 | 单进程 ~3min | spawn 8 workers 52s | 3.5x |
| 实验吞吐 | 单卡串行 | 6-8 GPU 并行 sweep | 6-8x |

另有一个全局收益: **shanten 的 DFS 缓存 bug 修复**（原为函数内闭包+lru_cache，
每次调用重建导致缓存永不命中）。提为模块级 `_dfs_cached` 后冷查询 353μs→2μs
（175x），特征编码/GRP/评估/搜索全部受益。

### 代码改动
- `dqn_offline.py`: batch 16384 + bf16 autocast（loss 转 fp32）+ 分段中间评估
- `gen_offline.py`: workers 支持到 160；`--bot-version` 选择陪练 Bot
- `eval_parallel.py`（新）: spawn 多进程评估
- `backend/rules/win.py`: `_dfs_cached` 模块级缓存
