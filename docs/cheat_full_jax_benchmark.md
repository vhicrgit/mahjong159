# cheat_full_jax: 用 JAX 加速 cheat_full 的尝试记录

日期: 2026-08-28
状态: 原型完成, 已基准, 结论明确

## 目标
用 JAX(GPU) 加速 cheat_full(神挂) 的出牌决策。cheat_full 慢在每步
beam search + Python rollout(`_rollout_to_end`, 直线模拟到终局)。

## 实现(bot_cheat_jax.py)
- 对全部合法候选**一次批量**构建共享世界 State
  (`fast_inject.build_world_states`)
- 用 `rollout._rollout_jit`(全 jit while_loop, NN 决策) 在 GPU 上推演到底
- 取 n_worlds 世界均值塑形回报, 选最优
- 关键优化: 候选 pad 到 MAX_CANDS=16, 世界数固定 32 -> 输入形状恒定,
  XLA 只编译一次(~142s), 之后每次决策 ~3.4s

## 基准(10 局面, 对比原版 BotCheat 满配)
| 指标 | 原版 cheat_full | cheat_full_jax |
|---|---|---|
| 平均耗时 | 14.6s | 3.5s (快 4 倍) |
| 最慢局面 | 44.2s | 4.2s (快 10 倍) |
| 简单局面 | 0.15s | 3.5s (反而慢) |
| 选牌一致率 | - | 2/10 (20%) |

## 结论
1. **速度**: 复杂局面(原版 >5s)下 cheat_full_jax 快 4-10 倍;
   简单局面(原版 <1s)反而慢 20 倍。适合"决策昂贵"的场景。
2. **正确性**: 选牌一致率仅 20%。原因是决策口径不同:
   原版 = beam search + 启发式 + BotV1 对手(前瞻搜索);
   jax = NN(q_forward) 直线推演(无搜索)。两者是不同 AI,
   不是"同一个 cheat_full 的加速版"。
3. **训练集成不可行**: choose_discard_jax 每次 ~3.4s, 若嵌入
   rollout_jax 每步(每局 ~40 步 hero 决策), 单局推演 ~140s,
   无法用于 GRPO 训练推演。
4. **保持原版决策质量的加速路径**(未做, 工程量大):
   - 把 bot_cheat 的 `_rollout_to_end`(depth=0 直线模拟) 批量展开,
     一次 GPU 推演评估所有树节点 —— 但推演决策仍需启发式/BotV1,
     _rollout_jit 是 NN 决策, 口径不符;
   - 或训练 NN 模仿 cheat_full 的决策(行为克隆), 再用 NN 推演。

## 遗留
- cheat_full_jax 已注册(_BOT_REGISTRY), --rollout-mode 可用,
  但实际走 nn 模式(NN 推演), choose_discard_jax 未嵌入推演循环。
- 若要让 cheat_full_jax 真正"决策", 需要独立调用
  `choose_discard_jax(game, seat, net, ...)`, 如离线局面评估。
