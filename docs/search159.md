# 159 新算法实验实现

本分支增加两个独立算法入口，默认对手阵容保持原配置。网页对手菜单可选择“搜索（实验，较慢）”或“有限期（实验）”。实现通过正确性测试不代表已经证明强于 v31；对战结论以冻结版本的独立评测为准。

实测结果见[本地实现与实测](search159_results.md)。网页实验入口采用已测试配置：历史树16训练+16确认、3个弃牌候选、1.96倍标准误门槛、弃牌向听不超过1时搜索；有限期K3、折扣0.7。`bot_param`可覆盖训练世界数或有限期限。

## 有限时域牌型决策

`backend/analysis/finite_horizon.py` 提供 `solve` 与 `rank_discards`，C 内核在 `backend/native/finite_horizon.c`。输入为自己暗手、整数未知牌池、未来摸牌次数 K。每次摸牌都从牌池扣除，包括没有改善向听的摸牌；随后在全部合法弃牌中重新选择最优动作。支持红中和副露后的短暗手。

不加折扣时求本模型中 K 次摸牌内的最优完成概率；折扣 δ 小于 1 时求 `E[δ^(T-1) · I(T≤K)]`，鼓励早完成。该值不是四人实战胜率，也不是原有分析器的期望胡牌巡数 E。这里没有对手抢先胡牌、实际轮转和碰杠。

支持节点预算并返回上下界、是否精确、节点数和缓存命中数。预算耗尽后的 `value` 是下界，不能把上下界重叠的候选直接当作精确排序。有限期 Bot 在无法确认最优候选时回退 v31；全为零或并列时保留 v31。Bot 用 `(wall_length - 6) // 4` 估计名义剩余摸牌次数，这是忽略未来鸣牌和提前结束的抽象。

```python
from backend.analysis.finite_horizon import rank_discards
rows = rank_discards(hand_counts, unseen_counts, 3, discount=0.7,
                     max_nodes=100_000)
```

## 可见信息决策树

`backend/ai/search159/planner.py` 在模拟中继续优化未来自己的决策。它不是仅比较当前弃牌、之后固定规则打法的根节点 rollout；根节点 rollout 仍作为 `mode="root"` 对照保留。

1. 输入只接受白名单 `Observation`，包含自己手牌、公共牌和合法动作。未知对手手牌与墙联合抽样，严格守恒每种牌的数量。
2. 根节点候选使用同一批模拟世界；候选包含碰、杠、不碰以及截断后的弃牌集，始终保留基准动作。`candidate_limit=28` 可保留全部弃牌。
3. 在未来自己的决策处建立树节点，用累计回报选择动作。默认节点键包含可见行动历史和自己的观察，不把隐藏手牌或实际墙序放入决策键。
4. 树训练完成后，用另一批世界评估冻结的未来策略。决策依据完整四人净分，包含碰、明杠、暗杠、补杠、159 翻牌及荒庄取消杠分。
5. `confidence_z` 要求相对于基准的成对均值差超过若干倍标准误。这个门槛只控制模拟噪声，不是多候选选择后的置信保证，也不覆盖信念模型误差。

模拟中的 159 分数使用对未查看墙序的翻牌条件期望，减少随机翻牌噪声；外层比赛仍使用 `Game.score_delta` 的真实翻牌结算。`simulator.py` 为生产引擎协议适配与 Python 参考，`fast_rollout.c/.py` 是完整终局推演加速路径，有完整状态对拍。非 v31 推演策略走 Python 路径。

未知牌分布目前是“满足当前公开数量的均匀联合分布”，没有根据之前对手每次出牌的可能性重加权，不能称为完整历史后验。固定对手策略下的搜索也不代表博弈均衡。

`sample_world` 的受支持用途是自己拥有合法决策机会的观察。它不是任意旁观时点的可达状态生成器；例如旁观他人14张待弃牌时点，不额外排除该隐藏手牌已经自摸胡牌的分配。当前 Bot 与评测均只在自己行动时调用。

### 稀疏树的第二轮改进

普通摸打中的精确历史节点难以重访，因此保留原版对照，并提供两个可选实验开关：

- `tree_key="public_hand"`：未来节点按自己的手牌、各家副露、杠账、墙余量、当前动作机会及公共信息产生的候选集合合并，省略弃牌历史与摸牌来源。这是有损策略抽象，可能混合未知牌分布不同的状态。
- `paired_future=True`：同一世界评估未来节点的多个候选。先根据已有统计选择要向父节点回传的动作，再进行本次评估；禁止按当前隐藏世界的最大回报回传。独立确认阶段只执行冻结策略选定的一个动作。

`future_candidate_limit` 约束实验开关启用时的未来弃牌数。比较预算应看 `leaf_rollouts`（实际终局估值次数）和用时，不能只比较根节点 `simulations`。

## 接入与复现

需要 Python 3.10+、仓库现有依赖、C 编译器以及原有原生查表文件。首次运行自动编译；共享库是本机生成物，不应复制其他架构的 `.so`。主原生库现在构建到 `build/native/`，两种新内核分别在 `build/search159/` 与 `work/search159/build/`。使用进程并行评测；现有原生缓存不提供跨线程并发安全保证。

```bash
python -m unittest discover -s tests -q

# 完整动作与真实结算的成对比赛：每个 seed 的四个座位一起作为统计簇
python -m tools.eval_search159 --a v31 --b finite --opp v31 \
  --finite-horizon 3 --finite-discount 0.7 --seeds 4096 \
  --seed0 6400000 --workers 4 --out work/search159/finite_holdout.json

python -m tools.eval_search159 --a v31 --b search --opp v31 \
  --simulations 8 --confirmation 16 --depth 2 --candidate-limit 3 \
  --confidence-z 1.96 --tree-key public_hand --paired-future \
  --future-candidate-limit 2 --max-search-shanten 1 \
  --seeds 128 --seed0 6500000 --workers 4 --out work/search159/tree_eval.json

# 保存一个真实局面的可见观察、各候选估值及搜索统计
python -m tools.analyze_search159 --seed 766107 --seat 2 --decision 2 \
  --mode search --simulations 64 --confirmation 64 --confidence-z 1.96 \
  --out work/search159/peng_example.json
```

`--opp` 指实际对手，`--opponent-policy` 指搜索内部的对手模型，应分别记录。评测输出含每局原始结果、源码 SHA-256、固定样本数量、按 seed 聚类的 95% Student-t 区间和决策耗时；检测到运行中源码变化会标记该结果，需要冻结后复验。

Git查询不可用会单独标记版本核验不完整，并保存错误诊断及前后快照；不会仅凭一次查询失败认定源码变化。首次4096种子结果保留了旧诊断逻辑产生的标记，对应审计文件记录了前后源码哈希完全一致。

`backend/ai/bot_search159.py` 同时支持统一 `choose_action` 与服务器现有 `decide_gang` / `decide_peng` / `choose_discard` 接口。同一观察只搜索一次，避免服务器连续询问不同动作时重复搜索或给出矛盾答案。`explain()` 返回实际使用的同一份计划。

原生公共改动包括四副露后单张/对子边界修正，以及二步价值的精确记忆缓存。缓存完整比较手牌与未知牌池，哈希碰撞只能导致覆盖，不能串用另一个局面的估值；其开关对照必须得到完全相同的动作和回报。
