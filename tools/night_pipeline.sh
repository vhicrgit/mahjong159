#!/bin/bash
# 夜间流水线: 等 v10 塑形数据 -> DQN 训练 -> GRPO(以DQN最优初始化)
cd ~/mahjong159
while [ ! -f models/offline_v10_shaped_40k.npz ]; do sleep 60; done
sleep 60  # 等待写入完成
echo "dqn v10 start: $(date)" >> /tmp/pipeline_status.log
CUDA_VISIBLE_DEVICES=0 python3 -m backend.rl.dqn_offline --init "" \
  --data models/offline_v10_shaped_40k.npz --size base --epochs 25 \
  --eval-games 800 --mid-eval-every 5 --cql-weight 1.0 \
  --out models/dqn_v10_shaped.pt > /tmp/dqn_v10_shaped.log 2>&1
echo "dqn v10 done: $(date)" >> /tmp/pipeline_status.log
CUDA_VISIBLE_DEVICES=0 python3 -m backend.rl.grpo_train \
  --init models/dqn_v10_shaped_best.pt --size base --feat-version 3 \
  --iters 40 --states-per-iter 128 --snaps-per-game 4 --worlds 16 \
  --top-m 4 --procs 20 --collect-procs 10 --eval-every 5 --eval-games 400 \
  --kl-beta 0.2 --lr 3e-6 --inner-epochs 1 \
  --out models/grpo_v3.pt --log logs/grpo_v3.log > /tmp/grpo_v3.log 2>&1
echo "grpo v3 done: $(date)" >> /tmp/pipeline_status.log
