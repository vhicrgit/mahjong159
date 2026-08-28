#!/bin/bash
# 启动 7 个并行 GRPO-JAX 训练流 (GPU0,2-7), 超参探索
cd /home/zuofengrui.zfr/mahjong159
export PYTHONPATH=/home/zuofengrui.zfr/mahjong159
export JAX_COMPILATION_CACHE_DIR=/home/zuofengrui.zfr/mahjong159/.jax_cache
export NVLIBS="$(cat .venv-jax/nvlibs.txt | sed 's|^|/home/zuofengrui.zfr/mahjong159/|')"
export LD_LIBRARY_PATH="${NVLIBS}:$LD_LIBRARY_PATH"
BASE="--size base --feat-version 2 --rollout jax --iters 60 --states-per-iter 32 --snaps-per-game 4 --worlds 128 --top-m 4 --procs 4 --collect-procs 3 --eval-every 10 --eval-games 300 --inner-epochs 1 --init models/dqn_shaped_100k_best.pt"

# 流A: 更激进 (lr=1e-5, kl=0.1)
CUDA_VISIBLE_DEVICES=0 setsid nohup .venv-jax/bin/python3 -m backend.rl.grpo_train $BASE \
  --lr 1e-5 --kl-beta 0.1 --seed0 730000 --out models/grpo_sweepA.pt --log logs/grpo_sweepA.log \
  > /tmp/grpo_sweepA.out 2>&1 < /dev/null &

# 流B: 更保守 (lr=3e-6, kl=0.5)
CUDA_VISIBLE_DEVICES=2 setsid nohup .venv-jax/bin/python3 -m backend.rl.grpo_train $BASE \
  --lr 3e-6 --kl-beta 0.5 --seed0 740000 --out models/grpo_sweepB.pt --log logs/grpo_sweepB.log \
  > /tmp/grpo_sweepB.out 2>&1 < /dev/null &

# 流C: 更多局面更少世界 (64 snaps, 64 worlds)
CUDA_VISIBLE_DEVICES=3 setsid nohup .venv-jax/bin/python3 -m backend.rl.grpo_train $BASE \
  --states-per-iter 64 --worlds 64 --seed0 750000 --out models/grpo_sweepC.pt --log logs/grpo_sweepC.log \
  > /tmp/grpo_sweepC.out 2>&1 < /dev/null &

# 流D: 更少局面更多世界 (16 snaps, 256 worlds)
CUDA_VISIBLE_DEVICES=4 setsid nohup .venv-jax/bin/python3 -m backend.rl.grpo_train $BASE \
  --states-per-iter 16 --worlds 256 --seed0 760000 --out models/grpo_sweepD.pt --log logs/grpo_sweepD.log \
  > /tmp/grpo_sweepD.out 2>&1 < /dev/null &

# 流E: 同默认但不同 seed
CUDA_VISIBLE_DEVICES=5 setsid nohup .venv-jax/bin/python3 -m backend.rl.grpo_train $BASE \
  --seed0 770000 --out models/grpo_sweepE.pt --log logs/grpo_sweepE.log \
  > /tmp/grpo_sweepE.out 2>&1 < /dev/null &

# 流F: 从 grpo_jax_v1 初始化(前一轮产物)
CUDA_VISIBLE_DEVICES=6 setsid nohup .venv-jax/bin/python3 -m backend.rl.grpo_train $BASE \
  --init models/grpo_jax_v1_best.pt --seed0 780000 --out models/grpo_sweepF.pt --log logs/grpo_sweepF.log \
  > /tmp/grpo_sweepF.out 2>&1 < /dev/null &

# 流G: small 模型快速验证
CUDA_VISIBLE_DEVICES=7 setsid nohup .venv-jax/bin/python3 -m backend.rl.grpo_train $BASE \
  --size small --init models/dqn_shaped_100k_best.pt --seed0 790000 --out models/grpo_sweepG.pt --log logs/grpo_sweepG.log \
  > /tmp/grpo_sweepG.out 2>&1 < /dev/null &

echo "7 sweeps launched"
sleep 5
pgrep -f "grpo_sweep" | wc -l
