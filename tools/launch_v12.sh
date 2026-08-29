#!/usr/bin/env bash
# v12 实验组: 用原生 v31n 做世界推演, 128 世界, 50 迭代, 每 5 迭代 eval, 3 个种子。
#
# 与 v11 的差别只有两处: (1) 推演/对手 bot 换成 C 实现(与 Python 版逐位一致,
# 见 backend/ai/test_parity_native.py); (2) 世界数 32 -> 128。其余超参照抄 v11,
# 保证可比。
#
# 并发控制: 每个 run 用 --procs 4, 同时只跑 2 个 run(共 8 进程), 机器只有 10 核。
set -u
cd "$(dirname "$0")/.."

PY=.venv-jax/bin/python
INIT=models/dqn_shaped_100k_best.pt
COMMON="--init $INIT --size base --feat-version 2 --iters 50 --eval-every 5 \
--eval-games 400 --states-per-iter 32 --snaps-per-game 4 --worlds 128 --top-m 4 \
--procs 4 --collect-procs 2 --lr 3e-6 --kl-beta 0.1 --min-sd 1.0"

mkdir -p logs models

JOBS=()
for seed in 720000 730000 740000; do
  for cfg in "v1n:v31n" "v31n:v31n"; do
    opp="${cfg%%:*}"; hero="${cfg##*:}"
    tag="v12_${opp}_${hero}_s${seed}"
    JOBS+=("--rollout-mode $hero --opp-model $opp --seed0 $seed \
--out models/${tag}.pt --log logs/${tag}.log|${tag}")
  done
done

run_one() {
  local spec="$1"
  local args="${spec%%|*}"
  local tag="${spec##*|}"
  echo "[$(date +%H:%M:%S)] start $tag"
  # shellcheck disable=SC2086
  $PY -m backend.rl.grpo_train $COMMON $args > "logs/${tag}.out" 2>&1
  echo "[$(date +%H:%M:%S)] done  $tag rc=$?"
}
export -f run_one
export PY COMMON

printf '%s\n' "${JOBS[@]}" | xargs -d '\n' -I{} -P 2 bash -c 'run_one "$@"' _ {}
echo "[$(date +%H:%M:%S)] 全部 v12 run 结束"
