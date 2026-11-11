#!/usr/bin/env bash
set -euo pipefail

# Multi-step, real-reward GRPO campaign for Qwen3-8B. The default parameter
# filter keeps the experiment runnable on a single GB10; set
# TRAIN_PARAM_PATTERNS to expand the trainable subset after a memory check.
: "${MODEL_DIR:=/home/slai/models/Qwen3-8B}"
: "${TORCH_DIST_DIR:=/home/slai/models/Qwen3-8B_torch_dist}"
: "${DATA_DIR:=/home/slai/data/dapo-math-17k/dapo-math-17k.jsonl}"
: "${SAVE_DIR:=/home/slai/artifacts/qwen3-8b-tco-grpo}"
: "${TRACE_PATH:=/home/slai/work/TCO-TSR/outputs/rl/qwen3_math500_grpo.{role}.jsonl}"
: "${NUM_ROLLOUT:=24}"
: "${ROLLOUT_BATCH_SIZE:=4}"
: "${N_SAMPLES_PER_PROMPT:=4}"
: "${GLOBAL_BATCH_SIZE:=16}"
: "${MAX_RESPONSE_LEN:=512}"
: "${MAX_TOKENS_PER_GPU:=4096}"
: "${SGLANG_MEM_FRACTION_STATIC:=0.20}"
: "${TRAIN_PARAM_PATTERNS:=output_layer}"
: "${PYTHON_BIN:=/home/slai/venv/bin/python}"
: "${CUDA_LIB_DIRS:=/home/slai/venv/lib/python3.12/site-packages/torch/lib:/home/slai/venv/lib/python3.12/site-packages/nvidia/cu13/lib:/home/slai/venv/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:/home/slai/venv/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib:/usr/local/cuda/targets/sbsa-linux/lib}"

export PATH="$(dirname "${PYTHON_BIN}"):${PATH}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="/home/slai/src/Megatron-LM:${PWD}:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="${CUDA_LIB_DIRS}:${LD_LIBRARY_PATH:-}"
export RAY_memory_usage_threshold="${RAY_memory_usage_threshold:-0.999}"
export RAY_memory_monitor_refresh_ms="${RAY_memory_monitor_refresh_ms:-0}"
export TCO_RL_TRACE_PATH="${TRACE_PATH}"
export TCO_RL_FAULT="none"
export TCO_RL_DETERMINISTIC_REWARD="0"
export TCO_RL_FLUSH_EACH_EVENT="1"

if [[ ! -f "${MODEL_DIR}/config.json" ]]; then
  echo "missing Qwen3-8B checkpoint: ${MODEL_DIR}" >&2
  exit 2
fi
if [[ ! -d "${TORCH_DIST_DIR}" ]]; then
  echo "missing Megatron checkpoint: ${TORCH_DIST_DIR}" >&2
  exit 2
fi

mkdir -p "${SAVE_DIR}/checkpoints" "${SAVE_DIR}/hf"
source scripts/models/qwen3-8B.sh

PARAM_ARGS=()
read -r -a PARAM_ARGS <<< "${TRAIN_PARAM_PATTERNS}"

"${PYTHON_BIN}" train.py \
  "${MODEL_ARGS[@]}" \
  --actor-num-nodes 1 \
  --actor-num-gpus-per-node 1 \
  --colocate \
  --hf-checkpoint "${MODEL_DIR}" \
  --ref-load "${TORCH_DIST_DIR}" \
  --prompt-data "${DATA_DIR}" \
  --input-key prompt \
  --label-key label \
  --apply-chat-template \
  --custom-rm-path slime_plugins.tco_rl.faults.reward \
  --num-rollout "${NUM_ROLLOUT}" \
  --rollout-batch-size "${ROLLOUT_BATCH_SIZE}" \
  --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT}" \
  --num-steps-per-rollout 1 \
  --global-batch-size "${GLOBAL_BATCH_SIZE}" \
  --rollout-shuffle \
  --rollout-seed 20261111 \
  --rollout-max-response-len "${MAX_RESPONSE_LEN}" \
  --rollout-max-context-len 8192 \
  --rollout-temperature 1.0 \
  --advantage-estimator grpo \
  --eps-clip 0.2 \
  --kl-loss-coef 0.0 \
  --optimizer adam \
  --use-stateless-adam \
  --no-save-optim \
  --lr 1e-6 \
  --save "${SAVE_DIR}/checkpoints" \
  --save-interval 8 \
  --save-hf "${SAVE_DIR}/hf/step-{rollout_id}" \
  --only-train-params-name-list "${PARAM_ARGS[@]}" \
  --tensor-model-parallel-size 1 \
  --pipeline-model-parallel-size 1 \
  --context-parallel-size 1 \
  --transformer-impl local \
  --no-rope-fusion \
  --no-persist-layer-norm \
  --no-gradient-accumulation-fusion \
  --no-masked-softmax-fusion \
  --grad-reduce-in-bf16 \
  --no-use-distributed-optimizer \
  --use-torch-fsdp2 \
  --megatron-fsdp-main-params-dtype bf16 \
  --megatron-fsdp-main-grads-dtype bf16 \
  --expert-model-parallel-size 1 \
  --expert-tensor-parallel-size 1 \
  --use-dynamic-batch-size \
  --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU}" \
  --recompute-activations \
  --train-memory-margin-bytes 0 \
  --rollout-num-gpus-per-engine 1 \
  --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION_STATIC}" \
  --sglang-disable-cuda-graph \
  --attention-dropout 0.0 \
  --hidden-dropout 0.0
