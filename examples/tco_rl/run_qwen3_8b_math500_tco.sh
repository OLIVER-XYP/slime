#!/usr/bin/env bash
set -euo pipefail

# One-step Qwen3-8B GRPO campaign. Set TCO_RL_FAULT=reward_flip for the fault run.
: "${MODEL_DIR:=/home/slai/models/Qwen3-8B}"
: "${TORCH_DIST_DIR:=/home/slai/models/Qwen3-8B_torch_dist}"
: "${DATA_DIR:=/home/slai/data/dapo-math-17k/dapo-math-17k.jsonl}"
: "${TRACE_PATH:=/home/slai/work/TCO-TSR/outputs/rl/qwen3_math500_slime.{role}.jsonl}"
: "${TCO_RL_FAULT:=none}"
: "${TCO_RL_DETERMINISTIC_REWARD:=1}"
: "${TCO_RL_MAX_RESPONSE_LEN:=128}"
: "${TCO_RL_MAX_TOKENS_PER_GPU:=2048}"
: "${TCO_RL_ROLLOUT_BATCH_SIZE:=1}"
: "${TCO_RL_N_SAMPLES_PER_PROMPT:=4}"
: "${TCO_RL_GLOBAL_BATCH_SIZE:=4}"
: "${PYTHON_BIN:=python3}"
: "${CUDA_LIB_DIRS:=/home/slai/venv/lib/python3.12/site-packages/torch/lib:/home/slai/venv/lib/python3.12/site-packages/nvidia/cu13/lib:/home/slai/venv/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:/home/slai/venv/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib:/usr/local/cuda/targets/sbsa-linux/lib}"

# Keep runtime JIT tools (for example ninja) visible to Ray/SGLang workers
# when PYTHON_BIN points into a virtual environment.
export PATH="$(dirname "${PYTHON_BIN}"):${PATH}"

if [[ ! -f "${MODEL_DIR}/config.json" ]]; then
  echo "missing Qwen3-8B checkpoint: ${MODEL_DIR}" >&2
  exit 2
fi
if [[ ! -d "${TORCH_DIST_DIR}" ]]; then
  echo "missing Megatron checkpoint: ${TORCH_DIST_DIR}" >&2
  echo "prepare it with tools/convert_hf_to_torch_dist.py first" >&2
  exit 2
fi

export PYTHONUNBUFFERED=1
export PYTHONPATH="${HOME}/src/Megatron-LM:${PWD}:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="${CUDA_LIB_DIRS}:${LD_LIBRARY_PATH:-}"
# FSDP2 keeps a CPU backup of the model; allow the one-step smoke run to use
# the remaining host memory while Ray transitions between rollout and train.
export RAY_memory_usage_threshold="${RAY_memory_usage_threshold:-0.999}"
# The 8B colocated rollout/train smoke run intentionally reuses nearly all host
# memory.  Ray's monitor can kill the training actor after a successful backward
# pass even when no object-store pressure exists, so leave killing disabled for
# this bounded experiment and rely on the process exit/OOM status for failures.
export RAY_memory_monitor_refresh_ms="${RAY_memory_monitor_refresh_ms:-0}"
export TCO_RL_TRACE_PATH="${TRACE_PATH}"
export TCO_RL_FAULT
export TCO_RL_DETERMINISTIC_REWARD
export TCO_RL_FAULT_SAMPLE_INDEX="${TCO_RL_FAULT_SAMPLE_INDEX:-0}"

source scripts/models/qwen3-8B.sh

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
  --num-rollout 1 \
  --rollout-batch-size "${TCO_RL_ROLLOUT_BATCH_SIZE}" \
  --n-samples-per-prompt "${TCO_RL_N_SAMPLES_PER_PROMPT}" \
  --num-steps-per-rollout 1 \
  --global-batch-size "${TCO_RL_GLOBAL_BATCH_SIZE}" \
  --rollout-max-response-len "${TCO_RL_MAX_RESPONSE_LEN}" \
  --rollout-temperature 1.0 \
  --advantage-estimator grpo \
  --eps-clip 0.2 \
  --kl-loss-coef 0.0 \
  --optimizer adam \
  --use-stateless-adam \
  --no-save-optim \
  --lr 1e-6 \
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
  --max-tokens-per-gpu "${TCO_RL_MAX_TOKENS_PER_GPU}" \
  --train-memory-margin-bytes 0 \
  --rollout-num-gpus-per-engine 1 \
  --sglang-mem-fraction-static 0.16 \
  --sglang-disable-cuda-graph \
  --attention-dropout 0.0 \
  --hidden-dropout 0.0
