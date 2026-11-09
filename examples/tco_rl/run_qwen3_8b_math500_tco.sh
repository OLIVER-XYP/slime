#!/usr/bin/env bash
set -euo pipefail

# One-step Qwen3-8B GRPO campaign. Set TCO_RL_FAULT=reward_flip for the fault run.
: "${MODEL_DIR:=/home/slai/models/Qwen3-8B}"
: "${TORCH_DIST_DIR:=/home/slai/models/Qwen3-8B_torch_dist}"
: "${DATA_DIR:=/home/slai/data/dapo-math-17k/dapo-math-17k.jsonl}"
: "${TRACE_PATH:=/home/slai/work/TCO-TSR/outputs/rl/qwen3_math500_slime.{role}.jsonl}"
: "${TCO_RL_FAULT:=none}"
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
export TCO_RL_TRACE_PATH="${TRACE_PATH}"
export TCO_RL_FAULT
export TCO_RL_FAULT_SAMPLE_INDEX="${TCO_RL_FAULT_SAMPLE_INDEX:-0}"

source scripts/models/qwen3-8B.sh

"${PYTHON_BIN}" train.py \
  "${MODEL_ARGS[@]}" \
  --actor-num-nodes 1 \
  --actor-num-gpus-per-node 1 \
  --colocate \
  --hf-checkpoint "${MODEL_DIR}" \
  --ref-load "${TORCH_DIST_DIR}" \
  --save /tmp/qwen3-8b-tco-rl \
  --save-interval 1 \
  --prompt-data "${DATA_DIR}" \
  --input-key prompt \
  --label-key label \
  --apply-chat-template \
  --custom-rm-path slime_plugins.tco_rl.faults.reward \
  --num-rollout 1 \
  --rollout-batch-size 1 \
  --n-samples-per-prompt 4 \
  --num-steps-per-rollout 1 \
  --global-batch-size 4 \
  --rollout-max-response-len 512 \
  --rollout-temperature 1.0 \
  --advantage-estimator grpo \
  --eps-clip 0.2 \
  --kl-loss-coef 0.0 \
  --optimizer adam \
  --lr 1e-6 \
  --tensor-model-parallel-size 1 \
  --pipeline-model-parallel-size 1 \
  --context-parallel-size 1 \
  --transformer-impl local \
  --no-rope-fusion \
  --no-persist-layer-norm \
  --no-gradient-accumulation-fusion \
  --grad-reduce-in-bf16 \
  --no-use-distributed-optimizer \
  --use-torch-fsdp2 \
  --megatron-fsdp-main-params-dtype bf16 \
  --megatron-fsdp-main-grads-dtype bf16 \
  --expert-model-parallel-size 1 \
  --expert-tensor-parallel-size 1 \
  --use-dynamic-batch-size \
  --max-tokens-per-gpu 4096 \
  --train-memory-margin-bytes 0 \
  --rollout-num-gpus-per-engine 1 \
  --sglang-mem-fraction-static 0.16 \
  --sglang-disable-cuda-graph \
  --attention-dropout 0.0 \
  --hidden-dropout 0.0
