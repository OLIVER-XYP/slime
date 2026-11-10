# TCO RL first real campaign

This directory contains the first Qwen3-8B single-step GRPO campaign.

Prepare the HF checkpoint and convert it to a Megatron distributed checkpoint:

```bash
cd /home/slai/work/slime
source scripts/models/qwen3-8B.sh
PYTHONPATH=/home/slai/src/Megatron-LM python tools/convert_hf_to_torch_dist.py \
  "${MODEL_ARGS[@]}" \
  --transformer-impl local \
  --no-rope-fusion \
  --no-persist-layer-norm \
  --no-gradient-accumulation-fusion \
  --hf-checkpoint /home/slai/models/Qwen3-8B \
  --save /home/slai/models/Qwen3-8B_torch_dist
```

Run the control:

```bash
cd /home/slai/work/slime
export TCO_RL_FAULT=none
bash examples/tco_rl/run_qwen3_8b_math500_tco.sh
```

Run the manually injected reward fault with a separate trace:

```bash
cd /home/slai/work/slime
export TCO_RL_FAULT=reward_flip
export TRACE_PATH=/home/slai/work/TCO-TSR/outputs/rl/qwen3_math500_slime.reward_flip.{role}.jsonl
bash examples/tco_rl/run_qwen3_8b_math500_tco.sh
```

The observer records rollout, reward, advantage, policy logprob, loss, backward,
and optimizer boundaries. The smoke script enables a deterministic alternating
reward fixture so truncated generations still produce non-zero group-relative
advantages. The deterministic TCO harness remains the reference for checking
the expected earliest boundary: `reward.verify` for this fault.

If the colocated 8B actor needs a lower memory envelope, override
`TCO_RL_MAX_RESPONSE_LEN` and `TCO_RL_MAX_TOKENS_PER_GPU` for the same command;
the defaults are 128 and 2048. If the actor still reaches a training-time memory
peak, use `TCO_RL_N_SAMPLES_PER_PROMPT=2` and
`TCO_RL_GLOBAL_BATCH_SIZE=2` together so the deterministic reward group remains
non-degenerate while halving the training batch.
