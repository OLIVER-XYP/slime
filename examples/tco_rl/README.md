# TCO RL MATH-500 campaign

This directory contains the reproducible Qwen3-8B math experiment used to
exercise TCO across the full RL chain. The public `HuggingFaceH4/MATH-500`
test set is used only for evaluation; GRPO training uses the public
`zhuzilin/dapo-math-17k` data. The evaluator and Slime reward hook share the
same answer parser and math verifier.

Prepare the complete evaluation data on the experiment host:

```bash
hf download HuggingFaceH4/MATH-500 --repo-type dataset \
  --local-dir /home/slai/data/MATH-500 --include '*.json*'
cd /home/slai/work/slime
python examples/tco_rl/prepare_math500.py \
  --input /home/slai/data/MATH-500/test.jsonl \
  --output /home/slai/data/MATH-500/math500_slime.jsonl
```

Run a fixed-seed full baseline. The output JSON contains every response and
the aggregate `pass@1` (or `pass@k` when `--n-samples k` is used):

```bash
PYTHONPATH=/home/slai/work/slime:/home/slai/src/Megatron-LM \
python examples/tco_rl/eval_math500.py \
  --model /home/slai/models/Qwen3-8B \
  --data /home/slai/data/MATH-500/test.jsonl \
  --out /home/slai/artifacts/math500_baseline_n1.json \
  --start-server --n-samples 1 --max-new-tokens 4096 \
  --temperature 0.6 --top-p 0.95 --max-concurrency 64
```

Run real multi-step GRPO with online rewards and checkpoint export:

```bash
cd /home/slai/work/slime
NUM_ROLLOUT=24 ROLLOUT_BATCH_SIZE=4 N_SAMPLES_PER_PROMPT=4 \
  bash examples/tco_rl/run_qwen3_8b_math500_grpo.sh
```

The default single-GB10 recipe trains `output_layer` only so the full rollout,
reward, advantage, loss, backward, optimizer, weight-sync, and checkpoint path
can run within the machine's unified-memory limit. `TRAIN_PARAM_PATTERNS` can
be changed after a memory check. Checkpoints are written under
`/home/slai/artifacts/qwen3-8b-tco-grpo/` and exported as Hugging Face models
under `hf/step-*`; each run also writes compact TCO JSONL traces.

Evaluate the final exported checkpoint with exactly the baseline configuration:

```bash
PYTHONPATH=/home/slai/work/slime:/home/slai/src/Megatron-LM \
python examples/tco_rl/eval_math500.py \
  --model /home/slai/artifacts/qwen3-8b-tco-grpo/hf/step-23 \
  --data /home/slai/data/MATH-500/test.jsonl \
  --out /home/slai/artifacts/math500_grpo_step23_n1.json \
  --start-server --n-samples 1 --max-new-tokens 4096 \
  --temperature 0.6 --top-p 0.95 --max-concurrency 64
```

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
