# TCO RL observer

This optional plugin records the evidence boundaries used by the TCO GRPO
contracts without copying full model tensors into the trace.  It is designed
for Slime custom rollout, advantage and train-step hooks.

Set `TCO_RL_TRACE_PATH` to a JSONL output path, then create one observer in the
driver or hook module:

```python
from slime_plugins.tco_rl import SlimeTCOObserver

observer = SlimeTCOObserver.from_env(experiment_id="qwen3-8b-grpo")
observer.record_rewards(rollout_id, rewards)
observer.record_advantages(rollout_id, advantages)
observer.record_loss(rollout_id, loss=float(loss))
observer.record_optimizer(rollout_id, step=step)
observer.flush()
```

The deterministic CPU reference and fault campaign live in the companion
`TCO-TSR` repository under `src/tco/rl`.  The event names and contract IDs are
shared so a real Slime trace can be handed to the same analysis pipeline.
