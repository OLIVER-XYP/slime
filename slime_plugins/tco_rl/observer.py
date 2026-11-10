"""Low-overhead, backend-neutral event recording for Slime RL stages.

The observer intentionally stores metadata and scalar summaries rather than
full model tensors. It can be used from Slime hooks without importing TCO's
optional PyTorch/Z3 dependencies into rollout or training workers.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


_GLOBAL_OBSERVER: "SlimeTCOObserver | None" = None


def _as_float(value: Any) -> float | None:
    """Convert a scalar tensor/number to a JSON-safe float when possible."""
    try:
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "item"):
            value = value.item()
        value = float(value)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError, RuntimeError):
        return None


def _summarize_values(values: Any) -> dict[str, Any]:
    """Return small scalar evidence without retaining tensors in the trace."""
    try:
        if hasattr(values, "detach"):
            values = values.detach().float()
            count = int(values.numel())
            if count == 0:
                return {"count": 0}
            return {
                "count": count,
                "mean": float(values.mean().item()),
                "min": float(values.min().item()),
                "max": float(values.max().item()),
            }
        values_list = list(values)
        numeric = [float(value) for value in values_list]
        if not numeric:
            return {"count": 0}
        return {
            "count": len(numeric),
            "mean": sum(numeric) / len(numeric),
            "min": min(numeric),
            "max": max(numeric),
        }
    except (AttributeError, TypeError, ValueError, RuntimeError):
        return {"count": None}


def _flatten_numbers(values: Any) -> list[float]:
    """Flatten tensor-like or nested Python values into scalar floats."""
    if hasattr(values, "detach"):
        try:
            return [float(value) for value in values.detach().float().reshape(-1).tolist()]
        except (AttributeError, TypeError, ValueError, RuntimeError):
            return []
    if isinstance(values, (str, bytes)):
        return []
    try:
        iterator = iter(values)
    except TypeError:
        try:
            return [float(values)]
        except (TypeError, ValueError):
            return []
    flattened: list[float] = []
    for value in iterator:
        flattened.extend(_flatten_numbers(value))
    return flattened


@dataclass
class SlimeTCOObserver:
    """Collect stage events and write a JSONL trace at the end of a run."""

    path: str | Path | None = None
    experiment_id: str = "slime-rl"
    flush_each_event: bool = True
    events: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_env(cls, *, experiment_id: str = "slime-rl") -> "SlimeTCOObserver":
        path = os.environ.get("TCO_RL_TRACE_PATH")
        if path:
            replacements = {
                "rank": os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")),
                "pid": os.getpid(),
                "role": os.environ.get("TCO_RL_ROLE", "unknown"),
            }
            path = path.format(**replacements)
        flush_each_event = os.environ.get("TCO_RL_FLUSH_EACH_EVENT", "1").strip().lower()
        return cls(
            path=path,
            experiment_id=experiment_id,
            flush_each_event=flush_each_event not in {"0", "false", "no", "off"},
        )

    def emit(
        self,
        *,
        stage: str,
        transition: str | None = None,
        operator: str,
        reads: dict[str, Any] | None = None,
        writes: dict[str, Any] | None = None,
        lineage: dict[str, str] | None = None,
        contract_id: str | None = None,
        verdict: str = "unknown",
        residual: float | None = None,
    ) -> dict[str, Any]:
        event = {
            "schema_version": "tco.rl.event/v1",
            "order": len(self.events),
            "experiment_id": self.experiment_id,
            "stage": stage,
            "transition": transition or stage,
            "operator": operator,
            "reads": reads or {},
            "writes": writes or {},
            "lineage": lineage or {},
            "contract_id": contract_id,
            "verdict": verdict,
            "residual": residual,
        }
        self.events.append(event)
        if self.flush_each_event:
            self.flush()
        return event

    @staticmethod
    def _rollout_lineage(rollout_id: Any) -> dict[str, str]:
        if isinstance(rollout_id, (list, tuple, set)):
            values = [str(value) for value in rollout_id]
            unique = list(dict.fromkeys(values))
            return {
                "rollout_id": unique[0] if len(unique) == 1 else "batch",
                "rollout_ids": ",".join(unique),
            }
        return {"rollout_id": str(rollout_id)}

    def record_rollout(self, rollout_id: int | str, samples: list[Any]) -> None:
        self.emit(
            stage="rollout.generate",
            operator="slime.rollout",
            reads={"sample_count": len(samples)},
            writes={
                "rollout_id": str(rollout_id),
                "sample_ids": [str(getattr(sample, "id", index)) for index, sample in enumerate(samples)],
            },
            lineage=self._rollout_lineage(rollout_id),
            contract_id="rl.rollout.lineage",
        )

    def record_rewards(
        self,
        rollout_id: int | str,
        rewards: Any,
        *,
        raw_rewards: Any | None = None,
    ) -> None:
        writes = {"rewards": [float(reward) for reward in rewards]}
        if raw_rewards is not None:
            writes["raw_rewards"] = [float(reward) for reward in raw_rewards]
        self.emit(
            stage="reward.verify",
            operator="slime.reward_model",
            reads={"rollout_id": str(rollout_id), "reward_summary": _summarize_values(rewards)},
            writes=writes,
            lineage=self._rollout_lineage(rollout_id),
            contract_id="rl.reward.verifier_consistency",
        )

    def record_advantages(self, rollout_id: Any, advantages: Any) -> None:
        self.emit(
            stage="advantage.compute",
            operator="slime.compute_advantages_and_returns",
            reads={"rollout_id": str(rollout_id)},
            writes={"advantage_summary": _summarize_values(advantages)},
            lineage=self._rollout_lineage(rollout_id),
            contract_id="rl.advantage.group_relative",
        )

    def record_policy(self, rollout_id: Any, *, old_log_probs: Any, current_log_probs: Any) -> None:
        """Record scalar policy-ratio evidence without copying token tensors."""
        try:
            delta = (current_log_probs - old_log_probs).detach().float()
            summary = {
                "token_count": int(delta.numel()),
                "max_abs_logprob_delta": float(delta.abs().max().item()) if delta.numel() else 0.0,
                "mean_abs_logprob_delta": float(delta.abs().mean().item()) if delta.numel() else 0.0,
            }
        except (AttributeError, TypeError, RuntimeError):
            old_values = _flatten_numbers(old_log_probs)
            current_values = _flatten_numbers(current_log_probs)
            deltas = [abs(current - old) for old, current in zip(old_values, current_values)]
            summary = {
                "token_count": len(deltas),
                "max_abs_logprob_delta": max(deltas, default=0.0),
                "mean_abs_logprob_delta": sum(deltas) / len(deltas) if deltas else 0.0,
            }
        self.emit(
            stage="policy.logprob",
            operator="slime.policy_loss_function",
            reads={"rollout_id": str(rollout_id)},
            writes=summary,
            lineage=self._rollout_lineage(rollout_id),
            contract_id="rl.policy.logprob_consistency",
        )

    def record_loss(
        self,
        rollout_id: Any,
        *,
        loss: Any,
        mask_sum: Any | None = None,
        grad_norm: Any | None = None,
    ) -> None:
        self.emit(
            stage="grpo.loss",
            operator="slime.policy_loss_function",
            reads={"rollout_id": str(rollout_id), "mask_sum": _as_float(mask_sum)},
            writes={"loss": _as_float(loss), "grad_norm": _as_float(grad_norm)},
            lineage=self._rollout_lineage(rollout_id),
            contract_id="rl.grpo.clipped_loss",
        )

    def record_backward(self, rollout_id: Any, *, grad_norm: Any | None = None) -> None:
        self.emit(
            stage="backward.compute",
            operator="slime.train_one_step",
            reads={"rollout_id": str(rollout_id)},
            writes={"grad_norm": _as_float(grad_norm)},
            lineage=self._rollout_lineage(rollout_id),
            contract_id="rl.backward.gradient_consistency",
        )

    def record_optimizer(self, rollout_id: Any, *, step: int, parameter_version: str | None = None) -> None:
        version = parameter_version or str(step)
        self.emit(
            stage="optimizer.step",
            operator="slime.optimizer",
            reads={"rollout_id": str(rollout_id), "step": int(step)},
            writes={"parameter_version": version},
            lineage={"rollout_id": str(rollout_id), "parameter_version": version},
            contract_id="optimizer.adamw.update",
        )

    def flush(self) -> Path | None:
        if self.path is None:
            return None
        path = Path(self.path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        temp_path.write_text(
            "".join(json.dumps(event, sort_keys=True) + "\n" for event in self.events),
            encoding="utf-8",
        )
        temp_path.replace(path)
        return path


def get_global_observer(*, experiment_id: str = "slime-rl") -> SlimeTCOObserver | None:
    """Return a process-local observer when TCO tracing is enabled."""
    global _GLOBAL_OBSERVER
    if not os.environ.get("TCO_RL_TRACE_PATH"):
        return None
    if _GLOBAL_OBSERVER is None:
        _GLOBAL_OBSERVER = SlimeTCOObserver.from_env(experiment_id=experiment_id)
    return _GLOBAL_OBSERVER

