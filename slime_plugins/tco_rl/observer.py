"""Low-overhead, backend-neutral event recording for Slime RL stages.

The observer intentionally stores metadata and scalar summaries rather than
full model tensors.  It can be used from Slime custom hooks without importing
TCO's optional PyTorch/Z3 dependencies into the rollout workers.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class SlimeTCOObserver:
    """Collect stage events and write a JSONL trace at the end of a run."""

    path: str | Path | None = None
    experiment_id: str = "slime-rl"
    events: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_env(cls, *, experiment_id: str = "slime-rl") -> "SlimeTCOObserver":
        return cls(path=os.environ.get("TCO_RL_TRACE_PATH"), experiment_id=experiment_id)

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
        return event

    def record_rollout(self, rollout_id: int | str, samples: list[Any]) -> None:
        self.emit(
            stage="rollout.generate",
            operator="slime.rollout",
            reads={"sample_count": len(samples)},
            writes={
                "rollout_id": str(rollout_id),
                "sample_ids": [str(getattr(sample, "id", index)) for index, sample in enumerate(samples)],
            },
            lineage={"rollout_id": str(rollout_id)},
            contract_id="rl.rollout.lineage",
        )

    def record_rewards(self, rollout_id: int | str, rewards: list[float]) -> None:
        self.emit(
            stage="reward.verify",
            operator="slime.reward_model",
            reads={"rollout_id": str(rollout_id)},
            writes={"rewards": [float(reward) for reward in rewards]},
            lineage={"rollout_id": str(rollout_id)},
            contract_id="rl.reward.verifier_consistency",
        )

    def record_advantages(self, rollout_id: int | str, advantages: list[Any]) -> None:
        self.emit(
            stage="advantage.compute",
            operator="slime.compute_advantages_and_returns",
            reads={"rollout_id": str(rollout_id)},
            writes={"sample_count": len(advantages)},
            lineage={"rollout_id": str(rollout_id)},
            contract_id="rl.advantage.group_relative",
        )

    def record_loss(self, rollout_id: int | str, *, loss: float, mask_sum: float | None = None) -> None:
        self.emit(
            stage="grpo.loss",
            operator="slime.policy_loss_function",
            reads={"rollout_id": str(rollout_id), "mask_sum": mask_sum},
            writes={"loss": float(loss)},
            lineage={"rollout_id": str(rollout_id)},
            contract_id="rl.grpo.clipped_loss",
        )

    def record_optimizer(self, rollout_id: int | str, *, step: int, parameter_version: str | None = None) -> None:
        self.emit(
            stage="optimizer.step",
            operator="slime.optimizer",
            reads={"rollout_id": str(rollout_id), "step": int(step)},
            writes={"parameter_version": parameter_version or str(step)},
            lineage={"rollout_id": str(rollout_id), "parameter_version": parameter_version or str(step)},
            contract_id="optimizer.adamw.update",
        )

    def flush(self) -> Path | None:
        if self.path is None:
            return None
        path = Path(self.path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(event, sort_keys=True) + "\n" for event in self.events), encoding="utf-8")
        return path

