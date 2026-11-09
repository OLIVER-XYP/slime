"""Small, explicit fault hooks for the first Slime TCO RL campaign.

The hooks are disabled unless ``TCO_RL_FAULT`` is set. They are deliberately
limited to one deterministic sample so a control/fault pair has a reviewable
causal difference.
"""

from __future__ import annotations

import os
from typing import Any

from slime.rollout.rm_hub.deepscaler import get_deepscaler_rule_based_reward


def apply_reward_fault(
    reward: float,
    *,
    sample_index: int | None,
    fault: str | None = None,
    target_index: int = 0,
    already_applied: bool = False,
) -> tuple[float, bool]:
    """Apply one reward flip and return ``(reward, applied)``."""
    if fault != "reward_flip" or already_applied or sample_index != target_index:
        return float(reward), already_applied
    return 1.0 - float(reward), True


async def reward(args: Any, sample: Any, **_: Any) -> float:
    """Score a math sample, with an opt-in single-sample reward flip."""
    response = sample.response
    label = sample.label
    if "</think>" in response:
        response = response.split("</think>")[-1]
    reward_value = get_deepscaler_rule_based_reward(response, label)

    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    target_index = int(os.environ.get("TCO_RL_FAULT_SAMPLE_INDEX", "0"))
    fault = os.environ.get("TCO_RL_FAULT", "none")
    sample_index = getattr(sample, "index", None)
    updated, applied = apply_reward_fault(
        reward_value,
        sample_index=sample_index,
        fault=fault,
        target_index=target_index,
        already_applied=bool(metadata.get("_tco_reward_fault_applied", False)),
    )
    if applied and not metadata.get("_tco_reward_fault_applied", False):
        metadata["_tco_reward_fault_applied"] = True
        sample.metadata = metadata
    return updated
