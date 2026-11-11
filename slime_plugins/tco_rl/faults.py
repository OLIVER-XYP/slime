"""Small, explicit fault hooks for the first Slime TCO RL campaign.

The hooks are disabled unless ``TCO_RL_FAULT`` is set. They are deliberately
limited to one deterministic sample so a control/fault pair has a reviewable
causal difference.
"""

from __future__ import annotations

import os
from typing import Any

from slime.rollout.rm_hub.math_utils import extract_answer, grade_answer_mathd, grade_answer_sympy


def _math_reward(response: str, label: Any) -> float:
    """Score both Qwen3 thinking and plain ``Answer:`` responses.

    Qwen3 can be configured with thinking disabled, in which case the response
    has no ``</think>`` marker.  The upstream DeepScaler hook treats such a
    response as malformed and returns zero, which would make real GRPO runs
    degenerate.  TCO's RL campaign keeps the same answer parser and verifier,
    while allowing either response envelope.
    """
    if "</think>" in response:
        model_solution = response.split("</think>")[-1]
    elif "###Response" in response:
        model_solution = response.split("###Response", 1)[1]
    else:
        model_solution = response
    model_answer = extract_answer(model_solution)
    if model_answer is None or label in (None, ""):
        return 0.0
    truth = extract_answer(str(label)) if "\\boxed" in str(label) else str(label)
    if truth is None:
        return 0.0
    return float(grade_answer_mathd(model_answer, truth) or grade_answer_sympy(model_answer, truth))


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
    reward_value = _math_reward(response, label)

    # The first GRPO smoke run is intentionally deterministic.  Qwen3's
    # colocated rollout can be truncated before it emits a parseable answer,
    # which would collapse every group to zero reward and hide the downstream
    # advantage/loss contracts.  Use alternating within-group rewards only when
    # explicitly enabled by the campaign script; normal Slime reward behavior
    # remains unchanged when the variable is unset or false.
    if os.environ.get("TCO_RL_DETERMINISTIC_REWARD", "0").lower() in {"1", "true", "yes"}:
        sample_index = getattr(sample, "index", None)
        if sample_index is not None:
            reward_value = 1.0 if int(sample_index) % 2 == 0 else 0.0

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
