#!/usr/bin/env python3
"""Evaluate a Qwen3 checkpoint on the complete public MATH-500 set.

The evaluator uses the same DeepScaler rule-based verifier as the Slime GRPO
reward hook. It records every response, verifier result, sampling settings,
and aggregate pass@k metrics so a training run is directly auditable.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import aiohttp
from transformers import AutoTokenizer

from slime.rollout.rm_hub.deepscaler import get_deepscaler_rule_based_reward


INSTRUCTION = (
    "Solve the following math problem step by step. "
    "The last line of your response must be of the form "
    "Answer: \\boxed{$Answer}, where $Answer is the answer.\n\n"
)


def _load_items(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def _prompt(tokenizer: Any, problem: str) -> str:
    messages = [{"role": "user", "content": INSTRUCTION + problem}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


async def _wait_ready(base_url: str, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    async with aiohttp.ClientSession() as session:
        while time.monotonic() < deadline:
            try:
                async with session.get(f"{base_url}/health", timeout=5) as response:
                    if response.status == 200:
                        return
            except (aiohttp.ClientError, asyncio.TimeoutError):
                pass
            await asyncio.sleep(2)
    raise TimeoutError(f"SGLang did not become ready at {base_url}")


async def _generate_one(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    base_url: str,
    prompt: str,
    sampling_params: dict[str, Any],
) -> str:
    async with semaphore:
        for attempt in range(4):
            try:
                async with session.post(
                    f"{base_url}/generate",
                    json={"text": prompt, "sampling_params": sampling_params},
                    timeout=aiohttp.ClientTimeout(total=600),
                ) as response:
                    response.raise_for_status()
                    payload = await response.json()
                    return str(payload.get("text", ""))
            except (aiohttp.ClientError, asyncio.TimeoutError):
                if attempt == 3:
                    raise
                await asyncio.sleep(2**attempt)
    raise AssertionError("unreachable")


async def _evaluate(
    items: list[dict[str, Any]],
    tokenizer: Any,
    *,
    base_url: str,
    n_samples: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    max_concurrency: int,
    seed: int,
) -> list[dict[str, Any]]:
    semaphore = asyncio.Semaphore(max_concurrency)
    sampling = {
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "skip_special_tokens": True,
    }
    results: list[str] = []
    async with aiohttp.ClientSession() as session:
        # Rebuild tasks with the live session. Keeping all requests in flight lets
        # SGLang batch the full MATH-500 evaluation efficiently.
        tasks = []
        for prompt_index, item in enumerate(items):
            prompt = _prompt(tokenizer, item["problem"])
            for sample_index in range(n_samples):
                params = dict(sampling)
                params["sampling_seed"] = seed + prompt_index * n_samples + sample_index
                tasks.append(_generate_one(session, semaphore, base_url, prompt, params))
        results = await asyncio.gather(*tasks)

    records = []
    offset = 0
    for prompt_index, item in enumerate(items):
        samples = []
        for sample_index in range(n_samples):
            response = results[offset]
            offset += 1
            reward = float(get_deepscaler_rule_based_reward(response, item["answer"]))
            samples.append(
                {
                    "sample_idx": sample_index,
                    "reward": reward,
                    "correct": reward > 0.5,
                    "response": response,
                }
            )
        records.append(
            {
                "prompt_idx": prompt_index,
                "unique_id": item["unique_id"],
                "answer": item["answer"],
                "samples": samples,
            }
        )
    return records


def _summary(records: list[dict[str, Any]], n_samples: int) -> dict[str, Any]:
    total = len(records)
    result: dict[str, Any] = {
        "prompts": total,
        "samples": total * n_samples,
        "sample_accuracy": sum(
            int(sample["correct"])
            for record in records
            for sample in record["samples"]
        )
        / max(1, total * n_samples),
    }
    for k in range(1, n_samples + 1):
        result[f"pass@{k}"] = sum(
            any(sample["correct"] for sample in record["samples"][:k])
            for record in records
        ) / max(1, total)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:30000")
    parser.add_argument("--start-server", action="store_true")
    parser.add_argument("--server-log", type=Path)
    parser.add_argument("--context-length", type=int, default=8192)
    parser.add_argument("--mem-fraction-static", type=float, default=0.30)
    parser.add_argument("--n-samples", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-concurrency", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20261111)
    args = parser.parse_args()

    process = None
    if args.start_server:
        log_path = args.server_log or args.out.with_suffix(".server.log")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log = log_path.open("w", encoding="utf-8")
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "sglang.launch_server",
                "--model-path",
                str(args.model),
                "--host",
                "127.0.0.1",
                "--port",
                str(args.base_url.rsplit(":", 1)[-1]),
                "--tp-size",
                "1",
                "--context-length",
                str(args.context_length),
                "--mem-fraction-static",
                str(args.mem_fraction_static),
                "--disable-cuda-graph",
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            env=os.environ.copy(),
        )

    try:
        items = _load_items(args.data)
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        asyncio.run(_wait_ready(args.base_url, 900))
        records = asyncio.run(
            _evaluate(
                items,
                tokenizer,
                base_url=args.base_url,
                n_samples=args.n_samples,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                max_concurrency=args.max_concurrency,
                seed=args.seed,
            )
        )
        payload = {
            "schema_version": "tco.rl.math500.eval/v1",
            "model": str(args.model),
            "data": str(args.data),
            "sampling": {
                "n_samples": args.n_samples,
                "max_new_tokens": args.max_new_tokens,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "seed": args.seed,
            },
            "summary": _summary(records, args.n_samples),
            "records": records,
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(payload["summary"], sort_keys=True))
    finally:
        if process is not None:
            process.send_signal(signal.SIGTERM)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
