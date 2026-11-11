#!/usr/bin/env python3
"""Convert the public MATH-500 JSONL into Slime prompt-data format."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


INSTRUCTION = (
    "Solve the following math problem step by step. "
    "The last line of your response must be of the form "
    "Answer: \\boxed{$Answer}, where $Answer is the answer.\n\n"
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    records = []
    with args.input.open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            item = json.loads(line)
            prompt = [{"role": "user", "content": INSTRUCTION + item["problem"]}]
            records.append(
                {
                    "prompt": prompt,
                    "label": item["answer"],
                    "unique_id": item["unique_id"],
                    "subject": item.get("subject"),
                    "level": item.get("level"),
                }
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as target:
        for item in records:
            target.write(json.dumps(item, ensure_ascii=False) + "\n")

    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    manifest = args.output.with_suffix(args.output.suffix + ".manifest.json")
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "tco.rl.math500.prompt_data/v1",
                "source": str(args.input),
                "records": len(records),
                "sha256": digest,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(args.output), "records": len(records), "sha256": digest}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
