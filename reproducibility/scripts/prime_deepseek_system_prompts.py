#!/usr/bin/env python3
"""Prime and verify DeepSeek cache units at observed system-prompt boundaries."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Any

import httpx


def load_prompts(path: Path, min_chars: int) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    unique: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        if int(row["system_prompt_chars"]) >= min_chars:
            unique[(str(row["method"]), str(row["system_prompt_hash"]))] = row
    return list(unique.values())


async def request(
    client: httpx.AsyncClient,
    *,
    api_key: str,
    model: str,
    user_id: str,
    system_prompt: str,
    user_prompt: str,
) -> dict[str, Any]:
    response = await client.post(
        "https://api.deepseek.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": model,
            "thinking": {"type": "disabled"},
            "user_id": user_id,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0,
            "max_tokens": 1,
        },
    )
    response.raise_for_status()
    return response.json()


async def main_async(args: argparse.Namespace) -> None:
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise SystemExit(f"{args.api_key_env} is required")
    prompts = load_prompts(args.prompt_log, args.min_chars)
    if not prompts:
        raise SystemExit("no cache-eligible system prompts found")

    async with httpx.AsyncClient(timeout=300.0) as client:
        for row in prompts:
            user_id = f"{args.isolation_prefix}-{row['method']}"[:512]
            await request(
                client,
                api_key=api_key,
                model=args.model,
                user_id=user_id,
                system_prompt=row["system_prompt"],
                user_prompt="A",
            )
            await request(
                client,
                api_key=api_key,
                model=args.model,
                user_id=user_id,
                system_prompt=row["system_prompt"],
                user_prompt="Z",
            )

        await asyncio.sleep(args.settle_seconds)
        validations = []
        for row in prompts:
            user_id = f"{args.isolation_prefix}-{row['method']}"[:512]
            payload = await request(
                client,
                api_key=api_key,
                model=args.model,
                user_id=user_id,
                system_prompt=row["system_prompt"],
                user_prompt="Q",
            )
            usage = payload.get("usage") or {}
            hit = int(usage.get("prompt_cache_hit_tokens") or 0)
            miss = int(usage.get("prompt_cache_miss_tokens") or 0)
            cache_eligible = hit + miss >= args.min_cache_tokens
            validations.append(
                {
                    "method": row["method"],
                    "system_prompt_hash": row["system_prompt_hash"],
                    "system_prompt_chars": row["system_prompt_chars"],
                    "cache_hit_tokens": hit,
                    "cache_miss_tokens": miss,
                    "cache_eligible": cache_eligible,
                    "valid": hit > 0 if cache_eligible else True,
                }
            )

    result = {
        "protocol_id": "deepseek_system_prompt_prime_v1",
        "model": args.model,
        "isolation_prefix": args.isolation_prefix,
        "min_system_prompt_chars": args.min_chars,
        "min_cache_tokens": args.min_cache_tokens,
        "templates": validations,
        "valid": all(row["valid"] for row in validations),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if not result["valid"]:
        failed = [row["system_prompt_hash"] for row in validations if not row["valid"]]
        raise SystemExit(f"cache priming validation failed for {len(failed)} templates")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--isolation-prefix", required=True)
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    parser.add_argument("--min-chars", type=int, default=256)
    parser.add_argument("--min-cache-tokens", type=int, default=128)
    parser.add_argument("--settle-seconds", type=float, default=20)
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
