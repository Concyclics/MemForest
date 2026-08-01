#!/usr/bin/env python3
"""OpenAI-compatible DeepSeek proxy that records billable token classes."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, Request, Response


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_app(
    upstream: str,
    log_path: Path,
    api_key: str,
    model: str,
    isolation_prefix: str,
) -> FastAPI:
    app = FastAPI()
    app.state.upstream = upstream.rstrip("/")
    app.state.log_path = log_path
    app.state.api_key = api_key
    app.state.model = model
    app.state.isolation_prefix = isolation_prefix
    app.state.method = "unassigned"
    app.state.lock = asyncio.Lock()
    app.state.client = None

    @app.on_event("startup")
    async def startup() -> None:
        limits = httpx.Limits(max_connections=512, max_keepalive_connections=256)
        app.state.client = httpx.AsyncClient(timeout=1800.0, limits=limits)
        log_path.parent.mkdir(parents=True, exist_ok=True)

    @app.on_event("shutdown")
    async def shutdown() -> None:
        if app.state.client is not None:
            await app.state.client.aclose()

    async def append(row: dict[str, Any]) -> None:
        line = json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n"
        async with app.state.lock:
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(line)

    @app.get("/control/status")
    async def status() -> dict[str, str]:
        return {"status": "ok", "method": app.state.method, "model": app.state.model}

    @app.post("/control/method/{method}")
    async def set_method(method: str) -> dict[str, str]:
        app.state.method = method
        await append({"record_type": "phase", "method": method, "time": utc_now()})
        return {"status": "ok", "method": method}

    @app.api_route(
        "/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"]
    )
    async def proxy(path: str, request: Request) -> Response:
        body = await request.body()
        request_json: Any = None
        try:
            request_json = json.loads(body) if body else None
        except json.JSONDecodeError:
            pass

        if isinstance(request_json, dict) and path.rstrip("/") == "v1/chat/completions":
            request_json["model"] = app.state.model
            request_json["thinking"] = {"type": "disabled"}
            request_json["user_id"] = (
                f"{app.state.isolation_prefix}-{app.state.method}"[:512]
            )
            request_json.pop("reasoning_effort", None)
            request_json.pop("top_k", None)
            body = json.dumps(request_json, ensure_ascii=False).encode("utf-8")

        headers = {
            key: value
            for key, value in request.headers.items()
            if key.lower() not in {"host", "content-length", "connection", "authorization"}
        }
        headers["authorization"] = f"Bearer {app.state.api_key}"
        started = utc_now()
        response = await app.state.client.request(
            request.method,
            f"{app.state.upstream}/{path}",
            params=request.query_params,
            content=body,
            headers=headers,
        )
        response_headers = {
            key: value
            for key, value in response.headers.items()
            if key.lower()
            not in {"content-length", "content-encoding", "transfer-encoding", "connection"}
        }
        response_json: Any = None
        try:
            response_json = response.json()
        except (json.JSONDecodeError, ValueError):
            pass

        usage = response_json.get("usage") if isinstance(response_json, dict) else None
        usage = usage if isinstance(usage, dict) else {}
        prompt = usage.get("prompt_tokens")
        hit = usage.get("prompt_cache_hit_tokens")
        miss = usage.get("prompt_cache_miss_tokens")
        output = usage.get("completion_tokens")
        prompt_value = (
            request_json.get("messages", request_json.get("prompt"))
            if isinstance(request_json, dict)
            else None
        )
        await append(
            {
                "record_type": "request",
                "method": app.state.method,
                "time": started,
                "path": f"/{path}",
                "status_code": response.status_code,
                "model": app.state.model,
                "request_hash": stable_hash(request_json) if request_json is not None else None,
                "prompt_hash": stable_hash(prompt_value) if prompt_value is not None else None,
                "prompt_tokens": prompt,
                "prompt_cache_hit_tokens": hit,
                "prompt_cache_miss_tokens": miss,
                "completion_tokens": output,
                "total_tokens": usage.get("total_tokens"),
                "usage_available": bool(usage),
                "error": response_json.get("error")
                if response.status_code >= 400 and isinstance(response_json, dict)
                else None,
            }
        )
        return Response(
            content=response.content,
            status_code=response.status_code,
            headers=response_headers,
            media_type=response.headers.get("content-type"),
        )

    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", default="https://api.deepseek.com")
    parser.add_argument("--log-path", type=Path, required=True)
    parser.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument("--isolation-prefix", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18001)
    args = parser.parse_args()
    api_key = os.getenv(args.api_key_env)
    if not api_key:
        raise SystemExit(f"{args.api_key_env} is required")

    import uvicorn

    uvicorn.run(
        build_app(args.upstream, args.log_path, api_key, args.model, args.isolation_prefix),
        host=args.host,
        port=args.port,
    )


if __name__ == "__main__":
    main()
