"""OpenAI-compatible chat and embedding clients for vNext."""

from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING, Any

from openai import OpenAI

from src.config.api import ChatCompletionConfig, EmbeddingConfig
from src.extraction.pipeline import JsonExtractionBackend

if TYPE_CHECKING:
    from src.logger.api_log import ApiCallLogger


class OpenAIChatClient:
    """Thin wrapper over OpenAI-compatible chat completions."""

    def __init__(
        self,
        config: ChatCompletionConfig,
        *,
        client: OpenAI | None = None,
        api_logger: "ApiCallLogger | None" = None,
    ) -> None:
        self.config = config
        self.client = client or OpenAI(
            base_url=str(config.url).replace("://0.0.0.0", "://127.0.0.1"),
            api_key=config.key,
        )
        self._api_logger = api_logger

    def generate_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        model_name: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        timeout: float | None = None,
        step_label: str = "llm",
        trace: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        used_model = model_name or self.config.model_name
        payload = {
            "model": used_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.config.temperature if temperature is None else temperature,
            "max_tokens": self.config.max_token if max_tokens is None else max_tokens,
            "top_p": self.config.top_p if top_p is None else top_p,
            "response_format": {"type": "json_object"},
        }
        if self.config.topk is not None:
            payload["extra_body"] = {"top_k": self.config.topk}

        start_time = time.time()
        create_kwargs: dict[str, Any] = {"timeout": timeout} if timeout is not None else {}
        response = self.client.chat.completions.create(**payload, **create_kwargs)
        end_time = time.time()

        content = response.choices[0].message.content or "{}"
        usage = getattr(response, "usage", None)
        prompt_tokens = usage.prompt_tokens if usage else -1
        completion_tokens = usage.completion_tokens if usage else -1

        if self._api_logger is not None:
            self._api_logger.log(
                step_label=step_label,
                model_name=used_model,
                request_id=None if trace is None else str(trace.get("request_id") or ""),
                session_id=None if trace is None else str(trace.get("session_id") or ""),
                cell_id=None if trace is None else str(trace.get("cell_id") or ""),
                cell_index=None if trace is None else _maybe_int(trace.get("cell_index")),
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                response_content=content,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                start_time=start_time,
                end_time=end_time,
            )

        return _parse_json_object(content)


class OpenAIEmbeddingClient:
    """Thin wrapper over OpenAI-compatible embeddings."""

    def __init__(self, config: EmbeddingConfig, *, client: OpenAI | None = None) -> None:
        self.config = config
        self.client = client or OpenAI(
            base_url=str(config.url).replace("://0.0.0.0", "://127.0.0.1"),
            api_key=config.key,
        )

    def embed_texts(self, texts: list[str], *, batch_size: int = 256) -> list[list[float]]:
        if not texts:
            return []
        if len(texts) <= batch_size:
            response = self.client.embeddings.create(
                model=self.config.model_name,
                input=texts,
            )
            return [list(item.embedding) for item in response.data]
        # Batch to avoid overwhelming the embedding server
        all_embeddings: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            response = self.client.embeddings.create(
                model=self.config.model_name,
                input=batch,
            )
            all_embeddings.extend(list(item.embedding) for item in response.data)
        return all_embeddings


class OpenAIJsonBackend(JsonExtractionBackend):
    """Extraction backend backed by an OpenAI-compatible chat client."""

    def __init__(
        self,
        chat_client: OpenAIChatClient,
        *,
        model_name: str,
        temperature: float,
        max_tokens: int,
        top_p: float,
        step_label: str = "extraction",
    ) -> None:
        self.chat_client = chat_client
        self.model_name = model_name
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.top_p = float(top_p)
        self.step_label = step_label

    def generate_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        trace: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self.chat_client.generate_json(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model_name=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
            step_label=self.step_label,
            trace=trace,
        )


def _parse_json_object(text: str) -> dict[str, Any]:
    raw = str(text or "").strip()
    if raw.startswith("```"):
        lines = raw.splitlines()
        if len(lines) >= 3 and lines[-1].strip().startswith("```"):
            raw = "\n".join(lines[1:-1]).strip()
            if raw.startswith("json"):
                raw = raw[4:].strip()
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start < 0 or end <= start:
            # Truncated response with no closing brace — try summary extraction
            return _extract_summary_from_truncated(raw)
        try:
            parsed = json.loads(raw[start : end + 1])
        except json.JSONDecodeError:
            return _extract_summary_from_truncated(raw)
    if not isinstance(parsed, dict):
        return {}
    # If parsed dict has no "summary" key but raw has a partial "summary" value,
    # attempt extraction (e.g. model wrote summary mid-object then got cut off)
    if "summary" not in parsed:
        fallback = _extract_summary_from_truncated(raw)
        if fallback.get("summary"):
            return fallback
    return parsed


def _extract_summary_from_truncated(raw: str) -> dict[str, Any]:
    """Last-resort extraction of a 'summary' value from truncated JSON text.

    Handles the common failure where the model writes:
        {"summary": "long prose that gets cut off without closing quote/brace"}
    """
    import re
    # Find: "summary": "..."  allowing for truncation before closing quote
    m = re.search(r'"summary"\s*:\s*"(.*)', raw, re.DOTALL)
    if not m:
        return {}
    raw_value = m.group(1)
    # Try to find the end of the string value (unescaped closing quote)
    end = _find_json_string_end(raw_value)
    if end >= 0:
        value = raw_value[:end]
    else:
        # Truncated before closing quote — use whatever text we have,
        # strip trailing incomplete word after last sentence-ending punctuation
        value = raw_value.rstrip()
        last_stop = max(value.rfind("."), value.rfind("!"), value.rfind("?"))
        if last_stop > len(value) // 2:
            value = value[: last_stop + 1]
    value = value.replace('\\"', '"').replace("\\n", " ").strip()
    return {"summary": value} if value else {}


def _find_json_string_end(s: str) -> int:
    """Return index of the closing unescaped double-quote in a JSON string body."""
    i = 0
    while i < len(s):
        if s[i] == "\\" and i + 1 < len(s):
            i += 2  # skip escape sequence
        elif s[i] == '"':
            return i
        else:
            i += 1
    return -1  # not found (truncated)


def _maybe_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
