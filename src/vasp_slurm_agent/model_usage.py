"""Provider-reported token counts and local request measurements."""

from __future__ import annotations

import json
from typing import Any
from uuid import uuid4

from .providers import PROVIDERS, model_label


class ModelResponse(str):
    """A normal response string with per-call usage attached."""

    def __new__(cls, text: str, usage: dict[str, Any]):
        result = super().__new__(cls, text)
        result.usage = dict(usage)
        return result


def _field(value, key):
    return value.get(key) if isinstance(value, dict) else getattr(value, key, None)


def _count(value):
    return value if type(value) is int and value >= 0 else None


def normalize_usage(value, provider: str) -> dict[str, int | str | None]:
    """Cache and reasoning counts are subsets, never extra tokens."""
    if provider in {"anthropic", "claude_cli"}:
        fresh, outputs = _count(_field(value, "input_tokens")), _field(value, "output_tokens")
        cached, written = _field(value, "cache_read_input_tokens"), _field(value, "cache_creation_input_tokens")
        # Claude reports fresh input separately from cache reads and writes.
        parts = [fresh, _count(cached) if cached is not None else 0,
                 _count(written) if written is not None else 0]
        inputs = sum(parts) if all(part is not None for part in parts) else None
        reasoning = None
    elif PROVIDERS.get(provider, {}).get("protocol") == "chat_completions":
        inputs, outputs = _field(value, "prompt_tokens"), _field(value, "completion_tokens")
        cached = _field(_field(value, "prompt_tokens_details"), "cached_tokens")
        if cached is None and provider == "deepseek":
            cached = _field(value, "prompt_cache_hit_tokens")
        reasoning = _field(_field(value, "completion_tokens_details"), "reasoning_tokens")
        if provider == "grok":
            # xAI reports visible completion tokens separately from reasoning.
            reported_total, prompt, visible = map(_count, (_field(value, "total_tokens"), inputs, outputs))
            if reported_total is not None and prompt is not None and reported_total >= prompt + (visible or 0):
                outputs = reported_total - prompt
            elif visible is not None and _count(reasoning) is not None:
                outputs = visible + reasoning
    else:
        inputs, outputs = _field(value, "input_tokens"), _field(value, "output_tokens")
        cached = (_field(value, "cached_input_tokens") if provider == "codex"
                  else _field(_field(value, "input_tokens_details"), "cached_tokens"))
        reasoning = (_field(value, "reasoning_output_tokens") if provider == "codex"
                     else _field(_field(value, "output_tokens_details"), "reasoning_tokens"))
    inputs, outputs, cached, reasoning = map(_count, (inputs, outputs, cached, reasoning))
    if inputs is not None and cached is not None and cached > inputs:
        cached = None
    if outputs is not None and reasoning is not None and reasoning > outputs:
        reasoning = None
    total = inputs + outputs if inputs is not None and outputs is not None else _count(_field(value, "total_tokens"))
    return {"input_tokens": inputs, "output_tokens": outputs, "cached_input_tokens": cached,
            "reasoning_tokens": reasoning, "total_tokens": total,
            "token_counts_source": "provider" if any(item is not None for item in (inputs, outputs, cached, reasoning, total)) else "unavailable"}


def codex_usage(events: str):
    """Read the final turn report, not item events or cumulative duplicates."""
    usage = None
    for line in events.splitlines():
        try:
            item = json.loads(line)
        except (ValueError, TypeError):
            continue
        if isinstance(item, dict) and item.get("type") == "turn.completed":
            usage = item.get("usage")
    return usage


def usage_record(provider: str, model: str, *, reported=None, latency_seconds=None,
                 payload: str | None = None, instructions: str | None = None,
                 schema: dict | None = None, output: str | None = None,
                 max_output_tokens: int | None = None) -> dict[str, Any]:
    return {**normalize_usage(reported, provider), "call_id": uuid4().hex,
            "provider": provider, "model": model_label(provider, model),
            "latency_seconds": round(latency_seconds, 6) if latency_seconds is not None else None,
            "input_characters": len(payload) if payload is not None else None,
            "instruction_characters": len(instructions) if instructions is not None else None,
            "schema_characters": len(json.dumps(schema, ensure_ascii=False, separators=(",", ":"))) if schema is not None else None,
            "output_characters": len(output) if isinstance(output, str) else None,
            "max_output_tokens": max_output_tokens}


def response_usage(response, settings) -> dict[str, Any]:
    usage = getattr(response, "usage", None)
    if isinstance(usage, dict):
        return dict(usage)
    return usage_record(settings.provider, settings.model, output=response)


def response_provenance(response, settings) -> dict[str, str]:
    """Use the connection resolved for this call, not an earlier settings view."""
    usage = getattr(response, "usage", None)
    if isinstance(usage, dict) and all(isinstance(usage.get(key), str) and usage[key]
                                       for key in ("provider", "model")):
        return {key: usage[key] for key in ("provider", "model")}
    return {"provider": settings.provider, "model": model_label(settings.provider, settings.model)}
