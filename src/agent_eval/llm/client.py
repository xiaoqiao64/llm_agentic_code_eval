from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from openai import OpenAI

from agent_eval.config import EvalConfig, deep_merge

_api_log_lock = threading.Lock()
_api_log_counter = 0


def _next_api_log_index() -> int:
    global _api_log_counter
    with _api_log_lock:
        _api_log_counter += 1
        return _api_log_counter


def _response_to_dict(response: Any) -> dict[str, Any]:
    if hasattr(response, "model_dump"):
        return response.model_dump()
    return {"repr": repr(response)}


def append_api_call_log(
    path: Path,
    request_kwargs: dict[str, Any],
    response: Any | None,
    latency_sec: float,
    error: str | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    entry: dict[str, Any] = {
        "call_index": _next_api_log_index(),
        "timestamp": datetime.now(UTC).isoformat(),
        "latency_sec": latency_sec,
        "request": request_kwargs,
    }
    if error is not None:
        entry["error"] = error
    elif response is not None:
        entry["response"] = _response_to_dict(response)
    with _api_log_lock:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False, indent=2))
            f.write("\n\n")
            f.flush()
            os.fsync(f.fileno())


def reasoning_tokens_from_usage(usage: Any | None) -> int:
    if usage is None:
        return 0
    details = getattr(usage, "completion_tokens_details", None)
    if details:
        n = getattr(details, "reasoning_tokens", None)
        if n:
            return int(n)
    n = getattr(usage, "reasoning_tokens", None)
    if n:
        return int(n)
    if hasattr(usage, "model_dump"):
        data = usage.model_dump()
        if data.get("reasoning_tokens"):
            return int(data["reasoning_tokens"])
        ctd = data.get("completion_tokens_details") or {}
        if isinstance(ctd, dict) and ctd.get("reasoning_tokens"):
            return int(ctd["reasoning_tokens"])
    return 0


def estimate_reasoning_tokens(reasoning_content: str) -> int:
    """Fallback when the API returns reasoning_content but usage.reasoning_tokens is 0."""
    text = reasoning_content.strip()
    if not text:
        return 0
    return max(1, (len(text.encode("utf-8")) + 3) // 4)


@dataclass
class ChatMetrics:
    latency_sec: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    total_tokens: int = 0


@dataclass
class ChatResult:
    content: str
    reasoning_content: str = ""
    metrics: ChatMetrics = field(default_factory=ChatMetrics)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)


class LLMClient:
    def __init__(self, config: EvalConfig) -> None:
        self.config = config
        self.client = OpenAI(
            base_url=config.base_url,
            api_key=config.api_key,
        )
        self._print_api_call_file = config.print_api_call_file

    def _build_request_kwargs(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        if self.config.seed is not None:
            kwargs["seed"] = self.config.seed

        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        if self.config.request_kwargs:
            kwargs = deep_merge(kwargs, self.config.request_kwargs)

        return kwargs

    def preview_request_kwargs(
        self,
        *,
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Merged chat.completions kwargs (messages/tools bodies shortened for display)."""
        kwargs = self._build_request_kwargs(
            [{"role": "user", "content": "<omitted>"}],
            tools=tools,
        )
        preview: dict[str, Any] = dict(kwargs)
        preview["messages"] = "<omitted; conversation grows each turn>"
        if tools is not None and "tools" in preview:
            preview["tools"] = f"<{len(tools)} OpenAI function specs>"
        return preview

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> ChatResult:
        kwargs = self._build_request_kwargs(messages, tools)
        start = time.perf_counter()
        try:
            response = self.client.chat.completions.create(**kwargs)
        except Exception as exc:
            latency = time.perf_counter() - start
            if self._print_api_call_file is not None:
                append_api_call_log(
                    self._print_api_call_file,
                    kwargs,
                    None,
                    latency,
                    error=str(exc),
                )
            raise
        latency = time.perf_counter() - start

        if self._print_api_call_file is not None:
            append_api_call_log(
                self._print_api_call_file, kwargs, response, latency
            )

        choice = response.choices[0].message
        content = choice.content or ""
        reasoning_content = getattr(choice, "reasoning_content", None) or ""

        metrics = ChatMetrics(latency_sec=latency)
        if response.usage:
            metrics.prompt_tokens = response.usage.prompt_tokens or 0
            metrics.completion_tokens = response.usage.completion_tokens or 0
            metrics.total_tokens = response.usage.total_tokens or 0
            metrics.reasoning_tokens = reasoning_tokens_from_usage(response.usage)
        if metrics.reasoning_tokens == 0 and reasoning_content.strip():
            metrics.reasoning_tokens = estimate_reasoning_tokens(reasoning_content)

        tool_calls: list[dict[str, Any]] = []
        if choice.tool_calls:
            for tc in choice.tool_calls:
                tool_calls.append(
                    {
                        "id": tc.id,
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    }
                )

        return ChatResult(
            content=content,
            reasoning_content=reasoning_content,
            metrics=metrics,
            tool_calls=tool_calls,
        )

    def thinking_label(self) -> str:
        kw = self._build_request_kwargs([{"role": "user", "content": ""}])
        parts: list[str] = []
        if "reasoning_effort" in kw:
            parts.append(f"reasoning_effort={kw['reasoning_effort']}")
        extra = kw.get("extra_body") or {}
        if budget := extra.get("thinking_token_budget"):
            parts.append(f"thinking_token_budget={budget}")
        thinking = extra.get("thinking")
        if isinstance(thinking, dict) and thinking.get("type"):
            parts.append(f"thinking.type={thinking['type']}")
        ctk = extra.get("chat_template_kwargs") or {}
        if "enable_thinking" in ctk:
            parts.append(f"enable_thinking={ctk['enable_thinking']}")
        return ", ".join(parts) if parts else "default"
