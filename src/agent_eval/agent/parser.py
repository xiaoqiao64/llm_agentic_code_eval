from __future__ import annotations

import json
import re
from typing import Any


def parse_text_tool_calls(content: str) -> list[dict[str, Any]]:
    """Parse tool calls from ```tool ... ``` blocks in model output."""
    calls: list[dict[str, Any]] = []
    pattern = re.compile(r"```tool\s*\n?(.*?)\n?```", re.DOTALL)
    for match in pattern.finditer(content):
        raw = match.group(1).strip()
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and "name" in payload:
            calls.append(
                {
                    "name": payload["name"],
                    "arguments": payload.get("arguments", {}),
                }
            )
    return calls


def parse_openai_tool_calls(tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    parsed: list[dict[str, Any]] = []
    for tc in tool_calls:
        args = tc.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {}
        parsed.append({"name": tc["name"], "arguments": args})
    return parsed
