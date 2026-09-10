from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "default.yaml"
TASKS_DIR = PROJECT_ROOT / "tasks"


def _parse_kwarg_value(raw: str) -> Any:
    lower = raw.lower()
    if lower == "true":
        return True
    if lower == "false":
        return False
    if lower in ("null", "none"):
        return None
    try:
        if any(ch in raw for ch in ".eE"):
            return float(raw)
        return int(raw)
    except ValueError:
        return raw


def _deep_set(root: dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    cur = root
    for part in parts[:-1]:
        nxt = cur.get(part)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[part] = nxt
        cur = nxt
    cur[parts[-1]] = value


def parse_dot_kwargs(pairs: list[str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for item in pairs:
        if "=" not in item:
            raise ValueError(
                f"Invalid --kwargs entry: {item!r} (expected KEY=VALUE)"
            )
        key, _, val = item.partition("=")
        key = key.strip()
        if not key:
            raise ValueError(
                f"Invalid --kwargs entry: {item!r} (expected KEY=VALUE)"
            )
        _deep_set(result, key, _parse_kwarg_value(val.strip()))
    return result


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


@dataclass
class EvalConfig:
    base_url: str = "http://localhost:8080/v1"
    model: str = "Qwen3.8-27B"
    api_key: str = "EMPTY"
    tool_mode: str = "text"
    profile: str = "quick"
    task_ids: list[str] | None = None
    task_timeout_sec: int = 240
    max_turns: int = 20
    temperature: float = 0.2
    max_tokens: int = 4096
    seed: int | None = None
    output_dir: Path = Path("results")
    profiles: dict[str, list[str]] = field(default_factory=dict)
    tasks_dir: Path = TASKS_DIR
    request_kwargs: dict[str, Any] = field(default_factory=dict)
    print_api_call_file: Path | None = None

    def resolved_task_ids(self) -> list[str]:
        if self.task_ids:
            return self.task_ids
        if self.profile not in self.profiles:
            raise ValueError(f"Unknown profile: {self.profile}")
        return self.profiles[self.profile]


def load_config(
    config_path: Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> EvalConfig:
    path = config_path or DEFAULT_CONFIG_PATH
    data: dict[str, Any] = {}
    if path.exists():
        with path.open(encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

    request_kwargs: dict[str, Any] = dict(data.get("request_kwargs") or {})
    if overrides:
        ovs = dict(overrides)
        override_request = ovs.pop("request_kwargs", None)
        data.update({k: v for k, v in ovs.items() if v is not None})
        if override_request:
            request_kwargs = deep_merge(request_kwargs, override_request)

    return EvalConfig(
        base_url=data.get("base_url", "http://localhost:8080/v1"),
        model=data.get("model", "Qwen3.8-27B"),
        api_key=data.get("api_key", "EMPTY"),
        tool_mode=data.get("tool_mode", "text"),
        profile=data.get("profile", "quick"),
        task_ids=data.get("task_ids"),
        task_timeout_sec=int(data.get("task_timeout_sec", 240)),
        max_turns=int(data.get("max_turns", 20)),
        temperature=float(data.get("temperature", 0.2)),
        max_tokens=int(data.get("max_tokens", 4096)),
        seed=data.get("seed"),
        output_dir=Path(data.get("output_dir", "results")),
        profiles=data.get("profiles", {}),
        tasks_dir=Path(data.get("tasks_dir", TASKS_DIR)),
        request_kwargs=request_kwargs,
        print_api_call_file=(
            Path(data["print_api_call_file"])
            if data.get("print_api_call_file")
            else None
        ),
    )
