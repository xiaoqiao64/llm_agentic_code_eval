from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "default.yaml"
TASKS_DIR = PROJECT_ROOT / "tasks"


@dataclass
class ThinkingPreset:
    reasoning_effort: str | None = None
    extra_body: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalConfig:
    base_url: str = "http://localhost:8080/v1"
    model: str = "Qwen3.8-27B"
    api_key: str = "EMPTY"
    reasoning_backend: str = "reasoning_effort"
    thinking_presets: dict[str, ThinkingPreset] = field(default_factory=dict)
    default_thinking_preset: str = "low"
    thinking_preset: str | None = None
    reasoning_effort: str | None = None
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

    def resolved_thinking(self) -> ThinkingPreset:
        if self.reasoning_effort is not None:
            return ThinkingPreset(reasoning_effort=self.reasoning_effort)
        preset_name = self.thinking_preset or self.default_thinking_preset
        if preset_name not in self.thinking_presets:
            raise ValueError(f"Unknown thinking preset: {preset_name}")
        return self.thinking_presets[preset_name]

    def resolved_task_ids(self) -> list[str]:
        if self.task_ids:
            return self.task_ids
        if self.profile not in self.profiles:
            raise ValueError(f"Unknown profile: {self.profile}")
        return self.profiles[self.profile]


def _parse_thinking_presets(raw: dict[str, Any]) -> dict[str, ThinkingPreset]:
    presets: dict[str, ThinkingPreset] = {}
    for name, value in raw.items():
        presets[name] = ThinkingPreset(
            reasoning_effort=value.get("reasoning_effort"),
            extra_body=value.get("extra_body") or {},
        )
    return presets


def load_config(
    config_path: Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> EvalConfig:
    path = config_path or DEFAULT_CONFIG_PATH
    data: dict[str, Any] = {}
    if path.exists():
        with path.open(encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

    if overrides:
        data.update({k: v for k, v in overrides.items() if v is not None})

    return EvalConfig(
        base_url=data.get("base_url", "http://localhost:8080/v1"),
        model=data.get("model", "Qwen3.8-27B"),
        api_key=data.get("api_key", "EMPTY"),
        reasoning_backend=data.get("reasoning_backend", "reasoning_effort"),
        thinking_presets=_parse_thinking_presets(data.get("thinking_presets", {})),
        default_thinking_preset=data.get("default_thinking_preset", "low"),
        thinking_preset=data.get("thinking_preset"),
        reasoning_effort=data.get("reasoning_effort"),
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
    )
