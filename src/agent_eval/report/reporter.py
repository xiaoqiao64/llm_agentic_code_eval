from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent_eval.harness.runner import TaskResult


@dataclass
class EvalReport:
    model: str
    base_url: str
    thinking_label: str
    profile: str
    task_ids: list[str]
    started_at: str
    total_sec: float
    pass_count: int
    total_count: int
    pass_rate: float
    results: list[TaskResult] = field(default_factory=list)
    matrix_label: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["results"] = [asdict(r) for r in self.results]
        return data


def write_report(report: EvalReport, output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "summary.json"
    md_path = output_dir / "summary.md"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)

    md_path.write_text(_render_markdown(report), encoding="utf-8")
    return json_path, md_path


def _render_markdown(report: EvalReport) -> str:
    lines = [
        "# Agent Eval Report",
        "",
        f"- **Model**: {report.model}",
        f"- **Base URL**: {report.base_url}",
        f"- **Thinking**: {report.thinking_label}",
        f"- **Profile**: {report.profile}",
        f"- **Started**: {report.started_at}",
        f"- **Pass rate**: {report.pass_count}/{report.total_count} ({report.pass_rate:.0%})",
        f"- **Total time**: {_fmt_sec(report.total_sec)}",
        "",
        "| Task | Pass | Time | Turns | Reasoning tokens | Stop reason |",
        "|------|------|------|-------|------------------|-------------|",
    ]
    for r in report.results:
        status = "✓" if r.passed else "✗"
        lines.append(
            f"| {r.task_id} | {status} | {_fmt_sec(r.total_elapsed_sec)} | "
            f"{r.agent_turns} | {r.total_reasoning_tokens} | {r.stop_reason} |"
        )
    lines.append(
        f"| **Total** | **{report.pass_count}/{report.total_count}** | "
        f"**{_fmt_sec(report.total_sec)}** | | | |"
    )
    lines.append("")
    return "\n".join(lines)


def write_matrix_summary(reports: list[EvalReport], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "matrix_summary.md"
    lines = [
        "# Matrix Comparison",
        "",
        "| Preset | Pass rate | Total time | Pass count |",
        "|--------|-----------|------------|------------|",
    ]
    for r in reports:
        label = r.matrix_label or r.thinking_label
        lines.append(
            f"| {label} | {r.pass_rate:.0%} | {_fmt_sec(r.total_sec)} | "
            f"{r.pass_count}/{r.total_count} |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _fmt_sec(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes}m{secs:02d}s"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
