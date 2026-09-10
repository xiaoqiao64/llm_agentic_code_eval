from __future__ import annotations

import argparse
import time
from pathlib import Path

from rich.console import Console
from rich.table import Table

from agent_eval.config import load_config
from agent_eval.harness.runner import TaskRunner
from agent_eval.llm.client import LLMClient
from agent_eval.report.reporter import EvalReport, now_iso, write_matrix_summary, write_report
from agent_eval.tasks.registry import load_tasks

console = Console()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agent-eval",
        description="Fast agentic coding evaluation for local OpenAI-compatible LLMs",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Run evaluation")
    run.add_argument("--config", type=Path, default=None)
    run.add_argument("--base-url", default=None)
    run.add_argument("--model", default=None)
    run.add_argument("--api-key", default=None)
    run.add_argument("--thinking-preset", default=None)
    run.add_argument("--reasoning-effort", default=None)
    run.add_argument("--tool-mode", choices=["text", "openai"], default=None)
    run.add_argument("--profile", default=None)
    run.add_argument("--tasks", default=None, help="Comma-separated task ids")
    run.add_argument("--task-timeout", type=int, default=None)
    run.add_argument("--max-turns", type=int, default=None)
    run.add_argument("--output", type=Path, default=None)
    run.add_argument("--seed", type=int, default=None)

    matrix = sub.add_parser("matrix", help="Run multiple thinking presets")
    matrix.add_argument("--config", type=Path, default=None)
    matrix.add_argument("--base-url", default=None)
    matrix.add_argument("--model", default=None)
    matrix.add_argument("--api-key", default=None)
    matrix.add_argument("--presets", required=True, help="Comma-separated presets")
    matrix.add_argument("--tool-mode", choices=["text", "openai"], default=None)
    matrix.add_argument("--profile", default=None)
    matrix.add_argument("--tasks", default=None)
    matrix.add_argument("--task-timeout", type=int, default=None)
    matrix.add_argument("--max-turns", type=int, default=None)
    matrix.add_argument("--output", type=Path, default=Path("results/matrix"))
    matrix.add_argument("--seed", type=int, default=None)

    sub.add_parser("list-tasks", help="List available tasks")
    return parser


def _parse_overrides(args: argparse.Namespace) -> dict:
    overrides: dict = {}
    if args.base_url:
        overrides["base_url"] = args.base_url
    if args.model:
        overrides["model"] = args.model
    if getattr(args, "api_key", None):
        overrides["api_key"] = args.api_key
    if getattr(args, "thinking_preset", None):
        overrides["thinking_preset"] = args.thinking_preset
    if getattr(args, "reasoning_effort", None):
        overrides["reasoning_effort"] = args.reasoning_effort
    if getattr(args, "tool_mode", None):
        overrides["tool_mode"] = args.tool_mode
    if getattr(args, "profile", None):
        overrides["profile"] = args.profile
    if getattr(args, "tasks", None):
        overrides["task_ids"] = [t.strip() for t in args.tasks.split(",") if t.strip()]
    if getattr(args, "task_timeout", None):
        overrides["task_timeout_sec"] = args.task_timeout
    if getattr(args, "max_turns", None):
        overrides["max_turns"] = args.max_turns
    if getattr(args, "output", None) and args.command == "run":
        overrides["output_dir"] = args.output
    if getattr(args, "seed", None) is not None:
        overrides["seed"] = args.seed
    return overrides


def _run_eval(config_path: Path | None, overrides: dict, output_dir: Path) -> EvalReport:
    config = load_config(config_path, overrides)
    task_ids = config.resolved_task_ids()
    tasks = load_tasks(config.tasks_dir, task_ids)

    llm = LLMClient(config)
    thinking_label = llm.thinking_label()

    trajectories_dir = output_dir / "trajectories"
    runner = TaskRunner(config, trajectories_dir=trajectories_dir)

    console.print(
        f"[bold]Running {len(tasks)} tasks[/bold] "
        f"(model={config.model}, {thinking_label})"
    )

    started_at = now_iso()
    start = time.perf_counter()
    results = []

    for i, task in enumerate(tasks, 1):
        task_config = config
        if task.timeout_sec != config.task_timeout_sec or task.max_turns != config.max_turns:
            overrides_task = {
                "task_timeout_sec": task.timeout_sec,
                "max_turns": task.max_turns,
            }
            task_config = load_config(config_path, {**overrides, **overrides_task})

        runner_task = TaskRunner(task_config, trajectories_dir=trajectories_dir)
        console.print(f"[cyan]({i}/{len(tasks)})[/cyan] {task.task_id} ...", end=" ")
        result = runner_task.run_task(task)
        results.append(result)
        status = "[green]PASS[/green]" if result.passed else "[red]FAIL[/red]"
        console.print(f"{status} ({result.total_elapsed_sec:.1f}s, {result.agent_turns} turns)")

    total_sec = time.perf_counter() - start
    pass_count = sum(1 for r in results if r.passed)

    report = EvalReport(
        model=config.model,
        base_url=config.base_url,
        thinking_label=thinking_label,
        profile=config.profile,
        task_ids=task_ids,
        started_at=started_at,
        total_sec=total_sec,
        pass_count=pass_count,
        total_count=len(results),
        pass_rate=pass_count / len(results) if results else 0.0,
        results=results,
        matrix_label=overrides.get("thinking_preset"),
    )
    json_path, md_path = write_report(report, output_dir)
    console.print(f"\n[bold]Report:[/bold] {md_path} ({json_path})")
    return report


def cmd_run(args: argparse.Namespace) -> None:
    overrides = _parse_overrides(args)
    output_dir = args.output or Path("results") / time.strftime("%Y%m%d_%H%M%S")
    _run_eval(args.config, overrides, output_dir)


def cmd_matrix(args: argparse.Namespace) -> None:
    presets = [p.strip() for p in args.presets.split(",") if p.strip()]
    reports: list[EvalReport] = []
    for preset in presets:
        overrides = _parse_overrides(args)
        overrides["thinking_preset"] = preset
        preset_dir = args.output / preset
        console.rule(f"[bold]Preset: {preset}[/bold]")
        reports.append(_run_eval(args.config, overrides, preset_dir))
    matrix_path = write_matrix_summary(reports, args.output)
    console.print(f"\n[bold]Matrix summary:[/bold] {matrix_path}")


def cmd_list_tasks(args: argparse.Namespace) -> None:
    from agent_eval.tasks.registry import list_all_tasks

    tasks = list_all_tasks()
    table = Table(title="Available Tasks")
    table.add_column("Task ID")
    for t in tasks:
        table.add_row(t)
    console.print(table)


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    if args.command == "run":
        cmd_run(args)
    elif args.command == "matrix":
        cmd_matrix(args)
    elif args.command == "list-tasks":
        cmd_list_tasks(args)


if __name__ == "__main__":
    main()
