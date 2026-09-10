from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from agent_eval.config import TASKS_DIR


@dataclass
class TaskDefinition:
    task_id: str
    prompt: str
    verify_cmd: str
    timeout_sec: int
    max_turns: int
    path: Path


def load_task(task_dir: Path) -> TaskDefinition:
    task_yaml = task_dir / "task.yaml"
    if not task_yaml.exists():
        raise FileNotFoundError(f"Missing task.yaml in {task_dir}")

    with task_yaml.open(encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    return TaskDefinition(
        task_id=task_dir.name,
        prompt=data["prompt"],
        verify_cmd=data.get("verify_cmd", "pytest verify/ -q"),
        timeout_sec=int(data.get("timeout_sec", 240)),
        max_turns=int(data.get("max_turns", 20)),
        path=task_dir,
    )


def load_tasks(tasks_dir: Path, task_ids: list[str]) -> list[TaskDefinition]:
    tasks: list[TaskDefinition] = []
    for task_id in task_ids:
        task_dir = tasks_dir / task_id
        if not task_dir.exists():
            raise FileNotFoundError(f"Task not found: {task_id}")
        tasks.append(load_task(task_dir))
    return tasks


def list_all_tasks(tasks_dir: Path = TASKS_DIR) -> list[str]:
    if not tasks_dir.exists():
        return []
    return sorted(
        d.name for d in tasks_dir.iterdir() if d.is_dir() and (d / "task.yaml").exists()
    )
