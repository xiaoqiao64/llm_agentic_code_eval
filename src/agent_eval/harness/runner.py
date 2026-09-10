from __future__ import annotations

import json
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agent_eval.agent.loop import AgentLoop, AgentRunResult
from agent_eval.agent.tools import ToolExecutor
from agent_eval.config import EvalConfig
from agent_eval.harness.workspace import cleanup_workspace, create_task_workspace
from agent_eval.llm.client import LLMClient
from agent_eval.tasks.registry import TaskDefinition


@dataclass
class TaskResult:
    task_id: str
    passed: bool
    agent_turns: int
    agent_elapsed_sec: float
    verify_elapsed_sec: float
    total_elapsed_sec: float
    stop_reason: str
    verify_output: str
    error: str | None = None
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_reasoning_tokens: int = 0
    trajectory: list[dict[str, Any]] = field(default_factory=list)


class TaskRunner:
    def __init__(self, config: EvalConfig, trajectories_dir: Path | None = None) -> None:
        self.config = config
        self.trajectories_dir = trajectories_dir

    def run_task(self, task: TaskDefinition) -> TaskResult:
        start = time.perf_counter()
        workspace = create_task_workspace(task.path)
        error: str | None = None
        agent_result: AgentRunResult | None = None
        verify_output = ""
        verify_elapsed = 0.0
        passed = False

        try:
            llm = LLMClient(self.config)
            executor = ToolExecutor(
                workspace,
                command_timeout_sec=min(60, self.config.task_timeout_sec),
            )
            deadline = start + self.config.task_timeout_sec
            agent = AgentLoop(self.config, llm, executor, task.prompt)
            agent_result = agent.run(deadline=deadline)

            verify_start = time.perf_counter()
            passed, verify_output = self._verify(task, workspace)
            verify_elapsed = time.perf_counter() - verify_start

            if self.trajectories_dir:
                self._save_trajectory(task.task_id, agent_result)

        except Exception as exc:  # noqa: BLE001
            error = str(exc)
        finally:
            cleanup_workspace(workspace)

        total_elapsed = time.perf_counter() - start
        return TaskResult(
            task_id=task.task_id,
            passed=passed and error is None,
            agent_turns=agent_result.turns if agent_result else 0,
            agent_elapsed_sec=agent_result.elapsed_sec if agent_result else 0.0,
            verify_elapsed_sec=verify_elapsed,
            total_elapsed_sec=total_elapsed,
            stop_reason=agent_result.stop_reason if agent_result else "error",
            verify_output=verify_output,
            error=error,
            total_prompt_tokens=agent_result.total_prompt_tokens if agent_result else 0,
            total_completion_tokens=agent_result.total_completion_tokens if agent_result else 0,
            total_reasoning_tokens=agent_result.total_reasoning_tokens if agent_result else 0,
            trajectory=AgentLoop.trajectory_to_dict(agent_result.trajectory)
            if agent_result
            else [],
        )

    def _verify(self, task: TaskDefinition, workspace: Path) -> tuple[bool, str]:
        verify_dir = task.path / "verify"
        import os

        env = os.environ.copy()
        env["PYTHONPATH"] = str(workspace)

        if verify_dir.exists():
            proc = subprocess.run(
                ["pytest", str(verify_dir), "-q"],
                cwd=workspace,
                capture_output=True,
                text=True,
                timeout=self.config.task_timeout_sec,
                env=env,
            )
        else:
            proc = subprocess.run(
                task.verify_cmd,
                shell=True,
                cwd=workspace,
                capture_output=True,
                text=True,
                timeout=self.config.task_timeout_sec,
                env=env,
            )

        output = (proc.stdout or "") + (proc.stderr or "")
        return proc.returncode == 0, output

    def _save_trajectory(self, task_id: str, agent_result: AgentRunResult) -> None:
        assert self.trajectories_dir is not None
        self.trajectories_dir.mkdir(parents=True, exist_ok=True)
        path = self.trajectories_dir / f"{task_id}.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(
                AgentLoop.trajectory_to_dict(agent_result.trajectory),
                f,
                indent=2,
                ensure_ascii=False,
            )
