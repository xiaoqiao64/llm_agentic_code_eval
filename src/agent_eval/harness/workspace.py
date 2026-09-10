from __future__ import annotations

import shutil
import tempfile
from pathlib import Path


def create_task_workspace(task_dir: Path) -> Path:
    """Copy task workspace/ into an isolated temp directory."""
    source = task_dir / "workspace"
    if not source.exists():
        raise FileNotFoundError(f"Missing workspace for task: {task_dir.name}")

    dest = Path(tempfile.mkdtemp(prefix=f"agent_eval_{task_dir.name}_"))
    shutil.copytree(source, dest, dirs_exist_ok=True)
    return dest


def cleanup_workspace(workspace: Path) -> None:
    shutil.rmtree(workspace, ignore_errors=True)
