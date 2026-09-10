from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


FORBIDDEN_PATTERNS = [
    "rm -rf /",
    "rm -rf /*",
    ":(){ :|:& };:",
    "mkfs.",
    "dd if=/dev/zero",
    "> /dev/sd",
]


@dataclass
class ToolResult:
    ok: bool
    output: str


class ToolExecutor:
    def __init__(self, workspace: Path, command_timeout_sec: int = 60) -> None:
        self.workspace = workspace.resolve()
        self.command_timeout_sec = command_timeout_sec

    def execute(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        try:
            if name == "read_file":
                return self._read_file(arguments)
            if name == "write_file":
                return self._write_file(arguments)
            if name == "edit_file":
                return self._edit_file(arguments)
            if name == "run_command":
                return self._run_command(arguments)
            return ToolResult(False, f"Unknown tool: {name}")
        except Exception as exc:  # noqa: BLE001
            return ToolResult(False, f"Tool error: {exc}")

    def _resolve_path(self, rel_path: str) -> Path:
        target = (self.workspace / rel_path).resolve()
        if not str(target).startswith(str(self.workspace)):
            raise ValueError(f"Path escapes workspace: {rel_path}")
        return target

    def _read_file(self, arguments: dict[str, Any]) -> ToolResult:
        path = self._resolve_path(arguments["path"])
        if not path.exists():
            return ToolResult(False, f"File not found: {arguments['path']}")
        content = path.read_text(encoding="utf-8")
        return ToolResult(True, content)

    def _write_file(self, arguments: dict[str, Any]) -> ToolResult:
        path = self._resolve_path(arguments["path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(arguments["content"], encoding="utf-8")
        return ToolResult(True, f"Wrote {arguments['path']} ({len(arguments['content'])} bytes)")

    def _edit_file(self, arguments: dict[str, Any]) -> ToolResult:
        path = self._resolve_path(arguments["path"])
        if not path.exists():
            return ToolResult(False, f"File not found: {arguments['path']}")
        content = path.read_text(encoding="utf-8")
        old = arguments["old"]
        new = arguments["new"]
        count = content.count(old)
        if count == 0:
            return ToolResult(False, f"old string not found in {arguments['path']}")
        if count > 1:
            return ToolResult(
                False,
                f"old string appears {count} times; provide a more specific snippet",
            )
        path.write_text(content.replace(old, new, 1), encoding="utf-8")
        return ToolResult(True, f"Edited {arguments['path']}")

    def _run_command(self, arguments: dict[str, Any]) -> ToolResult:
        command = arguments["command"]
        for pattern in FORBIDDEN_PATTERNS:
            if pattern in command:
                return ToolResult(False, f"Forbidden command pattern: {pattern}")

        try:
            proc = subprocess.run(
                command,
                shell=True,
                cwd=self.workspace,
                capture_output=True,
                text=True,
                timeout=self.command_timeout_sec,
                env={**os.environ, "PYTHONPATH": str(self.workspace)},
            )
        except subprocess.TimeoutExpired:
            return ToolResult(False, f"Command timed out after {self.command_timeout_sec}s")

        output_parts = []
        if proc.stdout:
            output_parts.append(proc.stdout.rstrip())
        if proc.stderr:
            output_parts.append(f"[stderr]\n{proc.stderr.rstrip()}")
        output_parts.append(f"[exit_code={proc.returncode}]")
        return ToolResult(proc.returncode == 0, "\n".join(output_parts))


OPENAI_TOOL_SPECS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file from the workspace",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a file in the workspace",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Replace an exact string in a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "old": {"type": "string"},
                    "new": {"type": "string"},
                },
                "required": ["path", "old", "new"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "Run a shell command in the workspace",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        },
    },
]

SYSTEM_PROMPT_TEXT = """You are a coding agent. Solve the task by using tools to read, write, edit files and run commands.

Available tools:
- read_file(path): read a file
- write_file(path, content): write a file
- edit_file(path, old, new): replace exact text in a file
- run_command(command): run a shell command in the workspace

To call a tool, output exactly one block in this format:
```tool
{"name": "tool_name", "arguments": {"arg": "value"}}
```

After each tool result you will receive a message with the output. Continue until the task is done.
When finished, respond with a short summary starting with "DONE:" and do not call more tools.
"""

SYSTEM_PROMPT_OPENAI = """You are a coding agent. Solve the task using the provided tools.
When finished, respond with a short summary starting with "DONE:".
"""
