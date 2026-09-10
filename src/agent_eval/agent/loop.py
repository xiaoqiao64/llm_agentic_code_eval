from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

from agent_eval.agent.parser import parse_openai_tool_calls, parse_text_tool_calls
from agent_eval.agent.tools import (
    OPENAI_TOOL_SPECS,
    SYSTEM_PROMPT_OPENAI,
    SYSTEM_PROMPT_TEXT,
    ToolExecutor,
)
from agent_eval.config import EvalConfig
from agent_eval.llm.client import LLMClient


@dataclass
class TurnRecord:
    turn: int
    assistant_content: str
    reasoning_content: str
    tool_calls: list[dict[str, Any]]
    tool_results: list[dict[str, Any]]
    latency_sec: float
    prompt_tokens: int
    completion_tokens: int
    reasoning_tokens: int


@dataclass
class AgentRunResult:
    turns: int
    elapsed_sec: float
    finished: bool
    stop_reason: str
    trajectory: list[TurnRecord] = field(default_factory=list)
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_reasoning_tokens: int = 0


class AgentLoop:
    def __init__(
        self,
        config: EvalConfig,
        llm: LLMClient,
        executor: ToolExecutor,
        task_prompt: str,
    ) -> None:
        self.config = config
        self.llm = llm
        self.executor = executor
        self.task_prompt = task_prompt
        self.tool_mode = config.tool_mode

    def _initial_messages(self) -> list[dict[str, Any]]:
        system = (
            SYSTEM_PROMPT_OPENAI
            if self.tool_mode == "openai"
            else SYSTEM_PROMPT_TEXT
        )
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": self.task_prompt},
        ]

    def run(self, deadline: float | None = None) -> AgentRunResult:
        messages = self._initial_messages()
        trajectory: list[TurnRecord] = []
        total_prompt = 0
        total_completion = 0
        total_reasoning = 0
        start = time.perf_counter()
        stop_reason = "max_turns"
        finished = False

        for turn in range(1, self.config.max_turns + 1):
            if deadline and time.perf_counter() >= deadline:
                stop_reason = "timeout"
                break

            tools = OPENAI_TOOL_SPECS if self.tool_mode == "openai" else None
            result = self.llm.chat(messages, tools=tools)

            total_prompt += result.metrics.prompt_tokens
            total_completion += result.metrics.completion_tokens
            total_reasoning += result.metrics.reasoning_tokens

            if self.tool_mode == "openai" and result.tool_calls:
                tool_calls = parse_openai_tool_calls(result.tool_calls)
            else:
                tool_calls = parse_text_tool_calls(result.content)

            assistant_msg: dict[str, Any] = {
                "role": "assistant",
                "content": result.content,
            }
            if result.reasoning_content:
                assistant_msg["reasoning_content"] = result.reasoning_content
            messages.append(assistant_msg)

            tool_results: list[dict[str, Any]] = []

            if not tool_calls:
                if result.content.strip().startswith("DONE:"):
                    finished = True
                    stop_reason = "done"
                else:
                    stop_reason = "no_tool_call"
                trajectory.append(
                    TurnRecord(
                        turn=turn,
                        assistant_content=result.content,
                        reasoning_content=result.reasoning_content,
                        tool_calls=[],
                        tool_results=[],
                        latency_sec=result.metrics.latency_sec,
                        prompt_tokens=result.metrics.prompt_tokens,
                        completion_tokens=result.metrics.completion_tokens,
                        reasoning_tokens=result.metrics.reasoning_tokens,
                    )
                )
                break

            for call in tool_calls:
                exec_result = self.executor.execute(call["name"], call["arguments"])
                tool_results.append(
                    {
                        "name": call["name"],
                        "arguments": call["arguments"],
                        "ok": exec_result.ok,
                        "output": exec_result.output,
                    }
                )
                if self.tool_mode == "openai":
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call.get("id", f"call_{turn}"),
                            "content": exec_result.output,
                        }
                    )
                else:
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                f"Tool result for {call['name']}:\n"
                                f"{exec_result.output}"
                            ),
                        }
                    )

            trajectory.append(
                TurnRecord(
                    turn=turn,
                    assistant_content=result.content,
                    reasoning_content=result.reasoning_content,
                    tool_calls=tool_calls,
                    tool_results=tool_results,
                    latency_sec=result.metrics.latency_sec,
                    prompt_tokens=result.metrics.prompt_tokens,
                    completion_tokens=result.metrics.completion_tokens,
                    reasoning_tokens=result.metrics.reasoning_tokens,
                )
            )

        elapsed = time.perf_counter() - start
        return AgentRunResult(
            turns=len(trajectory),
            elapsed_sec=elapsed,
            finished=finished,
            stop_reason=stop_reason,
            trajectory=trajectory,
            total_prompt_tokens=total_prompt,
            total_completion_tokens=total_completion,
            total_reasoning_tokens=total_reasoning,
        )

    @staticmethod
    def trajectory_to_dict(trajectory: list[TurnRecord]) -> list[dict[str, Any]]:
        return [
            {
                "turn": t.turn,
                "assistant_content": t.assistant_content,
                "reasoning_content": t.reasoning_content,
                "tool_calls": t.tool_calls,
                "tool_results": t.tool_results,
                "latency_sec": t.latency_sec,
                "prompt_tokens": t.prompt_tokens,
                "completion_tokens": t.completion_tokens,
                "reasoning_tokens": t.reasoning_tokens,
            }
            for t in trajectory
        ]
