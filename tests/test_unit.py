from pathlib import Path

from agent_eval.agent.parser import parse_text_tool_calls
from agent_eval.agent.tools import ToolExecutor
from agent_eval.config import deep_merge, load_config, parse_dot_kwargs
from agent_eval.llm.client import LLMClient, append_api_call_log


def test_parse_text_tool_calls():
    content = '''Let me read the file.
```tool
{"name": "read_file", "arguments": {"path": "foo.py"}}
```
'''
    calls = parse_text_tool_calls(content)
    assert len(calls) == 1
    assert calls[0]["name"] == "read_file"
    assert calls[0]["arguments"]["path"] == "foo.py"


def test_tool_executor_read_write(tmp_path: Path):
    executor = ToolExecutor(tmp_path)
    w = executor.execute("write_file", {"path": "a.txt", "content": "hello"})
    assert w.ok
    r = executor.execute("read_file", {"path": "a.txt"})
    assert r.ok and r.output == "hello"


def test_load_config_request_kwargs():
    config = load_config(
        overrides={
            "request_kwargs": parse_dot_kwargs(["reasoning_effort=low"]),
        }
    )
    assert config.request_kwargs["reasoning_effort"] == "low"


def test_parse_dot_kwargs():
    parsed = parse_dot_kwargs(
        [
            "extra_body.chat_template_kwargs.enable_thinking=True",
            "temperature=1",
            "extra_body.chat_template_kwargs.reasoning_effort=high",
        ]
    )
    assert parsed["temperature"] == 1
    assert parsed["extra_body"]["chat_template_kwargs"] == {
        "enable_thinking": True,
        "reasoning_effort": "high",
    }


def test_request_kwargs_override_priority():
    overrides = {
        "request_kwargs": parse_dot_kwargs(["temperature=1", "reasoning_effort=high"]),
    }
    config = load_config(overrides=overrides)
    client = LLMClient(config)
    req = client._build_request_kwargs([{"role": "user", "content": "hi"}])
    assert req["temperature"] == 1
    assert req["reasoning_effort"] == "high"
    preview = client.preview_request_kwargs()
    assert preview["temperature"] == 1
    assert preview["reasoning_effort"] == "high"
    assert preview["messages"].startswith("<omitted")


def test_append_api_call_log(tmp_path: Path):
    log_path = tmp_path / "nested" / "api_calls.jsonl"
    append_api_call_log(
        log_path,
        {"model": "m", "messages": [{"role": "user", "content": "hi"}]},
        {"id": "resp-1", "choices": []},
        0.12,
    )
    text = log_path.read_text(encoding="utf-8")
    assert '"model": "m"' in text
    assert '"latency_sec": 0.12' in text
    assert log_path.parent.is_dir()


def test_deep_merge_nested_extra_body():
    base = {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}
    override = parse_dot_kwargs(
        ["extra_body.chat_template_kwargs.enable_thinking=True"]
    )
    merged = deep_merge(base, override)
    assert merged["extra_body"]["chat_template_kwargs"]["enable_thinking"] is True
