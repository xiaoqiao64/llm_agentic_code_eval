from pathlib import Path

from agent_eval.agent.parser import parse_text_tool_calls
from agent_eval.agent.tools import ToolExecutor
from agent_eval.config import load_config


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


def test_load_config_thinking_presets():
    config = load_config()
    assert "low" in config.thinking_presets
    assert config.resolved_thinking().reasoning_effort == "low"
    config2 = load_config(overrides={"thinking_preset": "xhigh"})
    assert config2.resolved_thinking().reasoning_effort == "xhigh"
