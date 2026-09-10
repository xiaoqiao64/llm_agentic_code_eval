# Agent Eval — 本地模型 Agent 写代码快速评测

快速评测本地 OpenAI 兼容 API（llama.cpp / vLLM）在多轮 Agent 场景下的写代码能力。

详细技术说明见 [docs/TECHNICAL.md](docs/TECHNICAL.md)。

## 特性

- **8-10 个 Python mini-repo 任务**：修 bug、实现函数、解循环依赖、LRU、CLI、重构、async 竞态、FastAPI
- **Agent 工具链**：`read_file` / `write_file` / `edit_file` / `run_command`
- **请求参数**：通过 YAML `request_kwargs` 或 `--kwargs` 传入任意 chat completion 字段（如 `reasoning_effort`、thinking 相关 `extra_body`）
- **报告**：通过率、耗时、轮次、reasoning tokens；JSON + Markdown

## 安装

```bash
cd self/llm_agentic_code_eval
pip install -e .
```

## 快速开始

确保本地模型已启动（示例：Qwen3.8-27B @ llama.cpp 端口 8080）：

```bash
agent-eval run \
  --base-url http://localhost:8800/v1 \
  --model Qwen3.8-27B \
  --api-key YOUR_API_KEY \
  --profile quick
```

本地 llama.cpp 等常用占位密钥 `EMPTY`，也可在 [`config/default.yaml`](config/default.yaml) 里设置 `api_key`，或用 `--api-key` / `--api_key` 传入。

### Smoke 测试（约 10 分钟，3 个任务）

```bash
agent-eval run --profile smoke
```

### 指定任务

```bash
agent-eval run --tasks fix_counter_bug,implement_parse_log
```

## 模型 / thinking 参数

所有传给 `chat.completions` 的额外字段走 **`request_kwargs`**（YAML）或 **`--kwargs`**（CLI，优先级更高）。

Qwen3.8-27B（llama.cpp）示例：

```bash
agent-eval run --kwargs reasoning_effort=low
```

vLLM / chat template 示例：

```bash
agent-eval run --kwargs extra_body.chat_template_kwargs.enable_thinking=True
```

也可在 [`config/default.yaml`](config/default.yaml) 中写：

```yaml
request_kwargs:
  reasoning_effort: low
```

用 curl 验证 API 是否接受参数：

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen3.8-27B","messages":[{"role":"user","content":"hi"}],"reasoning_effort":"low"}'
```

## 工具模式

- **text**（默认）：` ```tool {...} ``` ` 文本协议，适合 llama.cpp 量化模型
- **openai**：原生 function calling，适合 vLLM

```bash
agent-eval run --tool-mode openai
```

## 任务列表

```bash
agent-eval list-tasks
```

| Profile | 任务数 | 说明 |
|---------|--------|------|
| `smoke` | 3 | 快速冒烟 |
| `quick` | 10 | 完整快速评测（约 30 分钟） |

## 输出

结果写入 `results/<timestamp>/`：

- `summary.json` — 机器可读
- `summary.md` — 人类可读表格
- `trajectories/` — 每任务完整对话轨迹

## 结果解读

- **Pass rate**：测试通过率，核心指标
- **Time**：本地模型越慢差异越大；对比时需固定硬件与量化
- **Turns**：轮次过多可能说明模型在无效重试
- **Reasoning tokens**：thinking 开时 token 与延迟通常显著增加

若 4bit 量化在 Agent 中表现差，可对比不同 `request_kwargs` / `--kwargs`，并查看 `trajectories/` 定位失败步骤。

## 配置

编辑 [`config/default.yaml`](config/default.yaml) 设置默认 `base_url`、`model`、`request_kwargs`、超时等。

## 许可

MIT
