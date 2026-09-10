# Agent Eval — 本地模型 Agent 写代码快速评测

快速评测本地 OpenAI 兼容 API（llama.cpp / vLLM）在多轮 Agent 场景下的写代码能力。

详细技术说明见 [docs/TECHNICAL.md](docs/TECHNICAL.md)。

## 特性

- **8-10 个 Python mini-repo 任务**：修 bug、实现函数、解循环依赖、LRU、CLI、重构、async 竞态、FastAPI
- **Agent 工具链**：`read_file` / `write_file` / `edit_file` / `run_command`
- **Thinking 配置**：支持 Qwen3.8 `reasoning_effort`（low / medium / xhigh）与 `thinking_token_budget`
- **报告**：通过率、耗时、轮次、reasoning tokens；JSON + Markdown
- **矩阵对比**：一次跑多个 thinking preset

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
  --thinking-preset low \
  --profile quick
```

本地 llama.cpp 等常用占位密钥 `EMPTY`，也可在 [`config/default.yaml`](config/default.yaml) 里设置 `api_key`，或用 `--api-key` / `--api_key` 传入。

### Smoke 测试（约 10 分钟，3 个任务）

```bash
agent-eval run --profile smoke --thinking-preset low
```

### 对比 thinking 级别

```bash
agent-eval matrix --presets low,medium,xhigh --profile quick --output results/matrix
```

### 指定任务

```bash
agent-eval run --tasks fix_counter_bug,implement_parse_log
```

## Thinking 配置

Qwen3.8-27B（llama.cpp）通过顶层 `reasoning_effort` 控制：

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen3.8-27B","messages":[{"role":"user","content":"hi"}],"reasoning_effort":"low"}'
```

预设定义在 [`config/default.yaml`](config/default.yaml)：

| Preset | 说明 |
|--------|------|
| `off` | 关闭 thinking（`enable_thinking: false` fallback） |
| `low` | `reasoning_effort: low` |
| `medium` | `reasoning_effort: medium` |
| `xhigh` | `reasoning_effort: xhigh` |
| `budget_512` | `thinking_token_budget: 512` |

CLI 覆盖：

```bash
agent-eval run --reasoning-effort medium
agent-eval run --thinking-preset budget_512
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

若 4bit 量化在 Agent 中表现差，可对比：
1. 同模型 8bit vs 4bit
2. `reasoning_effort` low vs off
3. 查看 `trajectories/` 定位失败步骤（未调用工具、错误编辑、不看报错等）

## 配置

编辑 [`config/default.yaml`](config/default.yaml) 设置默认 `base_url`、`model`、超时等。

## 许可

MIT
