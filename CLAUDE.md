# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This repo contains two projects. **Agent-S** is the primary focus — an open-source GUI agent framework by Simular AI for autonomous computer interaction via multimodal LLMs. **bytebot** is a separate desktop agent project (virtual desktop environments for task automation).

## Build & Development Commands

```bash
# Install (editable/dev mode)
pip install -e ".[dev]"

# Install (production)
pip install gui-agents

# Lint (uses black formatter)
black --check gui_agents

# Format code
black gui_agents

# Run the CLI agent
agent_s --provider openai --model gpt-5-2025-08-07 \
    --ground_provider huggingface --ground_url http://localhost:8080 \
    --ground_model ui-tars-1.5-7b --grounding_width 1920 --grounding_height 1080
```

Python 3.9–3.12 supported. CI runs on 3.10 and 3.11. `tesseract` must be installed on the system (`brew install tesseract` on macOS).

There is no test suite — evaluation is done via OSWorld and WindowsAgentArena benchmarks in `osworld_setup/`.

## Agent-S Architecture (Focus: S3)

The active/latest version is **S3** under `gui_agents/s3/`. Earlier versions (S1, S2, S2.5) exist under `gui_agents/s1/`, `gui_agents/s2/`, `gui_agents/s2_5/` respectively but S3 is the current development target.

### Key difference from earlier versions
S3 dropped the Manager/Worker hierarchy used in S2. It uses a single `Worker` agent directly, making it simpler, faster, and higher-performing (72.6% on OSWorld vs S2's ~63%).

### Component Graph

```
cli_app.py (entry point, argparse, execution loop)
    └── AgentS3 (agent_s.py) — thin orchestrator
            └── Worker (worker.py) — generates next actions
                    ├── LMMAgent (core/mllm.py) — multimodal LLM wrapper
                    │       └── Engine (core/engine.py) — provider-specific API clients
                    ├── OSWorldACI (grounding.py) — translates actions to executable pyautogui code
                    │       └── CodeAgent (code_agent.py) — optional Python/Bash execution
                    └── PROCEDURAL_MEMORY (memory/procedural_memory.py) — system prompt construction
```

### Core Modules

- **`gui_agents/s3/cli_app.py`** — CLI entry point (`agent_s` command). Runs an execution loop of up to 15 steps: screenshot → predict → execute. Handles Ctrl+C pause/resume, permission dialogs, screenshot scaling.

- **`gui_agents/s3/agents/agent_s.py`** — `UIAgent` base class and `AgentS3`. AgentS3 creates a `Worker` and delegates `predict(instruction, observation)` → `(info_dict, action_list)`.

- **`gui_agents/s3/agents/worker.py`** — The main agent logic. Manages trajectory history, reflection generation, format validation (via `call_llm_formatted`), and context flushing. Uses `use_thinking` for extended-thinking Claude models.

- **`gui_agents/s3/agents/grounding.py`** — `OSWorldACI` inherits from `ACI`. Methods decorated with `@agent_action` are automatically registered as available actions. These include mouse/keyboard operations, window management, app control, spreadsheet operations (`set_cell_values` via LibreOffice UNO, Linux only), and `call_code_agent`. The `@agent_action` decorator + `PROCEDURAL_MEMORY.construct_simple_worker_procedural_memory()` introspect the class to build system prompts dynamically.

- **`gui_agents/s3/core/mllm.py`** — `LMMAgent` wraps LLM engines with a unified message API. Factory pattern selects engine by `engine_type` string. Handles base64 image encoding, message history, and different content formats per provider (OpenAI-style `image_url` vs Anthropic-style `image` source).

- **`gui_agents/s3/core/engine.py`** — Large file (~17K lines) with provider-specific engine classes: `LMMEngineOpenAI`, `LMMEngineAnthropic`, `LMMEngineAzureOpenAI`, `LMMEnginevLLM`, `LMMEngineHuggingFace`, `LMMEngineGemini`, `LMMEngineOpenRouter`, `LMMEngineParasail`. Each implements `generate()` with exponential backoff. Anthropic engine also has `generate_with_thinking()`.

- **`gui_agents/s3/core/module.py`** — `BaseModule` provides `_create_agent()` factory for `LMMAgent` instances.

- **`gui_agents/s3/memory/procedural_memory.py`** — Constructs system prompts by introspecting `@agent_action`-decorated methods and their docstrings. Contains task execution guidelines and reflection prompts.

- **`gui_agents/s3/utils/common_utils.py`** — `call_llm_safe()` (retry logic), `call_llm_formatted()` (validation + reprompt loop), `parse_code_from_string()`, `split_thinking_response()`, `create_pyautogui_code()`.

- **`gui_agents/s3/utils/formatters.py`** — Response validators: `SINGLE_ACTION_FORMATTER`, `CODE_VALID_FORMATTER`. Lambda-based format checkers that return `(is_valid, feedback_message)`.

- **`gui_agents/s3/agents/code_agent.py`** — `CodeAgent` executes Python/Bash with a 20-step budget. Only available when `--enable_local_env` is set.

- **`gui_agents/s3/utils/local_env.py`** — `LocalEnv`/`LocalController` for subprocess-based code execution (30s timeout for Bash).

### Execution Flow (per step)

1. Screenshot is taken and passed as `observation`
2. `AgentS3.predict()` → `Worker.generate_next_action()`
3. Worker optionally generates a reflection on the trajectory so far
4. Worker constructs a prompt with reflection + text buffer + code agent results
5. `call_llm_formatted()` sends to LLM with format validation loop
6. Response is parsed for code via `parse_code_from_string()`
7. Code is translated to executable pyautogui via `create_pyautogui_code()` which calls `@agent_action` methods on `OSWorldACI`
8. Returned action string is `exec()`'d by `cli_app.py`
9. Context is flushed to stay within `max_trajectory_length`

### Environment Variables

```
OPENAI_API_KEY, ANTHROPIC_API_KEY, GEMINI_API_KEY,
AZURE_OPENAI_API_BASE, AZURE_OPENAI_API_KEY,
vLLM_ENDPOINT_URL, OPENROUTER_API_KEY, HF_TOKEN
```

### Platform-Specific Behavior

- **Linux**: Uses `wmctrl` for window management, supports `set_cell_values` (LibreOffice UNO)
- **macOS**: Uses `pyobjc` for OS integration
- **Windows**: Uses `pywinauto`/`pywin32`
- `set_cell_values` and `call_code_agent` actions are conditionally hidden from the system prompt based on platform/config

### OSWorld Evaluation

Evaluation scripts live in `osworld_setup/s3/`:
- `run.py` — AWS-based distributed evaluation
- `run_local.py` — Local evaluation
- `lib_run_single.py` — Single task executor
- `bbon/` — Best-of-N selection framework (trajectory comparison via judge LLM)
