# Agent-S + Bytebot Remote Desktop Integration

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Prerequisites](#prerequisites)
4. [Environment Setup](#environment-setup)
5. [Bytebot Desktop VM](#bytebot-desktop-vm)
6. [Remote Integration Layer](#remote-integration-layer)
7. [LLM Configuration](#llm-configuration)
8. [Running Agent-S Remotely](#running-agent-s-remotely)
9. [OSWorld Benchmark Integration](#osworld-benchmark-integration)
10. [Troubleshooting](#troubleshooting)
11. [File Reference](#file-reference)

---

## Overview

This integration enables **Agent-S** (an autonomous GUI agent framework) to control a **bytebot Docker VM** instead of the local desktop. This is essential for headless servers where no physical display exists.

**How it works**: Agent-S generates `pyautogui` code strings (e.g., `import pyautogui; pyautogui.click(500, 300)`) which are normally `exec()`'d on the local machine. We intercept at the Python module level by injecting drop-in replacements for `pyautogui`, `pyperclip`, and `subprocess` into `sys.modules`. These replacements translate calls into HTTP requests to bytebotd's REST API running inside the Docker container.

**Two-model architecture**:
- **Main LLM** (e.g., Claude Sonnet 4.5 on Azure, GPT-4o, etc.) — decides what actions to take based on screenshots
- **Grounding model** (e.g., UI-TARS-1.5-7B on vLLM) — converts high-level actions into exact screen coordinates

```
┌──────────────────────────────────────────────────────┐
│                    Host Machine                       │
│                                                      │
│  ┌──────────────────────────────────────────────┐    │
│  │            Agent-S (cli_app_remote.py)        │    │
│  │                                              │    │
│  │  AgentS3.predict()                           │    │
│  │    └─► Worker.generate_next_action()         │    │
│  │          ├─► Main LLM (cloud/API)            │    │
│  │          └─► OSWorldACI (grounding)          │    │
│  │                └─► Grounding model (vLLM)    │    │
│  │                      └─► pyautogui code      │    │
│  │                                              │    │
│  │  exec(code) ──► RemotePyAutoGUI ─────────┐  │    │
│  └──────────────────────────────────────┐    │  │    │
│                                         │    │  │    │
│  ┌──────────────────────────────────────┤    │  │    │
│  │      Bytebot Docker Container        │    │  │    │
│  │                                      │◄───┘  │    │
│  │  bytebotd REST API (:9990)           │ HTTP  │    │
│  │    ├── Xvfb :0 (1280x960)           │       │    │
│  │    ├── XFCE4 desktop                │       │    │
│  │    ├── x11vnc + noVNC               │       │    │
│  │    ├── Firefox, VS Code, etc.       │       │    │
│  │    └── wmctrl, xclip                │       │    │
│  └──────────────────────────────────────┘       │    │
│                                                      │
└──────────────────────────────────────────────────────┘
```

---

## Architecture

### Agent-S Execution Flow (per step)

```
1. Screenshot taken from bytebot VM via HTTP
2. AgentS3.predict(instruction, observation)
3.   └─► Worker.generate_next_action()
4.         ├─► Optionally generates reflection on trajectory so far
5.         ├─► Constructs prompt with reflection + text buffer
6.         ├─► call_llm_formatted() sends to Main LLM with format validation
7.         └─► Response parsed for code via parse_code_from_string()
8. Code translated to pyautogui via create_pyautogui_code()
9.   └─► @agent_action methods on OSWorldACI called
10.        └─► Grounding model (UI-TARS) converts actions to coordinates
11. Returned code string exec()'d — intercepted by RemotePyAutoGUI
12.   └─► HTTP POST to bytebotd REST API on the Docker VM
13. Loop back to step 1
```

### Component Graph

```
cli_app_remote.py (remote entry point)
    ├── BytebotClient (HTTP client for bytebotd)
    ├── RemotePyAutoGUI (drop-in pyautogui replacement)
    ├── RemotePyperclip (clipboard via docker exec + xclip)
    ├── RemoteSubprocess (commands via docker exec)
    └── AgentS3 (agent_s.py) — thin orchestrator
            └── Worker (worker.py) — generates next actions
                    ├── LMMAgent (core/mllm.py) — multimodal LLM wrapper
                    │       └── Engine (core/engine.py) — provider-specific API clients
                    ├── OSWorldACI (grounding.py) — translates actions → pyautogui code
                    │       └── Grounding model (e.g., UI-TARS via vLLM)
                    └── PROCEDURAL_MEMORY — system prompt construction
```

### sys.modules Injection Pattern

The key insight: Agent-S grounding agent generates code like:

```python
import pyautogui
pyautogui.click(640, 480)
```

This code is `exec()`'d. By replacing `sys.modules["pyautogui"]` before execution, all `import pyautogui` statements resolve to our `RemotePyAutoGUI` object, which sends HTTP requests to the Docker VM instead of controlling the local display.

```python
# In cli_app_remote.py main():
sys.modules["pyautogui"] = remote_pyautogui      # RemotePyAutoGUI instance
sys.modules["pyperclip"] = remote_pyperclip       # RemotePyperclip instance
sys.modules["subprocess"] = remote_subprocess      # RemoteSubprocess instance

# Now when exec'd code does `import pyautogui`, it gets RemotePyAutoGUI
exec(code[0])  # Transparently controls the remote VM
```

---

## Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | 3.12+ | System has `python3`, not `python` |
| uv | latest | Python package manager |
| Docker | 20.10+ | For bytebot container |
| Docker Compose | v2+ | `docker compose` (not `docker-compose`) |
| vLLM server | accessible | Serving UI-TARS-1.5-7B for grounding |
| Main LLM API | accessible | Claude/GPT/etc. for decision-making |

---

## Environment Setup

### 1. Clone and install

```bash
cd /home/twel/Projects/aitm

# Initialize uv project (already done)
uv init

# Install all dependencies (Agent-S installed as editable local package)
uv sync
```

The `pyproject.toml` defines all dependencies and mounts Agent-S as an editable package:

```toml
[project]
name = "aitm"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
    "numpy", "backoff", "pandas", "openai", "anthropic",
    "fastapi", "uvicorn", "paddleocr", "paddlepaddle",
    "together", "scikit-learn", "websockets", "tiktoken",
    "selenium", "pyautogui", "toml", "pytesseract",
    "google-genai", "black", "tqdm", "python-dotenv",
    "wrapt-timeout-decorator", "Pillow", "pyperclip",
    "requests", "gui-agents>=0.3.2",
]

[tool.uv.sources]
gui-agents = { path = "Agent-S", editable = true }
```

### 2. Environment variables

Create a `.env` file at the project root:

```bash
# For Azure-hosted Claude (AnthropicFoundry)
ANTHROPIC_API_KEY=<your-azure-api-key>
ANTHROPIC_BASE_URL=https://<your-resource>.services.ai.azure.com/anthropic/

# For direct Anthropic API (no base_url needed)
# ANTHROPIC_API_KEY=sk-ant-...

# For OpenAI
# OPENAI_API_KEY=sk-...

# For vLLM grounding model (if using env vars instead of CLI args)
# vLLM_ENDPOINT_URL=http://localhost:4244/v1
# vLLM_API_KEY=dummy
```

### 3. Verify installation

```bash
# Check imports
uv run python3 -c "from gui_agents.s3.remote.bytebot_client import BytebotClient; print('OK')"
uv run python3 -c "from gui_agents.s3.remote.remote_pyautogui import RemotePyAutoGUI; print('OK')"
uv run python3 -c "from gui_agents.s3.cli_app_remote import main; print('OK')"
```

---

## Bytebot Desktop VM

### What is it

Bytebot provides a complete Linux desktop environment in a Docker container:
- **OS**: Ubuntu 22.04
- **Desktop**: XFCE4 with Xvfb at 1280x960 pixels
- **Applications**: Firefox ESR, Thunderbird, VS Code, 1Password, gedit, xpaint
- **Remote access**: x11vnc + noVNC (web-based VNC viewer)
- **Control API**: bytebotd REST API on port 9990

### Start the container

```bash
cd /home/twel/Projects/aitm/bytebot/docker
docker compose -f docker-compose.development.yml up -d bytebot-desktop
```

Only the `bytebot-desktop` service is needed. The `postgres` service is for bytebot's own agent (which we replace with Agent-S).

### Verify the container

```bash
# Check container is running
docker ps | grep bytebot-desktop

# Test the REST API with a screenshot
curl -s -X POST http://localhost:9990/computer-use \
    -H "Content-Type: application/json" \
    -d '{"action": "screenshot"}' | python3 -c "
import sys, json
resp = json.load(sys.stdin)
print(f'Screenshot received: {len(resp.get(\"image\", \"\"))} bytes (base64)')
"

# Open the noVNC web viewer to watch the desktop
# Navigate to: http://localhost:9990/novnc
```

### Container architecture (supervisord)

The container runs multiple services via supervisord:

| Service | Command | Port | Purpose |
|---|---|---|---|
| xvfb | `Xvfb :0 -screen 0 1280x960x24` | — | Virtual framebuffer |
| xfce4 | `startxfce4` | — | Desktop environment |
| x11vnc | `x11vnc -display :0 -rfbport 5900` | 5900 | VNC server |
| websockify | `websockify 6080 localhost:5900` | 6080 | WebSocket proxy for noVNC |
| bytebotd | `node /bytebot/bytebotd/dist/main.js` | 9990 | REST API + noVNC proxy |

### bytebotd REST API

All actions go through `POST /computer-use` with a JSON body.

#### Screenshot

```bash
curl -X POST http://localhost:9990/computer-use \
    -H "Content-Type: application/json" \
    -d '{"action": "screenshot"}'
# Response: {"image": "<base64-encoded PNG>"}
```

#### Mouse actions

```json
// Click
{"action": "click_mouse", "coordinates": {"x": 640, "y": 480}, "button": "left", "clickCount": 1}

// Double-click
{"action": "click_mouse", "coordinates": {"x": 640, "y": 480}, "button": "left", "clickCount": 2}

// Right-click
{"action": "click_mouse", "coordinates": {"x": 640, "y": 480}, "button": "right", "clickCount": 1}

// Move mouse
{"action": "move_mouse", "coordinates": {"x": 640, "y": 480}}

// Drag
{"action": "drag_mouse", "path": [{"x": 100, "y": 100}, {"x": 500, "y": 500}], "button": "left"}

// Mouse button press/release
{"action": "press_mouse", "button": "left", "press": "down"}
{"action": "press_mouse", "button": "left", "press": "up"}

// Click with modifier keys
{"action": "click_mouse", "coordinates": {"x": 640, "y": 480}, "button": "left", "clickCount": 1, "holdKeys": ["ctrl"]}
```

#### Keyboard actions

```json
// Type text (character by character)
{"action": "type_text", "text": "Hello, World!"}

// Paste text (via clipboard, faster for long text)
{"action": "paste_text", "text": "A long paragraph..."}

// Press keys (press and release, like pressing Enter)
{"action": "type_keys", "keys": ["Return"]}

// Hotkey combination (all pressed simultaneously)
{"action": "type_keys", "keys": ["Control_L", "c"]}

// Hold/release keys
{"action": "press_keys", "keys": ["Shift_L"], "press": "down"}
{"action": "press_keys", "keys": ["Shift_L"], "press": "up"}
```

#### Scroll

```json
// Scroll up
{"action": "scroll", "direction": "up", "scrollCount": 3}

// Scroll down at specific position
{"action": "scroll", "direction": "down", "scrollCount": 5, "coordinates": {"x": 640, "y": 480}}

// Horizontal scroll
{"action": "scroll", "direction": "left", "scrollCount": 2}
```

#### Other actions

```json
// Wait
{"action": "wait", "duration": 1000}

// Read file from container
{"action": "read_file", "path": "/home/user/test.txt"}

// Write file to container
{"action": "write_file", "path": "/home/user/test.txt", "content": "Hello"}
```

### Changing screen resolution

The default resolution is 1280x960. To change it, modify the supervisord config inside the container or rebuild the image.

**Option A: Temporary change (lost on container restart)**
```bash
docker exec bytebot-desktop bash -c "
    supervisorctl stop xfce4 x11vnc websockify bytebotd
    supervisorctl stop xvfb
    # Edit supervisord.conf or use Xvfb command directly
    Xvfb :0 -screen 0 1920x1080x24 -ac -nolisten tcp &
    supervisorctl start xfce4 x11vnc websockify bytebotd
"
```

**Option B: Permanent change (rebuild image)**

Edit `bytebot/packages/bytebotd/root/etc/supervisor/conf.d/supervisord.conf`:
```ini
[program:xvfb]
command=Xvfb :0 -screen 0 1920x1080x24 -ac -nolisten tcp
```

Then rebuild:
```bash
cd bytebot/docker
docker compose -f docker-compose.development.yml build bytebot-desktop
docker compose -f docker-compose.development.yml up -d bytebot-desktop
```

---

## Remote Integration Layer

### Module: `gui_agents/s3/remote/`

This package bridges Agent-S with the bytebot Docker VM.

```
Agent-S/gui_agents/s3/remote/
├── __init__.py              # Package init (empty)
├── bytebot_client.py        # HTTP client for bytebotd REST API
├── remote_pyautogui.py      # RemotePyAutoGUI + RemotePyperclip + RemoteSubprocess
└── bytebot_env.py           # DesktopEnv-compatible adapter for OSWorld
```

### BytebotClient (`bytebot_client.py`)

Thin HTTP wrapper around the bytebotd REST API. Uses only `urllib` (no external deps).

```python
from gui_agents.s3.remote.bytebot_client import BytebotClient

client = BytebotClient(
    base_url="http://localhost:9990",
    screen_width=1280,
    screen_height=960,
)

# Take screenshot → raw PNG bytes
png_bytes = client.screenshot()

# Click at coordinates
client.click(640, 480, button="left", clicks=1)

# Type text
client.type_text("Hello, World!")

# Press keys (using X11/bytebotd key names)
client.type_keys(["Control_L", "c"])  # Ctrl+C

# Scroll
client.scroll(direction="down", count=3)

# Health check
if client.health_check():
    print("Connected!")
```

**Methods**:

| Method | Parameters | Description |
|---|---|---|
| `screenshot()` | — | Returns raw PNG bytes |
| `click(x, y, button, clicks, hold_keys)` | coordinates, button name, click count | Mouse click |
| `move_mouse(x, y)` | coordinates | Move cursor |
| `drag(path, button, hold_keys)` | list of `{x, y}` dicts | Drag mouse along path |
| `press_mouse(button, press)` | button, "up"/"down" | Hold/release mouse button |
| `scroll(direction, count, x, y, hold_keys)` | "up"/"down"/"left"/"right", count | Scroll wheel |
| `type_text(text, delay)` | string, optional ms delay | Type characters |
| `paste_text(text)` | string | Paste via clipboard |
| `type_keys(keys, delay)` | list of key names | Press and release key combo |
| `press_keys(keys, press)` | list of key names, "up"/"down" | Hold/release keys |
| `wait(duration_ms)` | milliseconds | Wait |
| `get_screen_size()` | — | Returns `(width, height)` tuple |
| `health_check()` | — | Returns `True` if responsive |

### RemotePyAutoGUI (`remote_pyautogui.py`)

Drop-in replacement for `pyautogui`. Implements the same method signatures so `exec()`'d code works without modification.

**pyautogui method → bytebotd API mapping**:

| pyautogui call | bytebotd action | Notes |
|---|---|---|
| `click(x, y, clicks, button)` | `click_mouse` | |
| `doubleClick(x, y)` | `click_mouse` (clickCount=2) | |
| `rightClick(x, y)` | `click_mouse` (button=right) | |
| `moveTo(x, y)` | `move_mouse` | |
| `dragTo(x, y, duration, button)` | `drag_mouse` | |
| `mouseDown(x, y, button)` | `press_mouse` (press=down) | |
| `mouseUp(x, y, button)` | `press_mouse` (press=up) | |
| `write(text, interval)` | `type_text` | ASCII character input |
| `typewrite(text, interval)` | `type_text` | Alias for write |
| `press(key)` | `type_keys` | Single key press+release |
| `hotkey(*keys)` | `type_keys` | Key combination |
| `keyDown(key)` | `press_keys` (press=down) | Hold key |
| `keyUp(key)` | `press_keys` (press=up) | Release key |
| `scroll(clicks, x, y)` | `scroll` | Positive=up, negative=down |
| `vscroll(clicks)` | `scroll` | Vertical scroll |
| `hscroll(clicks)` | `scroll` (left/right) | Horizontal scroll |
| `screenshot()` | `screenshot` | Returns PIL.Image |
| `size()` | — | Returns configured dimensions |

**Key name mapping** (`_KEY_MAP`):

pyautogui uses lowercase names (e.g., `"enter"`, `"ctrl"`), while bytebotd expects X11 keysym names (e.g., `"Return"`, `"Control_L"`). The `_KEY_MAP` dictionary handles translation:

| pyautogui name | bytebotd/X11 name |
|---|---|
| `enter`, `return` | `Return` |
| `tab` | `Tab` |
| `escape`, `esc` | `Escape` |
| `backspace` | `BackSpace` |
| `delete`, `del` | `Delete` |
| `ctrl`, `ctrlleft` | `Control_L` |
| `ctrlright` | `Control_R` |
| `alt`, `altleft` | `Alt_L` |
| `shift`, `shiftleft` | `Shift_L` |
| `command`, `win`, `super`, `meta` | `Super_L` |
| `up`/`down`/`left`/`right` | `Up`/`Down`/`Left`/`Right` |
| `home`/`end` | `Home`/`End` |
| `pageup`/`pagedown` | `Page_Up`/`Page_Down` |
| `f1`–`f12` | `F1`–`F12` |
| `capslock` | `Caps_Lock` |
| `insert` | `Insert` |
| `printscreen`, `prtsc` | `Print` |
| Single characters (`a`, `1`, etc.) | Passed through as-is |

### RemotePyperclip

Replaces `pyperclip` for clipboard operations on the remote VM.

```python
# How it works:
# 1. pyperclip.copy(text) → docker exec -i bytebot-desktop bash -c "DISPLAY=:0 xclip -selection clipboard"
#    with text piped to stdin
# 2. The grounding agent then does hotkey('ctrl', 'v') to paste
```

Why not use bytebotd's `paste_text` action? Because the Agent-S grounding agent generates two separate operations:
1. `pyperclip.copy(text)` — set clipboard
2. `pyautogui.hotkey('ctrl', 'v')` — paste

If `copy()` used `paste_text` (which both sets clipboard AND types Ctrl+V), the subsequent `hotkey()` call would paste twice.

### RemoteSubprocess

Replaces `subprocess` for running commands inside the container. The Agent-S grounding agent uses `subprocess` for window management (`wmctrl`) and system queries (`uname`, `ldconfig`).

```python
# All subprocess calls are wrapped in docker exec:
# subprocess.run(["wmctrl", "-l"]) → docker exec bytebot-desktop bash -c "wmctrl -l"
# subprocess.check_output("uname -p") → docker exec bytebot-desktop bash -c "uname -p"
```

Re-exports `PIPE`, `DEVNULL`, `CalledProcessError`, and `sys` from the real `subprocess` module to avoid `AttributeError`.

---

## LLM Configuration

### Main LLM (decision-making)

Agent-S supports multiple providers. Set via `--provider` and `--model` CLI args.

#### Anthropic (direct API)

```bash
--provider anthropic --model claude-sonnet-4-5-20250929
```
Env: `ANTHROPIC_API_KEY`

#### Anthropic on Azure Foundry

```bash
--provider anthropic --model claude-sonnet-4-5 \
    --model_url "https://<resource>.services.ai.azure.com/anthropic/"
```
Env: `ANTHROPIC_API_KEY` (Azure API key), `ANTHROPIC_BASE_URL`

The engine uses `AnthropicFoundry` client (not plain `Anthropic`) when `base_url` is provided:

```python
# In engine.py LMMEngineAnthropic.generate():
if self.base_url:
    self.llm_client = AnthropicFoundry(api_key=api_key, base_url=self.base_url)
else:
    self.llm_client = Anthropic(api_key=api_key)
```

#### OpenAI

```bash
--provider openai --model gpt-4o
```
Env: `OPENAI_API_KEY`

#### Azure OpenAI

```bash
--provider azure_openai --model gpt-4o
```
Env: `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`, `OPENAI_API_VERSION`

#### vLLM (self-hosted)

```bash
--provider vllm --model <model-name> --model_url http://<host>:<port>/v1
```
Env: `vLLM_API_KEY`, `vLLM_ENDPOINT_URL`

### Grounding model (coordinate prediction)

The grounding model converts high-level actions (e.g., "click the search box") into pixel coordinates. **UI-TARS-1.5-7B** by ByteDance is the recommended model.

#### vLLM (recommended)

```bash
--ground_provider vllm \
    --ground_url http://localhost:4244/v1 \
    --ground_api_key dummy \
    --ground_model ByteDance-Seed/UI-TARS-1.5-7B \
    --grounding_width 1920 --grounding_height 1080
```

Verify vLLM is serving:
```bash
curl http://localhost:4244/v1/models
# Should list ByteDance-Seed/UI-TARS-1.5-7B
```

#### HuggingFace TGI

```bash
--ground_provider huggingface \
    --ground_url http://localhost:8080 \
    --ground_model ui-tars-1.5-7b \
    --grounding_width 1920 --grounding_height 1080
```

### Screen resolution and coordinate scaling

The grounding model (UI-TARS) is trained at a specific resolution (typically 1920x1080). The bytebot VM runs at 1280x960. Agent-S handles this mismatch automatically via `OSWorldACI.resize_coordinates()`:

```
Grounding model outputs coordinates in its trained space (1920x1080)
    ↓ resize_coordinates()
Coordinates mapped to actual screen dimensions (1280x960)
    ↓
Sent to bytebotd via RemotePyAutoGUI
```

Set these CLI args to match your setup:
- `--screen_width` / `--screen_height`: Actual VM display size (default: 1280x960)
- `--grounding_width` / `--grounding_height`: Resolution the grounding model expects

---

## Running Agent-S Remotely

### Quick start

```bash
# 1. Start the bytebot VM
cd /home/twel/Projects/aitm/bytebot/docker
docker compose -f docker-compose.development.yml up -d bytebot-desktop

# 2. Verify VM is ready
curl -s -X POST http://localhost:9990/computer-use \
    -H "Content-Type: application/json" \
    -d '{"action": "screenshot"}' | python3 -c "
import sys, json; r = json.load(sys.stdin); print(f'OK: {len(r[\"image\"])} bytes')
"

# 3. Run Agent-S against the VM
cd /home/twel/Projects/aitm
uv run python3 -m gui_agents.s3.cli_app_remote \
    --bytebot_url http://localhost:9990 \
    --screen_width 1280 --screen_height 960 \
    --provider anthropic --model claude-sonnet-4-5 \
    --model_url "https://<resource>.services.ai.azure.com/anthropic/" \
    --ground_provider vllm \
    --ground_url http://localhost:4244/v1 \
    --ground_api_key dummy \
    --ground_model ByteDance-Seed/UI-TARS-1.5-7B \
    --grounding_width 1920 --grounding_height 1080 \
    --task "Open Firefox and search for 'hello world'"
```

### CLI reference (`cli_app_remote.py`)

```
usage: cli_app_remote.py [options]

Bytebot connection:
  --bytebot_url URL          bytebotd REST API URL (required)
  --screen_width INT          VM display width (default: 1280)
  --screen_height INT         VM display height (default: 960)
  --container_name NAME       Docker container name (default: bytebot-desktop)

Main LLM:
  --provider TYPE             LLM provider: openai, anthropic, vllm, etc. (default: openai)
  --model NAME                Model name (default: gpt-5-2025-08-07)
  --model_url URL             Model API base URL (triggers AnthropicFoundry for anthropic)
  --model_api_key KEY         Model API key (overrides env var)
  --model_temperature FLOAT   Fixed temperature (e.g., 1.0 for o3)

Grounding model:
  --ground_provider TYPE      Grounding provider: vllm, huggingface, etc. (required)
  --ground_url URL            Grounding model endpoint (required)
  --ground_api_key KEY        Grounding model API key
  --ground_model NAME         Grounding model name (required)
  --grounding_width INT       Grounding model image width (required)
  --grounding_height INT      Grounding model image height (required)

Agent behavior:
  --max_trajectory_length INT Max image turns in trajectory (default: 8)
  --enable_reflection         Enable reflection (default: true)
  --enable_local_env          Enable code execution inside container
  --task TEXT                 Task instruction (omit for interactive mode)
```

### Interactive mode

If `--task` is not provided, the CLI enters interactive mode:

```bash
uv run python3 -m gui_agents.s3.cli_app_remote \
    --bytebot_url http://localhost:9990 \
    --provider anthropic --model claude-sonnet-4-5 \
    --ground_provider vllm --ground_url http://localhost:4244/v1 \
    --ground_api_key dummy --ground_model ByteDance-Seed/UI-TARS-1.5-7B \
    --grounding_width 1920 --grounding_height 1080

# Enters interactive loop:
# Query: Open Firefox
# ... agent executes ...
# Would you like to provide another query? (y/n): y
# Query: Navigate to google.com
```

### Debugging with Ctrl+C

Press `Ctrl+C` during execution to pause:
- Press `Esc` to resume
- Press `Ctrl+C` again to quit

### Watching the VM

Open `http://localhost:9990/novnc` in a browser to watch the desktop in real-time via noVNC.

### Logs

Logs are written to the `logs/` directory:
- `logs/normal-<timestamp>.log` — INFO level
- `logs/debug-<timestamp>.log` — DEBUG level (verbose)
- `logs/sdebug-<timestamp>.log` — Desktop env debug

---

## OSWorld Benchmark Integration

### BytebotDesktopEnv (`bytebot_env.py`)

A `DesktopEnv`-compatible adapter that uses the bytebot container instead of VMware/AWS VMs. Implements the interface expected by `lib_run_single.py`:

```python
class BytebotDesktopEnv:
    def reset(self, task_config)       # Run setup commands via docker exec
    def _get_obs(self)                 # Screenshot from bytebotd
    def step(self, action, sleep)      # exec() with remote modules injected
    def evaluate(self)                 # Run eval scripts via docker exec
    def close(self)                    # No-op (container left running)
```

### Running OSWorld evaluation

```bash
cd /home/twel/Projects/aitm/Agent-S/osworld_setup/s3

# Run the first 20 tests using bytebot
uv run python3 run_local.py \
    --provider_name bytebot \
    --bytebot_url http://localhost:9990 \
    --container_name bytebot-desktop \
    --screen_width 1280 --screen_height 960 \
    --model_provider anthropic --model claude-sonnet-4-5 \
    --model_url "https://<resource>.services.ai.azure.com/anthropic/" \
    --ground_provider vllm \
    --ground_url http://localhost:4244/v1 \
    --ground_api_key dummy \
    --ground_model ByteDance-Seed/UI-TARS-1.5-7B \
    --grounding_width 1920 --grounding_height 1080 \
    --test_all_meta_path /home/twel/Projects/aitm/Agent-S/evaluation_sets/test_all.json \
    --test_config_base_dir /home/twel/Projects/bytebot-azure/osworld-bench/OSWorld/evaluation_examples \
    --max_examples 20 \
    --result_dir ./results
```

Key arguments:
- `--provider_name bytebot` — use BytebotDesktopEnv instead of OSWorld's DesktopEnv
- `--test_all_meta_path` — path to `test_all.json` (369 examples) or `test_small_new.json` (65 examples)
- `--test_config_base_dir` — directory containing `examples/{domain}/{example_id}.json` files
- `--max_examples 20` — limit to the first 20 examples across domains
- `--domain os` — optionally run only one domain (e.g., `os`, `chrome`, `gimp`)

The test metadata JSON maps domains to example IDs:
```json
{
  "chrome": ["bb5e4c0d-...", "7b6c7e24-...", ...],
  "os": ["5ea617a3-...", ...],
  "gimp": [...]
}
```

The actual example configs live at `{test_config_base_dir}/examples/{domain}/{example_id}.json`.

### How `BytebotDesktopEnv` handles OSWorld task configs

Each OSWorld example has a `config` list with structured setup steps:

```json
{
  "instruction": "Recover the deleted poster from the Trash",
  "config": [
    {
      "type": "download",
      "parameters": {
        "files": [{"url": "https://huggingface.co/datasets/.../poster.webp", "path": "/home/user/Desktop/poster.webp"}]
      }
    },
    {
      "type": "execute",
      "parameters": {
        "command": "gio trash /home/user/Desktop/poster.webp",
        "shell": true
      }
    }
  ],
  "evaluator": {
    "func": "exact_match",
    "result": {"type": "vm_command_line", "command": "[ -f /home/user/Desktop/poster.webp ] && echo 'File exists.'"},
    "expected": {"type": "rule", "rules": {"expected": "File exists.\n"}}
  }
}
```

`BytebotDesktopEnv.reset()` processes each config step:

| Config type | What it does |
|---|---|
| `download` | Downloads URL to local cache, `docker cp` into container |
| `execute` | Runs shell command via `docker exec` (supports `{SCREEN_WIDTH}` etc.) |
| `launch` | Runs command non-blocking (`nohup ... &`) |
| `command` | Alias for execute |
| `open` | Opens file with `xdg-open` inside container |
| `sleep` | `time.sleep(seconds)` |
| `activate_window` | `wmctrl -a` inside container |
| `close_window` | `wmctrl -c` inside container |
| `chrome_open_tabs` | Opens URLs in Firefox (bytebot uses Firefox, not Chrome) |

### Evaluation

`BytebotDesktopEnv.evaluate()` handles OSWorld's evaluator format:

1. Runs `postconfig` steps (same setup handlers)
2. Gets actual result using `result.type`:
   - `vm_command_line` — runs command in container, captures stdout
   - `vm_file` — copies file from container via `docker cp`
   - `list_directory` — lists directory contents
3. Gets expected result using `expected.type`:
   - `rule` — direct expected value from config
   - `vm_command_line` — runs command for expected value
   - `cloud_file` — downloads reference file from URL
4. Compares using metric (`func`):
   - `exact_match` — string equality
   - `match_in_list` — substring match
   - `fuzzy_match` — case-insensitive match
   - `is_in_list` — value in list
   - `infeasible` — checks if agent correctly reported FAIL

**Limitation**: Some OSWorld evaluator types (Chrome CDP-based, file format comparison like DOCX/PPTX, GIMP config parsing) are not implemented in BytebotDesktopEnv. These require the full OSWorld `desktop_env` package. The bytebot adapter covers the most common evaluation types (`vm_command_line` + `exact_match`), which handles the majority of `os` domain tasks.

---

## Troubleshooting

### Cannot connect to bytebotd

```
ERROR: Cannot connect to bytebotd at http://localhost:9990
```

**Fix**: Ensure the container is running and port 9990 is exposed:
```bash
docker ps | grep bytebot-desktop
docker compose -f docker-compose.development.yml up -d bytebot-desktop
curl -X POST http://localhost:9990/computer-use -H "Content-Type: application/json" -d '{"action": "screenshot"}'
```

### Azure Anthropic endpoint unreachable

If `AnthropicFoundry` hangs or DNS fails:
```bash
# Test connectivity
curl -s -o /dev/null -w "%{http_code}" --connect-timeout 5 "https://<resource>.services.ai.azure.com"

# Fix WSL2 DNS (if applicable)
echo "nameserver 8.8.8.8" | sudo tee -a /etc/resolv.conf

# Or port-forward from a machine with access
ssh -L 8443:<resource>.services.ai.azure.com:443 jump-host
```

### vLLM grounding model not responding

```bash
# Check vLLM is serving
curl http://localhost:4244/v1/models
# Should return: {"data": [{"id": "ByteDance-Seed/UI-TARS-1.5-7B", ...}]}

# If port-forwarded, ensure the tunnel is alive
ss -tlnp | grep 4244
```

### Key mapping issues

If keyboard actions produce wrong characters, check the `_KEY_MAP` in `remote_pyautogui.py`. Common issues:
- pyautogui uses `"enter"` but bytebotd expects `"Return"`
- Modifier keys: `"ctrl"` → `"Control_L"`, `"alt"` → `"Alt_L"`

### Clipboard not working

`RemotePyperclip` requires `xclip` installed in the container (it is by default). Verify:
```bash
docker exec bytebot-desktop which xclip
# Should output: /usr/bin/xclip
```

### `python` command not found

The system has `python3`, not `python`. Always use:
```bash
uv run python3 -m gui_agents.s3.cli_app_remote ...
```

### Screen resolution mismatch

If the grounding model outputs coordinates outside the screen, check:
1. `--screen_width`/`--screen_height` match the actual Xvfb resolution (default: 1280x960)
2. `--grounding_width`/`--grounding_height` match UI-TARS training resolution (usually 1920x1080)

The `OSWorldACI.resize_coordinates()` handles the scaling automatically.

---

## File Reference

### Created files

| File | Purpose |
|---|---|
| `Agent-S/gui_agents/s3/remote/__init__.py` | Package init |
| `Agent-S/gui_agents/s3/remote/bytebot_client.py` | HTTP client for bytebotd REST API |
| `Agent-S/gui_agents/s3/remote/remote_pyautogui.py` | Drop-in pyautogui/pyperclip/subprocess replacements |
| `Agent-S/gui_agents/s3/remote/bytebot_env.py` | DesktopEnv-compatible adapter for OSWorld eval |
| `Agent-S/gui_agents/s3/cli_app_remote.py` | Remote CLI entry point |
| `pyproject.toml` | Unified uv project with all dependencies |
| `.env` | API keys and endpoint URLs |

### Modified files

| File | Change |
|---|---|
| `Agent-S/gui_agents/s3/core/engine.py` | Added `AnthropicFoundry` import; `LMMEngineAnthropic` uses `AnthropicFoundry` when `base_url` is set |
| `Agent-S/osworld_setup/s3/run_local.py` | Added `--provider_name bytebot` support with `BytebotDesktopEnv`; lazy-imported `DesktopEnv` |

### Unchanged files (reference)

| File | Purpose |
|---|---|
| `Agent-S/gui_agents/s3/cli_app.py` | Original local CLI (for machines with a display) |
| `Agent-S/gui_agents/s3/agents/agent_s.py` | `AgentS3` orchestrator — unchanged |
| `Agent-S/gui_agents/s3/agents/worker.py` | Worker agent — unchanged |
| `Agent-S/gui_agents/s3/agents/grounding.py` | `OSWorldACI` with `@agent_action` methods — unchanged |
| `Agent-S/gui_agents/s3/core/mllm.py` | `LMMAgent` wrapper — unchanged |
| `Agent-S/gui_agents/s3/utils/common_utils.py` | `create_pyautogui_code()`, parsing — unchanged |
| `Agent-S/gui_agents/s3/memory/procedural_memory.py` | System prompt construction — unchanged |
| `Agent-S/osworld_setup/s3/lib_run_single.py` | Single task executor — unchanged |
| `bytebot/docker/docker-compose.development.yml` | Docker Compose for bytebot-desktop |
| `bytebot/packages/bytebotd/Dockerfile` | Container image build |
