"""Remote CLI entry point for Agent-S3 controlling a bytebot Docker VM.

This module mirrors cli_app.py but replaces local pyautogui/subprocess calls
with HTTP requests to bytebotd's REST API. The remote VM provides the desktop
environment (Xvfb + XFCE4) while Agent-S runs externally.

Usage:
    python -m gui_agents.s3.cli_app_remote \
        --bytebot_url http://localhost:9990 \
        --screen_width 1280 --screen_height 960 \
        --provider openai --model gpt-5-2025-08-07 \
        --ground_provider vllm \
        --ground_url http://<vllm-server>:8000/v1 \
        --ground_model ui-tars-1.5-7b \
        --grounding_width 1920 --grounding_height 1080 \
        --task "Open Firefox and search for hello world"
"""

import argparse
import datetime
import io
import logging
import os
import signal
import sys
import time

from PIL import Image

from gui_agents.s3.agents.grounding import OSWorldACI
from gui_agents.s3.agents.agent_s import AgentS3
from gui_agents.s3.utils.local_env import LocalEnv
from gui_agents.s3.remote.bytebot_client import BytebotClient
from gui_agents.s3.remote.remote_pyautogui import (
    RemotePyAutoGUI,
    RemotePyperclip,
    RemoteSubprocess,
)

# Global flag to track pause state for debugging
paused = False


def get_char():
    """Get a single character from stdin without pressing Enter"""
    try:
        import termios
        import tty

        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch
    except Exception:
        return input()


def signal_handler(signum, frame):
    """Handle Ctrl+C signal for debugging during agent execution"""
    global paused

    if not paused:
        print("\n\nAgent-S Remote Workflow Paused")
        print("=" * 50)
        print("Options:")
        print("  - Press Ctrl+C again to quit")
        print("  - Press Esc to resume workflow")
        print("=" * 50)

        paused = True

        while paused:
            try:
                print("\n[PAUSED] Waiting for input... ", end="", flush=True)
                char = get_char()

                if ord(char) == 3:  # Ctrl+C
                    print("\n\nExiting Agent-S...")
                    sys.exit(0)
                elif ord(char) == 27:  # Esc
                    print("\n\nResuming Agent-S workflow...")
                    paused = False
                    break
                else:
                    print(f"\n   Unknown command: '{char}' (ord: {ord(char)})")

            except KeyboardInterrupt:
                print("\n\nExiting Agent-S...")
                sys.exit(0)
    else:
        print("\n\nExiting Agent-S...")
        sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)

# ------------------------------------------------------------------
# Logging setup
# ------------------------------------------------------------------

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

datetime_str: str = datetime.datetime.now().strftime("%Y%m%d@%H%M%S")

log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

file_handler = logging.FileHandler(
    os.path.join("logs", "normal-{:}.log".format(datetime_str)), encoding="utf-8"
)
debug_handler = logging.FileHandler(
    os.path.join("logs", "debug-{:}.log".format(datetime_str)), encoding="utf-8"
)
stdout_handler = logging.StreamHandler(sys.stdout)
sdebug_handler = logging.FileHandler(
    os.path.join("logs", "sdebug-{:}.log".format(datetime_str)), encoding="utf-8"
)

file_handler.setLevel(logging.INFO)
debug_handler.setLevel(logging.DEBUG)
stdout_handler.setLevel(logging.INFO)
sdebug_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter(
    fmt="\x1b[1;33m[%(asctime)s \x1b[31m%(levelname)s \x1b[32m%(module)s/%(lineno)d-%(processName)s\x1b[1;33m] \x1b[0m%(message)s"
)
file_handler.setFormatter(formatter)
debug_handler.setFormatter(formatter)
stdout_handler.setFormatter(formatter)
sdebug_handler.setFormatter(formatter)

stdout_handler.addFilter(logging.Filter("desktopenv"))
sdebug_handler.addFilter(logging.Filter("desktopenv"))

logger.addHandler(file_handler)
logger.addHandler(debug_handler)
logger.addHandler(stdout_handler)
logger.addHandler(sdebug_handler)


def scale_screen_dimensions(width: int, height: int, max_dim_size: int):
    scale_factor = min(max_dim_size / width, max_dim_size / height, 1)
    safe_width = int(width * scale_factor)
    safe_height = int(height * scale_factor)
    return safe_width, safe_height


def run_agent(
    agent,
    instruction: str,
    scaled_width: int,
    scaled_height: int,
    remote_pyautogui: RemotePyAutoGUI,
):
    global paused
    obs = {}
    traj = "Task:\n" + instruction

    for step in range(15):
        while paused:
            time.sleep(0.1)

        # Get screenshot from remote VM
        screenshot = remote_pyautogui.screenshot()
        screenshot = screenshot.resize((scaled_width, scaled_height), Image.LANCZOS)

        buffered = io.BytesIO()
        screenshot.save(buffered, format="PNG")
        screenshot_bytes = buffered.getvalue()
        obs["screenshot"] = screenshot_bytes

        while paused:
            time.sleep(0.1)

        print(f"\nStep {step + 1}/15: Getting next action from agent...")

        info, code = agent.predict(instruction=instruction, observation=obs)

        if "done" in code[0].lower() or "fail" in code[0].lower():
            print("Task completed." if "done" in code[0].lower() else "Task failed.")
            break

        if "next" in code[0].lower():
            continue

        if "wait" in code[0].lower():
            print("Agent requested wait...")
            time.sleep(5)
            continue

        time.sleep(1.0)
        print("EXECUTING CODE:", code[0])

        while paused:
            time.sleep(0.1)

        # Execute in a namespace where pyautogui, pyperclip, subprocess are
        # the remote adapters. The import statements inside exec'd code will
        # resolve from sys.modules (already patched in main()).
        exec(code[0])
        time.sleep(1.0)

        if "reflection" in info and "executor_plan" in info:
            traj += (
                "\n\nReflection:\n"
                + str(info["reflection"])
                + "\n\n----------------------\n\nPlan:\n"
                + info["executor_plan"]
            )


def main():
    parser = argparse.ArgumentParser(
        description="Run AgentS3 against a remote bytebot VM."
    )

    # Bytebot connection
    parser.add_argument(
        "--bytebot_url",
        type=str,
        required=True,
        help="URL of the bytebotd REST API (e.g. http://localhost:9990)",
    )
    parser.add_argument(
        "--screen_width",
        type=int,
        default=1280,
        help="Width of the remote VM display (default: 1280, matching bytebot Xvfb)",
    )
    parser.add_argument(
        "--screen_height",
        type=int,
        default=960,
        help="Height of the remote VM display (default: 960, matching bytebot Xvfb)",
    )
    parser.add_argument(
        "--container_name",
        type=str,
        default="bytebot-desktop",
        help="Docker container name for subprocess execution (default: bytebot-desktop)",
    )

    # Main LLM config
    parser.add_argument(
        "--provider", type=str, default="openai",
        help="Main LLM provider (openai, anthropic, vllm, etc.)",
    )
    parser.add_argument(
        "--model", type=str, default="gpt-5-2025-08-07",
        help="Main LLM model name",
    )
    parser.add_argument("--model_url", type=str, default="")
    parser.add_argument("--model_api_key", type=str, default="")
    parser.add_argument("--model_temperature", type=float, default=None)

    # Grounding model config
    parser.add_argument("--ground_provider", type=str, required=True)
    parser.add_argument("--ground_url", type=str, required=True)
    parser.add_argument("--ground_api_key", type=str, default="")
    parser.add_argument("--ground_model", type=str, required=True)
    parser.add_argument("--grounding_width", type=int, required=True)
    parser.add_argument("--grounding_height", type=int, required=True)

    # Agent config
    parser.add_argument("--max_trajectory_length", type=int, default=8)
    parser.add_argument("--enable_reflection", action="store_true", default=True)
    parser.add_argument("--enable_local_env", action="store_true", default=False)
    parser.add_argument("--task", type=str, help="Task instruction for Agent-S3")

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Set up remote connection
    # ------------------------------------------------------------------

    client = BytebotClient(
        base_url=args.bytebot_url,
        screen_width=args.screen_width,
        screen_height=args.screen_height,
    )

    print(f"Connecting to bytebot VM at {args.bytebot_url}...")
    if not client.health_check():
        print(f"ERROR: Cannot connect to bytebotd at {args.bytebot_url}")
        print("Make sure the bytebot desktop container is running:")
        print("  cd bytebot/docker && docker compose -f docker-compose.development.yml up -d bytebot-desktop")
        sys.exit(1)
    print(f"Connected. Screen size: {args.screen_width}x{args.screen_height}")

    # ------------------------------------------------------------------
    # Inject remote modules into sys.modules
    # ------------------------------------------------------------------

    remote_pyautogui = RemotePyAutoGUI(client)
    remote_pyperclip = RemotePyperclip(client, container_name=args.container_name)
    remote_subprocess = RemoteSubprocess(client, container_name=args.container_name)

    # Save originals for potential restoration
    _orig_pyautogui = sys.modules.get("pyautogui")
    _orig_pyperclip = sys.modules.get("pyperclip")
    _orig_subprocess = sys.modules.get("subprocess")

    # Inject remote adapters — any `import pyautogui` etc. in exec'd code
    # will now resolve to these objects
    sys.modules["pyautogui"] = remote_pyautogui
    sys.modules["pyperclip"] = remote_pyperclip
    sys.modules["subprocess"] = remote_subprocess

    # ------------------------------------------------------------------
    # Configure Agent-S3
    # ------------------------------------------------------------------

    screen_width, screen_height = args.screen_width, args.screen_height
    scaled_width, scaled_height = scale_screen_dimensions(
        screen_width, screen_height, max_dim_size=2400
    )

    engine_params = {
        "engine_type": args.provider,
        "model": args.model,
        "base_url": args.model_url,
        "api_key": args.model_api_key,
        "temperature": getattr(args, "model_temperature", None),
    }

    engine_params_for_grounding = {
        "engine_type": args.ground_provider,
        "model": args.ground_model,
        "base_url": args.ground_url,
        "api_key": args.ground_api_key,
        "grounding_width": args.grounding_width,
        "grounding_height": args.grounding_height,
    }

    local_env = None
    if args.enable_local_env:
        print("WARNING: Local coding environment enabled. Code will execute inside the bytebot container.")
        local_env = LocalEnv()

    grounding_agent = OSWorldACI(
        env=local_env,
        platform="linux",
        engine_params_for_generation=engine_params,
        engine_params_for_grounding=engine_params_for_grounding,
        width=screen_width,
        height=screen_height,
    )

    agent = AgentS3(
        engine_params,
        grounding_agent,
        platform="linux",
        max_trajectory_length=args.max_trajectory_length,
        enable_reflection=args.enable_reflection,
    )

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    task = args.task

    if isinstance(task, str) and task.strip():
        agent.reset()
        run_agent(agent, task, scaled_width, scaled_height, remote_pyautogui)
        return

    while True:
        query = input("Query: ")
        agent.reset()
        run_agent(agent, query, scaled_width, scaled_height, remote_pyautogui)

        response = input("Would you like to provide another query? (y/n): ")
        if response.lower() != "y":
            break


if __name__ == "__main__":
    main()
