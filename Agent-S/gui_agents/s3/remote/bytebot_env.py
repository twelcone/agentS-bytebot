"""DesktopEnv-compatible adapter that uses a bytebot Docker container as the
VM backend for OSWorld benchmark evaluation.

This replaces OSWorld's DesktopEnv (which uses VMware/AWS/Docker providers)
with an adapter that communicates with bytebotd's REST API, while exposing
the same interface that lib_run_single.py and run_local.py expect.

Handles OSWorld's structured task configs:
  - config[].type == "download"  → download URL, write file into container
  - config[].type == "execute"   → run shell command inside container
  - config[].type == "launch"    → run command (non-blocking)
  - config[].type == "command"   → alias for execute
  - config[].type == "sleep"     → time.sleep
  - config[].type == "open"      → open a file with xdg-open

Handles OSWorld's evaluator format:
  - result.type == "vm_command_line"  → run command, get stdout
  - result.type == "vm_file"         → read file from container
  - func == "exact_match"            → compare result to expected
  - func == "fuzzy_match"            → fuzzy string comparison
"""

import json
import logging
import os
import subprocess
import tempfile
import time
from typing import Any, Dict, List, Optional, Tuple

import requests

from gui_agents.s3.remote.bytebot_client import BytebotClient
from gui_agents.s3.remote.remote_pyautogui import (
    RemotePyAutoGUI,
    RemotePyperclip,
    RemoteSubprocess,
)

logger = logging.getLogger("desktopenv.bytebot_env")


class BytebotDesktopEnv:
    """DesktopEnv-compatible adapter backed by a bytebot Docker container.

    Implements the DesktopEnv methods used by Agent-S evaluation:
        - reset(task_config)
        - _get_obs()
        - step(action, sleep_after_execution)
        - evaluate()
        - close()
    """

    def __init__(
        self,
        bytebot_url: str = "http://localhost:9990",
        screen_size: Tuple[int, int] = (1280, 960),
        container_name: str = "bytebot-desktop",
        os_type: str = "Ubuntu",
        cache_dir: str = "cache",
        **kwargs,
    ):
        self.bytebot_url = bytebot_url
        self.screen_size = screen_size
        self.screen_width = screen_size[0]
        self.screen_height = screen_size[1]
        self.container_name = container_name
        self.os_type = os_type
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

        self.client = BytebotClient(
            base_url=bytebot_url,
            screen_width=screen_size[0],
            screen_height=screen_size[1],
        )
        self.remote_pyautogui = RemotePyAutoGUI(self.client)
        self.remote_pyperclip = RemotePyperclip(self.client, container_name=container_name)
        self.remote_subprocess = RemoteSubprocess(self.client, container_name=container_name)

        self._task_config = None
        self.action_history: List[str] = []
        self.controller = None  # Not used but referenced by some OSWorld code

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self, task_config: Optional[Dict[str, Any]] = None) -> Dict:
        """Reset the environment for a new task.

        Processes OSWorld's structured config list: each item has
        {"type": "<type>", "parameters": {...}}.
        """
        self._task_config = task_config
        self.action_history.clear()

        if task_config is None:
            return self._get_obs()

        config = task_config.get("config", [])
        for i, cfg in enumerate(config):
            config_type = cfg.get("type", "")
            parameters = cfg.get("parameters", {})
            setup_method = f"_setup_{config_type}"

            if hasattr(self, setup_method):
                try:
                    logger.info("Setup step %d/%d: %s", i + 1, len(config), config_type)
                    getattr(self, setup_method)(**parameters)
                except Exception as e:
                    logger.error("Setup step %d failed (%s): %s", i + 1, config_type, e)
            else:
                logger.warning("Unknown setup type '%s', skipping", config_type)

        return self._get_obs()

    # ------------------------------------------------------------------
    # Setup handlers (match OSWorld config types)
    # ------------------------------------------------------------------

    def _setup_download(self, files: List[Dict[str, str]], **kwargs):
        """Download files from URLs and place them inside the container."""
        for f in files:
            url = f["url"]
            path = f["path"]
            if not url or not path:
                logger.warning("Invalid download entry: url=%s path=%s", url, path)
                continue

            # Download to local cache
            cache_filename = os.path.basename(path)
            cache_path = os.path.join(self.cache_dir, cache_filename)

            if not os.path.exists(cache_path):
                logger.info("Downloading %s → %s", url, cache_path)
                try:
                    resp = requests.get(url, stream=True, timeout=300)
                    resp.raise_for_status()
                    with open(cache_path, "wb") as fp:
                        for chunk in resp.iter_content(chunk_size=8192):
                            fp.write(chunk)
                except Exception as e:
                    logger.error("Download failed for %s: %s", url, e)
                    continue
            else:
                logger.info("Using cached file: %s", cache_path)

            # Ensure parent directory exists inside container
            parent_dir = os.path.dirname(path)
            if parent_dir:
                self._docker_exec(f"mkdir -p {parent_dir}")

            # Copy file into container
            try:
                subprocess.run(
                    ["docker", "cp", cache_path, f"{self.container_name}:{path}"],
                    capture_output=True, timeout=60, check=True,
                )
                # Fix ownership
                self._docker_exec(f"chown user:user {path}")
                logger.info("Copied to container: %s", path)
            except Exception as e:
                logger.error("Failed to copy %s to container: %s", cache_path, e)

    def _setup_execute(
        self,
        command,
        shell: bool = False,
        stdout: str = "",
        stderr: str = "",
        until: Optional[Dict] = None,
        **kwargs,
    ):
        """Execute a command inside the container."""
        cmd_str = self._build_command_string(command)
        cmd_str = self._replace_screen_vars(cmd_str)
        logger.info("Execute: %s", cmd_str)

        max_attempts = 5 if until else 1
        for attempt in range(max_attempts):
            result = self._docker_exec(cmd_str, timeout=120)
            if result is None:
                continue

            if stdout and result.stdout:
                with open(os.path.join(self.cache_dir, stdout), "w") as fp:
                    fp.write(result.stdout)
            if stderr and result.stderr:
                with open(os.path.join(self.cache_dir, stderr), "w") as fp:
                    fp.write(result.stderr)

            if not until:
                break

            # Check termination conditions
            if "returncode" in until and result.returncode == until["returncode"]:
                break
            if "stdout" in until and until["stdout"] in (result.stdout or ""):
                break
            if "stderr" in until and until["stderr"] in (result.stderr or ""):
                break

            time.sleep(0.3)

    def _setup_command(self, **kwargs):
        """Alias for execute."""
        self._setup_execute(**kwargs)

    def _setup_launch(self, command, shell: bool = False, **kwargs):
        """Launch a command (non-blocking) inside the container."""
        cmd_str = self._build_command_string(command)
        cmd_str = self._replace_screen_vars(cmd_str)
        # Run in background with nohup
        logger.info("Launch: %s", cmd_str)
        self._docker_exec(f"nohup {cmd_str} &>/dev/null &", timeout=10)

    def _setup_open(self, path: str, **kwargs):
        """Open a file with xdg-open inside the container."""
        logger.info("Open: %s", path)
        self._docker_exec(f"DISPLAY=:0 xdg-open '{path}' &>/dev/null &", timeout=10)
        time.sleep(3)

    def _setup_sleep(self, seconds: float = 1.0, **kwargs):
        """Sleep for a specified duration."""
        logger.info("Sleep: %.1fs", seconds)
        time.sleep(seconds)

    def _setup_activate_window(self, window_name: str, strict: bool = False, by_class: bool = False, **kwargs):
        """Activate a window by name using wmctrl."""
        if by_class:
            self._docker_exec(f"DISPLAY=:0 wmctrl -x -a '{window_name}'")
        elif strict:
            self._docker_exec(f"DISPLAY=:0 wmctrl -a '{window_name}'")
        else:
            self._docker_exec(f"DISPLAY=:0 wmctrl -a '{window_name}'")

    def _setup_close_window(self, window_name: str, strict: bool = False, by_class: bool = False, **kwargs):
        """Close a window by name using wmctrl."""
        if by_class:
            self._docker_exec(f"DISPLAY=:0 wmctrl -x -c '{window_name}'")
        else:
            self._docker_exec(f"DISPLAY=:0 wmctrl -c '{window_name}'")

    def _setup_chrome_open_tabs(self, urls_to_open: List[str], **kwargs):
        """Open URLs in Firefox (bytebot uses Firefox, not Chrome)."""
        for url in urls_to_open:
            logger.info("Opening URL: %s", url)
            self._docker_exec(f"DISPLAY=:0 firefox-esr '{url}' &>/dev/null &", timeout=10)
            time.sleep(2)

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def _get_obs(self) -> Dict:
        """Get the current observation (screenshot) from the remote VM."""
        screenshot_bytes = self.client.screenshot()
        return {"screenshot": screenshot_bytes}

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(self, action: str, sleep_after_execution: float = 1.0) -> Tuple:
        """Execute an action on the remote VM and return the new observation."""
        import sys as _sys

        self.action_history.append(action)

        # Handle special action strings
        if isinstance(action, str) and action.strip().upper() in ("WAIT", "FAIL", "DONE"):
            done = action.strip().upper() == "DONE"
            time.sleep(sleep_after_execution)
            return self._get_obs(), 0, done, {}

        # Temporarily inject remote modules for the exec'd code
        orig_pyautogui = _sys.modules.get("pyautogui")
        orig_pyperclip = _sys.modules.get("pyperclip")
        orig_subprocess = _sys.modules.get("subprocess")

        _sys.modules["pyautogui"] = self.remote_pyautogui
        _sys.modules["pyperclip"] = self.remote_pyperclip
        _sys.modules["subprocess"] = self.remote_subprocess

        try:
            exec(action)
        except Exception as e:
            logger.error("Action execution error: %s\nAction: %s", e, action)
        finally:
            if orig_pyautogui is not None:
                _sys.modules["pyautogui"] = orig_pyautogui
            elif "pyautogui" in _sys.modules:
                del _sys.modules["pyautogui"]
            if orig_pyperclip is not None:
                _sys.modules["pyperclip"] = orig_pyperclip
            elif "pyperclip" in _sys.modules:
                del _sys.modules["pyperclip"]
            if orig_subprocess is not None:
                _sys.modules["subprocess"] = orig_subprocess

        time.sleep(sleep_after_execution)
        obs = self._get_obs()
        return obs, 0, False, {}

    # ------------------------------------------------------------------
    # Evaluate
    # ------------------------------------------------------------------

    def evaluate(self) -> float:
        """Run evaluation for the current task.

        Handles OSWorld evaluator format:
          evaluator.func: metric name (exact_match, fuzzy_match, etc.)
          evaluator.result: how to get actual result (vm_command_line, vm_file)
          evaluator.expected: what the expected result is (rule, vm_command_line)
          evaluator.postconfig: setup steps to run before evaluation
        """
        if self._task_config is None:
            logger.warning("No task config for evaluation")
            return 0.0

        evaluator = self._task_config.get("evaluator", {})
        func_name = evaluator.get("func", "")

        if not func_name:
            logger.warning("No evaluator function in task config")
            return 0.0

        # Run postconfig if present
        postconfig = evaluator.get("postconfig", [])
        for i, cfg in enumerate(postconfig):
            config_type = cfg.get("type", "")
            parameters = cfg.get("parameters", {})
            setup_method = f"_setup_{config_type}"
            if hasattr(self, setup_method):
                try:
                    getattr(self, setup_method)(**parameters)
                except Exception as e:
                    logger.error("Postconfig step %d failed: %s", i + 1, e)

        # Handle infeasible tasks
        if func_name == "infeasible":
            if self.action_history and "FAIL" in str(self.action_history[-1]).upper():
                return 1.0
            return 0.0

        # Check if last action was FAIL
        if self.action_history and "FAIL" in str(self.action_history[-1]).upper():
            return 0.0

        # Get actual result
        result_config = evaluator.get("result", {})
        actual = self._get_result(result_config)

        # Get expected result
        expected_config = evaluator.get("expected", {})
        expected = self._get_expected(expected_config)

        # Compare using the metric function
        score = self._evaluate_metric(func_name, actual, expected, evaluator.get("options", {}))
        logger.info("Evaluation: func=%s, score=%.2f", func_name, score)
        return score

    def _get_result(self, config: Dict) -> Any:
        """Get the actual result from the VM based on the result config."""
        result_type = config.get("type", "")

        if result_type == "vm_command_line":
            command = config.get("command", "")
            if isinstance(command, list):
                command = " ".join(command)
            command = self._replace_screen_vars(command)
            result = self._docker_exec(command, timeout=60)
            return result.stdout if result else ""

        elif result_type == "vm_command_error":
            command = config.get("command", "")
            if isinstance(command, list):
                command = " ".join(command)
            command = self._replace_screen_vars(command)
            result = self._docker_exec(command, timeout=60)
            return result.stderr if result else ""

        elif result_type == "vm_file":
            path = config.get("path", "")
            if not path:
                return None
            # Copy file from container to local temp
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                tmp_path = tmp.name
            try:
                subprocess.run(
                    ["docker", "cp", f"{self.container_name}:{path}", tmp_path],
                    capture_output=True, timeout=30, check=True,
                )
                with open(tmp_path, "rb") as fp:
                    return fp.read()
            except Exception as e:
                logger.error("Failed to get vm_file %s: %s", path, e)
                return None
            finally:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)

        elif result_type == "vm_screen_size":
            return {"width": self.screen_width, "height": self.screen_height}

        elif result_type == "list_directory":
            path = config.get("path", "/home/user")
            result = self._docker_exec(f"ls -1 '{path}'", timeout=30)
            if result and result.stdout:
                return result.stdout.strip().split("\n")
            return []

        else:
            logger.warning("Unsupported result type: %s", result_type)
            return None

    def _get_expected(self, config: Dict) -> Any:
        """Get the expected result based on the expected config."""
        if not config:
            return None

        expected_type = config.get("type", "")

        if expected_type == "rule":
            rules = config.get("rules", {})
            return rules.get("expected", rules)

        elif expected_type == "vm_command_line":
            command = config.get("command", "")
            if isinstance(command, list):
                command = " ".join(command)
            command = self._replace_screen_vars(command)
            result = self._docker_exec(command, timeout=60)
            return result.stdout if result else ""

        elif expected_type == "vm_file":
            path = config.get("path", "")
            if not path:
                return None
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                tmp_path = tmp.name
            try:
                subprocess.run(
                    ["docker", "cp", f"{self.container_name}:{path}", tmp_path],
                    capture_output=True, timeout=30, check=True,
                )
                with open(tmp_path, "rb") as fp:
                    return fp.read()
            except Exception as e:
                logger.error("Failed to get expected vm_file %s: %s", path, e)
                return None
            finally:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)

        elif expected_type == "cloud_file":
            url = config.get("path", "") or config.get("url", "")
            if not url:
                return None
            try:
                resp = requests.get(url, timeout=60)
                resp.raise_for_status()
                return resp.content
            except Exception as e:
                logger.error("Failed to download expected cloud_file: %s", e)
                return None

        else:
            logger.warning("Unsupported expected type: %s", expected_type)
            return config

    def _evaluate_metric(self, func_name: str, actual: Any, expected: Any, options: Dict = None) -> float:
        """Evaluate using the specified metric function."""
        options = options or {}

        if func_name == "exact_match":
            if actual is None or expected is None:
                return 0.0
            actual_str = str(actual).strip() if not isinstance(actual, str) else actual.strip()
            if isinstance(expected, list):
                return 1.0 if actual_str in [str(e).strip() for e in expected] else 0.0
            expected_str = str(expected).strip()
            return 1.0 if actual_str == expected_str else 0.0

        elif func_name == "match_in_list":
            if actual is None or expected is None:
                return 0.0
            actual_str = str(actual).strip()
            if isinstance(expected, list):
                return 1.0 if any(str(e).strip() in actual_str for e in expected) else 0.0
            return 1.0 if str(expected).strip() in actual_str else 0.0

        elif func_name == "fuzzy_match":
            if actual is None or expected is None:
                return 0.0
            actual_lower = str(actual).strip().lower()
            if isinstance(expected, list):
                return 1.0 if any(str(e).strip().lower() in actual_lower for e in expected) else 0.0
            return 1.0 if str(expected).strip().lower() in actual_lower else 0.0

        elif func_name == "is_in_list":
            if actual is None or expected is None:
                return 0.0
            actual_str = str(actual).strip()
            if isinstance(expected, list):
                return 1.0 if actual_str in [str(e).strip() for e in expected] else 0.0
            return 1.0 if str(expected).strip() in actual_str else 0.0

        else:
            # Fallback: try exact string comparison
            logger.warning("Unknown metric '%s', falling back to exact_match", func_name)
            if actual is None or expected is None:
                return 0.0
            return 1.0 if str(actual).strip() == str(expected).strip() else 0.0

    # ------------------------------------------------------------------
    # Close
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Clean up resources. The container is left running for reuse."""
        pass

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _docker_exec(self, cmd: str, timeout: int = 60, user: str = "user") -> Optional[subprocess.CompletedProcess]:
        """Execute a command inside the bytebot container via docker exec."""
        try:
            return subprocess.run(
                ["docker", "exec", "-u", user, "-e", "DISPLAY=:0", self.container_name, "bash", "-c", cmd],
                capture_output=True, text=True, timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            logger.warning("docker exec timed out: %s", cmd[:100])
            return None
        except Exception as e:
            logger.error("docker exec error: %s — %s", cmd[:100], e)
            return None

    def _build_command_string(self, command) -> str:
        """Convert command (str or list) to a single string."""
        if isinstance(command, str):
            return command
        return " ".join(str(c) for c in command)

    def _replace_screen_vars(self, cmd: str) -> str:
        """Replace OSWorld template variables in commands."""
        cmd = cmd.replace("{CLIENT_PASSWORD}", "password")
        cmd = cmd.replace("{SCREEN_WIDTH}", str(self.screen_width))
        cmd = cmd.replace("{SCREEN_HEIGHT}", str(self.screen_height))
        cmd = cmd.replace("{SCREEN_WIDTH_HALF}", str(self.screen_width // 2))
        cmd = cmd.replace("{SCREEN_HEIGHT_HALF}", str(self.screen_height // 2))
        return cmd
