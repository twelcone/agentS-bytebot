"""HTTP client for the bytebotd REST API (port 9990).

bytebotd runs inside a Docker container and provides a POST /computer-use
endpoint that accepts JSON action payloads for mouse, keyboard, screenshot,
and other desktop control operations.
"""

import base64
import logging
import urllib.request
import urllib.error
import json
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("desktopenv.bytebot_client")


class BytebotClient:
    """Thin HTTP client wrapping the bytebotd REST API."""

    def __init__(self, base_url: str, screen_width: int = 1280, screen_height: int = 960):
        self.base_url = base_url.rstrip("/")
        self.endpoint = f"{self.base_url}/computer-use"
        self.screen_width = screen_width
        self.screen_height = screen_height

    def _post(self, payload: dict) -> dict:
        """Send a POST request to the /computer-use endpoint."""
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.endpoint,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                body = resp.read().decode("utf-8")
                if body:
                    return json.loads(body)
                return {}
        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8", errors="replace")
            logger.error("bytebotd HTTP %d: %s", e.code, error_body)
            raise
        except urllib.error.URLError as e:
            logger.error("bytebotd connection error: %s", e.reason)
            raise

    # ------------------------------------------------------------------
    # Screenshot
    # ------------------------------------------------------------------

    def screenshot(self) -> bytes:
        """Take a screenshot and return raw PNG bytes."""
        resp = self._post({"action": "screenshot"})
        b64_data = resp.get("image", "")
        return base64.b64decode(b64_data)

    # ------------------------------------------------------------------
    # Mouse actions
    # ------------------------------------------------------------------

    def click(
        self,
        x: int,
        y: int,
        button: str = "left",
        clicks: int = 1,
        hold_keys: Optional[List[str]] = None,
    ) -> None:
        payload: Dict = {
            "action": "click_mouse",
            "coordinates": {"x": int(x), "y": int(y)},
            "button": button,
            "clickCount": clicks,
        }
        if hold_keys:
            payload["holdKeys"] = hold_keys
        self._post(payload)

    def move_mouse(self, x: int, y: int) -> None:
        self._post({
            "action": "move_mouse",
            "coordinates": {"x": int(x), "y": int(y)},
        })

    def drag(
        self,
        path: List[Dict[str, int]],
        button: str = "left",
        hold_keys: Optional[List[str]] = None,
    ) -> None:
        payload: Dict = {
            "action": "drag_mouse",
            "path": path,
            "button": button,
        }
        if hold_keys:
            payload["holdKeys"] = hold_keys
        self._post(payload)

    def press_mouse(self, button: str = "left", press: str = "up") -> None:
        self._post({
            "action": "press_mouse",
            "button": button,
            "press": press,
        })

    def scroll(
        self,
        direction: str,
        count: int,
        x: Optional[int] = None,
        y: Optional[int] = None,
        hold_keys: Optional[List[str]] = None,
    ) -> None:
        payload: Dict = {
            "action": "scroll",
            "direction": direction,
            "scrollCount": abs(count),
        }
        if x is not None and y is not None:
            payload["coordinates"] = {"x": int(x), "y": int(y)}
        if hold_keys:
            payload["holdKeys"] = hold_keys
        self._post(payload)

    # ------------------------------------------------------------------
    # Keyboard actions
    # ------------------------------------------------------------------

    def type_text(self, text: str, delay: Optional[int] = None) -> None:
        payload: Dict = {"action": "type_text", "text": text}
        if delay is not None:
            payload["delay"] = delay
        self._post(payload)

    def paste_text(self, text: str) -> None:
        self._post({"action": "paste_text", "text": text})

    def type_keys(self, keys: List[str], delay: Optional[int] = None) -> None:
        """Press and release keys sequentially (like pressing Enter, Tab, etc.)."""
        payload: Dict = {"action": "type_keys", "keys": keys}
        if delay is not None:
            payload["delay"] = delay
        self._post(payload)

    def press_keys(self, keys: List[str], press: str = "down") -> None:
        """Hold or release keys (press='down' or press='up')."""
        self._post({"action": "press_keys", "keys": keys, "press": press})

    # ------------------------------------------------------------------
    # Other actions
    # ------------------------------------------------------------------

    def wait(self, duration_ms: int) -> None:
        self._post({"action": "wait", "duration": duration_ms})

    def get_screen_size(self) -> Tuple[int, int]:
        return (self.screen_width, self.screen_height)

    def health_check(self) -> bool:
        """Check if bytebotd is responsive by taking a screenshot."""
        try:
            self.screenshot()
            return True
        except Exception:
            return False
