from __future__ import annotations

import time
from pathlib import Path

import yaml


class ActionMapper:
    def __init__(self, config_path: str | Path, execute: bool) -> None:
        payload = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
        self.intents: dict = payload.get("intents", {})
        self.execute = execute
        self.paused = False
        self._last_action_at: dict[str, float] = {}

        self._pyautogui = None
        if execute:
            import pyautogui

            pyautogui.FAILSAFE = True
            self._pyautogui = pyautogui

    def maybe_execute(self, label: str, confidence: float) -> str:
        if label == "unknown":
            return "unknown"

        spec = self.intents.get(label)
        if not spec:
            return f"unmapped:{label}"

        now = time.monotonic()
        cooldown = float(spec.get("cooldown", 0.4))
        last = self._last_action_at.get(label, 0.0)
        if now - last < cooldown:
            return f"cooldown:{label}"

        action = str(spec.get("action", "noop"))
        if self.paused and action != "pause":
            return "paused"

        self._last_action_at[label] = now
        description = self._describe(label, confidence, spec)

        if not self.execute:
            print(f"[dry-run] {description}")
            return description

        pg = self._pyautogui
        if pg is None:
            raise RuntimeError("pyautogui is not initialized.")

        if action == "click":
            pg.click()
        elif action == "double_click":
            pg.doubleClick()
        elif action == "right_click":
            pg.rightClick()
        elif action == "scroll":
            pg.scroll(int(spec.get("amount", 0)))
        elif action == "hotkey":
            pg.hotkey(*spec.get("keys", []))
        elif action == "pause":
            self.paused = not self.paused
        elif action == "noop":
            pass
        else:
            return f"unsupported:{action}"

        print(description)
        return description

    def _describe(self, label: str, confidence: float, spec: dict) -> str:
        action = str(spec.get("action", "noop"))
        if action == "hotkey":
            return f"{label} ({confidence:.2f}) -> hotkey({'+'.join(spec.get('keys', []))})"
        if action == "scroll":
            return f"{label} ({confidence:.2f}) -> scroll({spec.get('amount', 0)})"
        if action == "pause":
            next_state = "resume" if self.paused else "pause"
            return f"{label} ({confidence:.2f}) -> {next_state}"
        return f"{label} ({confidence:.2f}) -> {action}"

