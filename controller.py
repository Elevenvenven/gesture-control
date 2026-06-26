"""Mouse and keyboard control with cursor smoothing."""

from __future__ import annotations

import pyautogui

pyautogui.FAILSAFE = True


class CursorSmoother:
    """Exponential moving average for smoother cursor movement."""

    def __init__(self, alpha: float = 0.55) -> None:
        self.alpha = alpha
        self._x: float | None = None
        self._y: float | None = None

    def reset(self) -> None:
        self._x = None
        self._y = None

    def map_to_screen(
        self,
        norm_x: float,
        norm_y: float,
        margin: float,
        screen_w: int,
        screen_h: int,
    ) -> tuple[int, int]:
        span = 1.0 - margin * 2
        x = min(max((norm_x - margin) / span, 0.0), 1.0)
        y = min(max((norm_y - margin) / span, 0.0), 1.0)
        if self._x is None:
            self._x, self._y = x, y
        else:
            self._x = self.alpha * x + (1.0 - self.alpha) * self._x
            self._y = self.alpha * y + (1.0 - self.alpha) * self._y
        return int(self._x * screen_w), int(self._y * screen_h)


class InputController:
    def __init__(self, smooth_alpha: float = 0.55) -> None:
        self.smoother = CursorSmoother(alpha=smooth_alpha)
        self._screen_w, self._screen_h = pyautogui.size()
        self._last_pos: tuple[int, int] | None = None

    def reset_cursor(self) -> None:
        self.smoother.reset()
        self._last_pos = None

    def move_cursor(self, norm_x: float, norm_y: float, margin: float) -> None:
        x, y = self.smoother.map_to_screen(
            norm_x, norm_y, margin, self._screen_w, self._screen_h
        )
        if self._last_pos == (x, y):
            return
        pyautogui.moveTo(x, y, _pause=False)
        self._last_pos = (x, y)

    def click(self) -> None:
        pyautogui.click(_pause=False)

    def right_click(self) -> None:
        pyautogui.click(button="right", _pause=False)

    def double_click(self) -> None:
        pyautogui.doubleClick(_pause=False)

    def mouse_down(self) -> None:
        pyautogui.mouseDown(_pause=False)

    def mouse_up(self) -> None:
        pyautogui.mouseUp(_pause=False)

    def scroll(self, amount: int) -> None:
        pyautogui.scroll(amount, _pause=False)

    def navigate_back(self) -> None:
        pyautogui.hotkey("command", "[", _pause=False)

    def navigate_forward(self) -> None:
        pyautogui.hotkey("command", "]", _pause=False)
