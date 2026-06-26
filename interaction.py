"""Interaction state machine — maps hand tracking to user actions."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from collections import deque

from controller import InputController
from gestures import GestureType, HandFrame, SwipeState


class InteractionMode(Enum):
    PAUSED = "paused"
    READY = "ready"
    POINTER = "pointer"
    SCROLL = "scroll"
    PINCH = "pinch"
    DRAG = "drag"


class ActionType(Enum):
    NONE = "none"
    CLICK = "click"
    RIGHT_CLICK = "right_click"
    DOUBLE_CLICK = "double_click"
    SCROLL_UP = "scroll_up"
    SCROLL_DOWN = "scroll_down"
    NAV_BACK = "nav_back"
    NAV_FORWARD = "nav_forward"
    PAUSE = "pause"
    RESUME = "resume"


@dataclass
class InteractionConfig:
    pause_hold: float = 0.6
    pause_cooldown: float = 1.2
    scroll_enter_frames: int = 5
    scroll_exit_frames: int = 3
    pointer_index_min: float = 0.40
    min_confidence: float = 0.48
    quality_min: float = 0.35
    scroll_threshold: float = 0.085
    nav_threshold: float = 0.13
    scroll_amount: int = 5
    scroll_cooldown: float = 0.22
    nav_cooldown: float = 0.8
    click_cooldown: float = 0.45
    right_click_cooldown: float = 0.8
    double_click_cooldown: float = 0.8
    drag_hold: float = 0.38
    discrete_lockout: float = 0.35
    swipe_frames: int = 7
    cursor_margin: float = 0.12
    stable_frames: int = 3


@dataclass
class InteractionUpdate:
    mode: InteractionMode
    action: ActionType
    status: str
    active: bool
    cursor_x: float | None = None
    cursor_y: float | None = None


MODE_COLORS = {
    InteractionMode.PAUSED: (80, 80, 220),
    InteractionMode.READY: (80, 180, 255),
    InteractionMode.POINTER: (40, 220, 80),
    InteractionMode.SCROLL: (0, 200, 255),
    InteractionMode.PINCH: (0, 180, 255),
    InteractionMode.DRAG: (0, 140, 255),
}

MODE_HINTS = {
    InteractionMode.PAUSED: "握拳 0.6s 恢复",
    InteractionMode.READY: "伸出食指开始控制",
    InteractionMode.POINTER: "食指移动 | 捏合点击 | 握拳暂停",
    InteractionMode.SCROLL: "上下滑滚动 | 左右滑前进后退 | 收掌退出",
    InteractionMode.PINCH: "捏合中 — 松开点击",
    InteractionMode.DRAG: "拖拽中 — 松开放下",
}


def _quality_factor(quality: float) -> float:
    return 0.65 + 0.35 * max(0.0, min(1.0, quality))


def _cursor_point(landmarks) -> tuple[float, float]:
    tip, pip = landmarks[8], landmarks[6]
    blend = 0.78
    return tip.x * blend + pip.x * (1 - blend), tip.y * blend + pip.y * (1 - blend)


class GestureEdgeDetector:
    """Fire one-shot actions only when a gesture becomes stable."""

    def __init__(self, stable_frames: int = 3) -> None:
        self.stable_frames = stable_frames
        self._current = GestureType.NONE
        self._count = 0
        self._last_fired: GestureType | None = None

    def reset(self) -> None:
        self._current = GestureType.NONE
        self._count = 0
        self._last_fired = None

    def check(self, gesture: GestureType, confidence: float, min_conf: float) -> GestureType | None:
        if gesture == self._current:
            self._count += 1
        else:
            self._current = gesture
            self._count = 1

        if self._count >= self.stable_frames and confidence >= min_conf:
            if self._last_fired != gesture:
                self._last_fired = gesture
                return gesture
        return None


class InteractionEngine:
    """
    Interaction layers (priority top → bottom):

    1. Pause / resume — fist hold (global)
    2. Pinch / drag — click while pointing
    3. Scroll mode — open palm swipe (exclusive, blocks pointer)
    4. Pointer — default index-finger cursor
    5. Discrete shortcuts — two/three finger edge-triggered clicks
    """

    def __init__(self, config: InteractionConfig, input_ctrl: InputController) -> None:
        self.cfg = config
        self.input = input_ctrl
        self.swipe = SwipeState(maxlen=config.swipe_frames)
        self.edge = GestureEdgeDetector(config.stable_frames)

        self.mode = InteractionMode.READY
        self.paused = False
        self.dragging = False

        self._fist_hold_start: float | None = None
        self._pinch_start: float | None = None
        self._scroll_enter = 0
        self._scroll_exit = 0
        self._discrete_lock_until = 0.0

        self._last_scroll = 0.0
        self._last_nav = 0.0
        self._last_click = 0.0
        self._last_right = 0.0
        self._last_double = 0.0
        self._last_pause_toggle = 0.0

    def close(self) -> None:
        if self.dragging:
            self.input.mouse_up()
            self.dragging = False

    def _min_conf(self, quality: float) -> float:
        return self.cfg.min_confidence * _quality_factor(quality)

    def _in_discrete_lockout(self, now: float) -> bool:
        return now < self._discrete_lock_until

    def _lock_discrete(self, now: float) -> None:
        self._discrete_lock_until = now + self.cfg.discrete_lockout

    def _reset_pinch(self) -> None:
        self._pinch_start = None
        if self.dragging:
            self.input.mouse_up()
            self.dragging = False

    def _reset_scroll_counters(self) -> None:
        self._scroll_enter = 0
        self._scroll_exit = 0
        self.swipe.clear()

    def _try_pause(self, hand: HandFrame, now: float) -> InteractionUpdate | None:
        if not hand.is_closed_fist:
            self._fist_hold_start = None
            return None

        if self._fist_hold_start is None:
            self._fist_hold_start = now

        held = now - self._fist_hold_start
        if held < self.cfg.pause_hold:
            label = "恢复" if self.paused else "暂停"
            pct = int(held / self.cfg.pause_hold * 100)
            return InteractionUpdate(
                InteractionMode.PAUSED if self.paused else self.mode,
                ActionType.NONE,
                f"保持握拳以{label}... {pct}%",
                True,
            )

        if now - self._last_pause_toggle < self.cfg.pause_cooldown:
            return InteractionUpdate(self.mode, ActionType.NONE, "请稍候...", True)

        self.paused = not self.paused
        self._last_pause_toggle = now
        self._fist_hold_start = None
        self._reset_pinch()
        self._reset_scroll_counters()
        self.edge.reset()

        if self.paused:
            self.mode = InteractionMode.PAUSED
            return InteractionUpdate(InteractionMode.PAUSED, ActionType.PAUSE, "已暂停", True)

        self.mode = InteractionMode.READY
        return InteractionUpdate(InteractionMode.READY, ActionType.RESUME, "已恢复", True)

    def _update_scroll_mode(self, hand: HandFrame) -> None:
        is_open = hand.stable_gesture == GestureType.OPEN_PALM and hand.confidence >= self._min_conf(hand.quality) * 0.75

        if self.mode == InteractionMode.SCROLL:
            if is_open:
                self._scroll_exit = 0
            else:
                self._scroll_exit += 1
                if self._scroll_exit >= self.cfg.scroll_exit_frames:
                    self.mode = InteractionMode.POINTER
                    self._reset_scroll_counters()
        elif is_open and not hand.is_pinching:
            self._scroll_enter += 1
            if self._scroll_enter >= self.cfg.scroll_enter_frames:
                self.mode = InteractionMode.SCROLL
                self._scroll_exit = 0
                self.swipe.clear()
        else:
            self._scroll_enter = max(0, self._scroll_enter - 1)

    def _handle_scroll(self, hand: HandFrame, now: float) -> InteractionUpdate:
        self.swipe.append(hand.center_x, hand.center_y)
        if not self.swipe.ready:
            return InteractionUpdate(InteractionMode.SCROLL, ActionType.NONE, "滚动模式 — 滑动手掌", True)

        dx, dy = self.swipe.delta()
        can_nav = now - self._last_nav >= self.cfg.nav_cooldown
        can_scroll = now - self._last_scroll >= self.cfg.scroll_cooldown

        if abs(dx) >= self.cfg.nav_threshold and abs(dx) > abs(dy) * 1.2 and can_nav:
            if dx > 0:
                self.input.navigate_forward()
                self.swipe.clear()
                self._last_nav = now
                return InteractionUpdate(InteractionMode.SCROLL, ActionType.NAV_FORWARD, "前进", True)
            self.input.navigate_back()
            self.swipe.clear()
            self._last_nav = now
            return InteractionUpdate(InteractionMode.SCROLL, ActionType.NAV_BACK, "后退", True)

        if abs(dy) >= self.cfg.scroll_threshold and can_scroll:
            amount = -self.cfg.scroll_amount if dy > 0 else self.cfg.scroll_amount
            self.input.scroll(amount)
            self.swipe.clear()
            self._last_scroll = now
            direction = "下" if dy > 0 else "上"
            action = ActionType.SCROLL_DOWN if dy > 0 else ActionType.SCROLL_UP
            return InteractionUpdate(InteractionMode.SCROLL, action, f"滚动{direction}", True)

        return InteractionUpdate(InteractionMode.SCROLL, ActionType.NONE, "滚动模式 — 滑动手掌", True)

    def _handle_pinch(self, hand: HandFrame, now: float) -> InteractionUpdate:
        cx, cy = _cursor_point(hand.landmarks)
        self.input.move_cursor(cx, cy, self.cfg.cursor_margin)
        self.swipe.clear()

        if self._pinch_start is None:
            self._pinch_start = now
            self.mode = InteractionMode.PINCH
            return InteractionUpdate(InteractionMode.PINCH, ActionType.NONE, "捏合", True, cx, cy)

        if not self.dragging and now - self._pinch_start >= self.cfg.drag_hold:
            self.input.mouse_down()
            self.dragging = True
            self.mode = InteractionMode.DRAG
            return InteractionUpdate(InteractionMode.DRAG, ActionType.NONE, "拖拽", True, cx, cy)

        mode = InteractionMode.DRAG if self.dragging else InteractionMode.PINCH
        label = "拖拽中" if self.dragging else "捏合中 — 松开点击"
        return InteractionUpdate(mode, ActionType.NONE, label, True, cx, cy)

    def _release_pinch(self, now: float) -> InteractionUpdate | None:
        if self._pinch_start is None:
            return None

        action = ActionType.NONE
        status = "点击"
        if self.dragging:
            self.input.mouse_up()
            self.dragging = False
            status = "放下"
        elif now - self._last_click >= self.cfg.click_cooldown:
            self.input.click()
            self._last_click = now
            action = ActionType.CLICK
        else:
            status = "点击冷却中"

        self._pinch_start = None
        self.mode = InteractionMode.POINTER
        return InteractionUpdate(InteractionMode.POINTER, action, status, action != ActionType.NONE)

    def _handle_pointer(self, hand: HandFrame, now: float) -> InteractionUpdate:
        idx_ext = hand.finger_scores["index"].extended
        cx, cy = _cursor_point(hand.landmarks)

        if idx_ext >= self.cfg.pointer_index_min:
            self.input.move_cursor(cx, cy, self.cfg.cursor_margin)
            self.mode = InteractionMode.POINTER
            return InteractionUpdate(InteractionMode.POINTER, ActionType.NONE, "移动鼠标", True, cx, cy)

        self.mode = InteractionMode.READY
        return InteractionUpdate(InteractionMode.READY, ActionType.NONE, "伸出食指移动鼠标", False)

    def _try_discrete(self, hand: HandFrame, now: float) -> InteractionUpdate | None:
        if self._in_discrete_lockout(now):
            return None

        min_conf = self._min_conf(hand.quality)
        fired = self.edge.check(hand.stable_gesture, hand.confidence, min_conf)
        if fired is None:
            return None

        if fired == GestureType.TWO_FINGERS and now - self._last_right >= self.cfg.right_click_cooldown:
            self.input.right_click()
            self._last_right = now
            self._lock_discrete(now)
            return InteractionUpdate(self.mode, ActionType.RIGHT_CLICK, "右键", True)

        if fired == GestureType.THREE_FINGERS and now - self._last_double >= self.cfg.double_click_cooldown:
            self.input.double_click()
            self._last_double = now
            self._lock_discrete(now)
            return InteractionUpdate(self.mode, ActionType.DOUBLE_CLICK, "双击", True)

        return None

    def update(self, hand: HandFrame | None, now: float) -> InteractionUpdate:
        if hand is None:
            self._fist_hold_start = None
            self._reset_pinch()
            self._reset_scroll_counters()
            self.edge.reset()
            self.input.reset_cursor()
            self.mode = InteractionMode.READY
            return InteractionUpdate(InteractionMode.READY, ActionType.NONE, "请伸出手", False)

        if hand.quality < self.cfg.quality_min:
            return InteractionUpdate(self.mode, ActionType.NONE, "请将手移到画面中央", False)

        pause = self._try_pause(hand, now)
        if pause:
            return pause

        if self.paused:
            return InteractionUpdate(
                InteractionMode.PAUSED,
                ActionType.NONE,
                f"已暂停 — 握拳 {self.cfg.pause_hold:.1f}s 恢复",
                False,
            )

        if hand.is_pinching:
            return self._handle_pinch(hand, now)

        released = self._release_pinch(now)
        if released:
            return released

        self._update_scroll_mode(hand)

        if self.mode == InteractionMode.SCROLL:
            return self._handle_scroll(hand, now)

        discrete = self._try_discrete(hand, now)
        if discrete:
            return discrete

        return self._handle_pointer(hand, now)
