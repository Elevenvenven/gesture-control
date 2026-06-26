"""Control the computer with hand gestures via webcam."""

from __future__ import annotations

import argparse
import threading
import time

import cv2

from controller import InputController
from gestures import GestureStabilizer, GestureType, HandTracker, SwipeState


class FrameGrabber:
    """Background camera reader — keeps only the latest frame to cut latency."""

    def __init__(self, cap: cv2.VideoCapture) -> None:
        self._cap = cap
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._ok = False
        self._frame = None
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.is_set():
            ok, frame = self._cap.read()
            with self._lock:
                self._ok = ok
                if ok:
                    self._frame = frame

    def read(self) -> tuple[bool, object | None]:
        with self._lock:
            if self._frame is None:
                return self._ok, None
            return self._ok, self._frame.copy()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)


def list_cameras(max_index: int = 5) -> None:
    print("Scanning cameras...")
    for index in range(max_index):
        cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
        if not cap.isOpened():
            print(f"  [{index}] not available")
            continue
        ok, _ = cap.read()
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        status = "ok" if ok else "opened but no frame"
        print(f"  [{index}] {w}x{h} ({status})")
        cap.release()


def open_camera(camera_index: int, warmup: float, width: int, height: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(camera_index, cv2.CAP_AVFOUNDATION)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if not cap.isOpened():
        raise RuntimeError(
            f"Cannot open camera {camera_index}. "
            "Try --list-cameras or a different --camera index."
        )

    deadline = time.monotonic() + warmup
    while time.monotonic() < deadline:
        ok, _ = cap.read()
        if ok:
            return cap
        time.sleep(0.05)

    cap.release()
    raise RuntimeError(
        f"Camera {camera_index} opened but returned no frames. Try --list-cameras."
    )


def draw_overlay(
    frame,
    lines: list[str],
    *,
    active: bool,
    fps: float,
    gesture: GestureType | None,
    confidence: float,
    pinch: float | None,
    landmarks,
) -> None:
    color = (40, 220, 80) if active else (80, 180, 255)
    panel_h = 36 + 30 * len(lines)
    cv2.rectangle(frame, (10, 10), (780, panel_h), (0, 0, 0), -1)
    for i, text in enumerate(lines):
        cv2.putText(
            frame,
            text,
            (24, 48 + i * 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.68,
            color if i == 0 else (210, 210, 210),
            2 if i == 0 else 1,
            cv2.LINE_AA,
        )

    cv2.putText(
        frame,
        f"FPS {fps:.0f}",
        (frame.shape[1] - 120, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (180, 180, 180),
        1,
        cv2.LINE_AA,
    )

    if gesture and gesture != GestureType.NONE:
        label = gesture.value.replace("_", " ").upper()
        cv2.putText(
            frame,
            f"{label} {int(confidence * 100)}%",
            (24, frame.shape[0] - 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            color,
            2,
            cv2.LINE_AA,
        )

    if landmarks is not None and pinch is not None:
        tip = landmarks[8]
        thumb = landmarks[4]
        h, w = frame.shape[:2]
        tx, ty = int(thumb.x * w), int(thumb.y * h)
        ix, iy = int(tip.x * w), int(tip.y * h)
        cv2.circle(frame, (tx, ty), 8, (0, 180, 255), -1)
        cv2.circle(frame, (ix, iy), 8, (0, 255, 120), -1)
        cv2.line(frame, (tx, ty), (ix, iy), (0, 200, 255), 2)


class GestureApp:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        tracker_kwargs: dict = {
            "detection_confidence": args.detection_confidence,
            "tracking_confidence": args.tracking_confidence,
            "model_quality": args.model_quality,
            "use_gpu": not args.cpu_only,
        }
        if args.process_width is not None:
            tracker_kwargs["process_width"] = args.process_width
        if args.landmark_alpha is not None:
            tracker_kwargs["landmark_alpha"] = args.landmark_alpha
        tracker_kwargs["vote_window"] = args.vote_window
        self.tracker = HandTracker(**tracker_kwargs)
        self.input = InputController(args.smooth_alpha)
        self.swipe = SwipeState(maxlen=args.frames)
        self.stabilizer = GestureStabilizer(frames=args.stable_frames)

        self.last_scroll_at = 0.0
        self.last_nav_at = 0.0
        self.last_click_at = 0.0
        self.last_right_click_at = 0.0
        self.last_double_click_at = 0.0
        self.last_pause_toggle_at = 0.0
        self._fist_hold_start: float | None = None
        self.pinch_start_at: float | None = None
        self.dragging = False
        self.paused = False
        self.last_action = "Ready"
        self.gesture_active = False
        self.current_gesture: GestureType | None = None
        self.current_confidence = 0.0
        self.current_landmarks = None
        self.current_pinch: float | None = None

    def close(self) -> None:
        if self.dragging:
            self.input.mouse_up()
        self.tracker.close()

    def _reset_interaction(self) -> None:
        self.swipe.clear()
        self.pinch_start_at = None
        if self.dragging:
            self.input.mouse_up()
            self.dragging = False

    def _cursor_point(self, landmarks) -> tuple[float, float]:
        tip = landmarks[8]
        pip = landmarks[6]
        blend = 0.78
        return tip.x * blend + pip.x * (1 - blend), tip.y * blend + pip.y * (1 - blend)

    def _move_from_index(self, hand) -> None:
        x, y = self._cursor_point(hand.landmarks)
        self.input.move_cursor(x, y, self.args.cursor_margin)

    def _handle_pinch(self, hand, now: float) -> None:
        self._move_from_index(hand)
        self.swipe.clear()
        self.gesture_active = True

        if self.pinch_start_at is None:
            self.pinch_start_at = now
            self.last_action = "Pinch"
        elif not self.dragging and now - self.pinch_start_at >= self.args.drag_hold:
            self.input.mouse_down()
            self.dragging = True
            self.last_action = "Drag"
        elif self.dragging:
            self.last_action = "Dragging"

    def _release_pinch(self, now: float) -> None:
        if self.pinch_start_at is None:
            return
        if self.dragging:
            self.input.mouse_up()
            self.dragging = False
            self.last_action = "Drop"
        elif now - self.last_click_at >= self.args.click_cooldown:
            self.input.click()
            self.last_click_at = now
            self.last_action = "Click"
            self.gesture_active = True
        self.pinch_start_at = None

    def _handle_open_palm(self, hand, now: float) -> None:
        self.gesture_active = True
        self.swipe.append(hand.center_x, hand.center_y)
        if not self.swipe.ready:
            return

        dx, dy = self.swipe.delta()
        can_scroll = now - self.last_scroll_at >= self.args.cooldown
        can_nav = now - self.last_nav_at >= self.args.nav_cooldown

        if abs(dx) >= self.args.nav_threshold and abs(dx) > abs(dy) * 1.2 and can_nav:
            if dx > 0:
                self.input.navigate_forward()
                self.last_action = "Forward"
            else:
                self.input.navigate_back()
                self.last_action = "Back"
            self.last_nav_at = now
            self.swipe.clear()
        elif abs(dy) >= self.args.threshold and can_scroll:
            scroll_amount = -self.args.scroll if dy > 0 else self.args.scroll
            self.input.scroll(scroll_amount)
            direction = "down" if dy > 0 else "up"
            self.last_action = f"Scroll {direction}"
            self.last_scroll_at = now
            self.swipe.clear()

    def _handle_pause(self, hand, now: float) -> bool:
        """Return True if this frame is consumed by pause logic."""
        if not hand.is_closed_fist:
            self._fist_hold_start = None
            return False

        if self._fist_hold_start is None:
            self._fist_hold_start = now

        held = now - self._fist_hold_start
        if held < self.args.pause_hold:
            pct = int(held / self.args.pause_hold * 100)
            label = "resume" if self.paused else "pause"
            self.last_action = f"Hold fist to {label}... {pct}%"
            self.gesture_active = True
            return True

        if now - self.last_pause_toggle_at < self.args.pause_cooldown:
            return True

        self.paused = not self.paused
        self.last_pause_toggle_at = now
        self._fist_hold_start = None
        self.last_action = "Paused" if self.paused else "Resumed"
        self._reset_interaction()
        self.gesture_active = True
        return True

    def process_hand(self, hand, now: float) -> None:
        gesture = hand.stable_gesture
        conf = hand.confidence
        min_conf = self.args.min_confidence

        if self._handle_pause(hand, now):
            return

        if self.paused:
            self.last_action = f"Paused - hold fist {self.args.pause_hold:.1f}s to resume"
            self._reset_interaction()
            return

        if hand.is_pinching:
            self._handle_pinch(hand, now)
            return

        self._release_pinch(now)

        if gesture == GestureType.INDEX_ONLY and conf >= min_conf * 0.85:
            self.swipe.clear()
            self._move_from_index(hand)
            self.last_action = "Move mouse"
            self.gesture_active = True
            return

        stable = self.stabilizer.update(gesture)

        if (
            stable == GestureType.TWO_FINGERS
            and conf >= min_conf
            and now - self.last_right_click_at >= self.args.right_click_cooldown
        ):
            self.input.right_click()
            self.last_right_click_at = now
            self.last_action = "Right click"
            self.swipe.clear()
            self.gesture_active = True
            return

        if (
            stable == GestureType.THREE_FINGERS
            and conf >= min_conf
            and now - self.last_double_click_at >= self.args.double_click_cooldown
        ):
            self.input.double_click()
            self.last_double_click_at = now
            self.last_action = "Double click"
            self.swipe.clear()
            self.gesture_active = True
            return

        if (gesture == GestureType.OPEN_PALM or hand.palm_score >= 4) and conf >= min_conf * 0.8:
            self._handle_open_palm(hand, now)
            return

        self.swipe.clear()
        if hand.is_closed_fist:
            self.last_action = "Hold fist to pause"
        else:
            self.last_action = "Open palm, index finger, pinch, 2 or 3 fingers"

    def process_no_hand(self) -> None:
        self._fist_hold_start = None
        self._reset_interaction()
        self.stabilizer.update(GestureType.NONE)
        self.input.reset_cursor()
        self.last_action = "Show one hand"

    def run(self) -> None:
        cap = open_camera(
            self.args.camera, self.args.warmup, self.args.camera_width, self.args.camera_height
        )
        grabber = FrameGrabber(cap)
        print(f"Camera {self.args.camera} ready. Press Q or Esc to quit.")

        fps = 0.0
        frame_times: list[float] = []
        start_ms = int(time.monotonic() * 1000)
        help_lines = [
            f"{self.last_action} | Q/Esc quit",
            "Open palm: scroll / swipe for back-forward",
            "Index: move | pinch: click | hold pinch: drag",
            "Two fingers: right click | three: double click | hold fist 0.6s: pause",
        ]

        try:
            while True:
                t0 = time.monotonic()
                ok, frame = grabber.read()
                if not ok or frame is None:
                    time.sleep(0.005)
                    continue

                frame = cv2.flip(frame, 1)
                now = time.monotonic()
                timestamp_ms = int(now * 1000) - start_ms
                self.gesture_active = False
                hand = self.tracker.process(frame, self.args.pinch_threshold, timestamp_ms)

                if hand:
                    self.current_gesture = hand.stable_gesture
                    self.current_confidence = hand.confidence
                    self.current_landmarks = hand.landmarks
                    self.current_pinch = hand.pinch
                    HandTracker.draw_landmarks(frame, hand.landmarks, hand.finger_scores)
                    self.process_hand(hand, now)
                else:
                    self.current_gesture = None
                    self.current_confidence = 0.0
                    self.current_landmarks = None
                    self.current_pinch = None
                    self.process_no_hand()

                help_lines[0] = f"{self.last_action} | Q/Esc quit"
                draw_overlay(
                    frame,
                    help_lines,
                    active=self.gesture_active,
                    fps=fps,
                    gesture=self.current_gesture,
                    confidence=self.current_confidence,
                    pinch=self.current_pinch,
                    landmarks=self.current_landmarks,
                )

                cv2.imshow("Gesture Control", frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    break

                frame_times.append(time.monotonic() - t0)
                if len(frame_times) > 30:
                    frame_times.pop(0)
                if frame_times:
                    fps = 1.0 / (sum(frame_times) / len(frame_times))
        finally:
            grabber.stop()
            cap.release()
            cv2.destroyAllWindows()
            self.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Control the computer with hand gestures.")
    parser.add_argument("--list-cameras", action="store_true", help="List available camera indices.")
    parser.add_argument(
        "--camera",
        type=int,
        default=1,
        help="Camera index. On this Mac, 1 is the built-in FaceTime HD Camera.",
    )
    parser.add_argument(
        "--camera-width",
        type=int,
        default=640,
        help="Camera capture width (lower = faster).",
    )
    parser.add_argument(
        "--camera-height",
        type=int,
        default=480,
        help="Camera capture height (lower = faster).",
    )
    parser.add_argument(
        "--process-width",
        type=int,
        default=None,
        help="Detection frame width (default: 720 accurate / 480 fast).",
    )
    parser.add_argument(
        "--model-quality",
        choices=("fast", "accurate"),
        default="accurate",
        help="fast=480px inference, accurate=720px inference (recommended).",
    )
    parser.add_argument(
        "--landmark-alpha",
        type=float,
        default=None,
        help="Landmark smoothing (higher = more responsive).",
    )
    parser.add_argument("--cpu-only", action="store_true", help="Disable GPU delegate.")
    parser.add_argument("--scroll", type=int, default=5, help="Scroll amount per gesture.")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.085,
        help="Required vertical hand movement in normalized coordinates.",
    )
    parser.add_argument(
        "--nav-threshold",
        type=float,
        default=0.13,
        help="Required horizontal open-palm movement for back/forward.",
    )
    parser.add_argument("--frames", type=int, default=7, help="Frames used to measure a swipe.")
    parser.add_argument("--cooldown", type=float, default=0.22, help="Seconds between scroll actions.")
    parser.add_argument("--nav-cooldown", type=float, default=0.8, help="Seconds between nav actions.")
    parser.add_argument("--click-cooldown", type=float, default=0.45, help="Seconds between pinch clicks.")
    parser.add_argument(
        "--right-click-cooldown",
        type=float,
        default=0.8,
        help="Seconds between two-finger right clicks.",
    )
    parser.add_argument(
        "--double-click-cooldown",
        type=float,
        default=0.8,
        help="Seconds between three-finger double clicks.",
    )
    parser.add_argument(
        "--pause-hold",
        type=float,
        default=0.6,
        help="Seconds to hold a closed fist before pause/resume triggers.",
    )
    parser.add_argument(
        "--pause-cooldown",
        type=float,
        default=1.2,
        help="Seconds between fist pause/resume toggles.",
    )
    parser.add_argument(
        "--drag-hold",
        type=float,
        default=0.38,
        help="Seconds to hold a pinch before it becomes a drag.",
    )
    parser.add_argument(
        "--pinch-threshold",
        type=float,
        default=0.35,
        help="Thumb-index distance threshold for pinch.",
    )
    parser.add_argument(
        "--cursor-margin",
        type=float,
        default=0.12,
        help="Camera edge margin ignored when mapping finger position.",
    )
    parser.add_argument(
        "--smooth-alpha",
        type=float,
        default=0.55,
        help="Cursor smoothing factor (higher = more responsive).",
    )
    parser.add_argument(
        "--stable-frames",
        type=int,
        default=3,
        help="Frames a click gesture must hold before triggering.",
    )
    parser.add_argument(
        "--warmup",
        type=float,
        default=3.0,
        help="Seconds to wait for the camera to produce its first frame.",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.52,
        help="Minimum gesture confidence (0-1) before triggering actions.",
    )
    parser.add_argument(
        "--vote-window",
        type=int,
        default=7,
        help="Frames used for temporal gesture voting.",
    )
    parser.add_argument("--detection-confidence", type=float, default=0.78)
    parser.add_argument("--tracking-confidence", type=float, default=0.78)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.list_cameras:
        list_cameras()
        return
    GestureApp(args).run()


if __name__ == "__main__":
    main()
