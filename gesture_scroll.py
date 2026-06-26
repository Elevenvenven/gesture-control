"""Control the computer with hand gestures via webcam."""

from __future__ import annotations

import argparse
import threading
import time

import cv2

from controller import InputController
from gestures import GestureType, HandTracker
from interaction import (
    MODE_COLORS,
    MODE_HINTS,
    InteractionConfig,
    InteractionEngine,
    InteractionMode,
)


class FrameGrabber:
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
        print(f"  [{index}] {w}x{h} ({'ok' if ok else 'no frame'})")
        cap.release()


def open_camera(camera_index: int, warmup: float, width: int, height: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(camera_index, cv2.CAP_AVFOUNDATION)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {camera_index}. Try --list-cameras.")

    deadline = time.monotonic() + warmup
    while time.monotonic() < deadline:
        ok, _ = cap.read()
        if ok:
            return cap
        time.sleep(0.05)

    cap.release()
    raise RuntimeError(f"Camera {camera_index} opened but returned no frames.")


def draw_overlay(
    frame,
    *,
    mode: InteractionMode,
    status: str,
    hint: str,
    active: bool,
    fps: float,
    gesture: GestureType | None,
    confidence: float,
    quality: float,
    landmarks,
    pinch: float | None,
) -> None:
    mode_color = MODE_COLORS.get(mode, (80, 180, 255))
    accent = mode_color if active else (80, 180, 255)

    lines = [
        f"[{mode.value.upper()}] {status} | Q/Esc 退出",
        hint,
        "食+中=右键 | 食+中+无=双击 | 张掌滑动=滚动",
    ]
    panel_h = 36 + 30 * len(lines)
    cv2.rectangle(frame, (10, 10), (820, panel_h), (0, 0, 0), -1)
    cv2.rectangle(frame, (10, 10), (820, panel_h), mode_color, 2)

    for i, text in enumerate(lines):
        cv2.putText(
            frame,
            text,
            (24, 48 + i * 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.66,
            accent if i == 0 else (210, 210, 210),
            2 if i == 0 else 1,
            cv2.LINE_AA,
        )

    cv2.putText(frame, f"FPS {fps:.0f}", (frame.shape[1] - 120, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 180, 180), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Q {int(quality * 100)}%", (frame.shape[1] - 120, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160, 160, 160), 1, cv2.LINE_AA)

    if gesture and gesture != GestureType.NONE:
        cv2.putText(
            frame,
            f"{gesture.value.replace('_', ' ').upper()} {int(confidence * 100)}%",
            (24, frame.shape[0] - 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            accent,
            2,
            cv2.LINE_AA,
        )

    if landmarks is not None and pinch is not None:
        tip, thumb = landmarks[8], landmarks[4]
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
            "vote_window": args.vote_window,
        }
        if args.process_width is not None:
            tracker_kwargs["process_width"] = args.process_width
        if args.landmark_alpha is not None:
            tracker_kwargs["landmark_alpha"] = args.landmark_alpha

        self.tracker = HandTracker(**tracker_kwargs)
        self.input = InputController(args.smooth_alpha)
        cfg = InteractionConfig(
            pause_hold=args.pause_hold,
            pause_cooldown=args.pause_cooldown,
            min_confidence=args.min_confidence,
            scroll_threshold=args.threshold,
            nav_threshold=args.nav_threshold,
            scroll_amount=args.scroll,
            scroll_cooldown=args.cooldown,
            nav_cooldown=args.nav_cooldown,
            click_cooldown=args.click_cooldown,
            right_click_cooldown=args.right_click_cooldown,
            double_click_cooldown=args.double_click_cooldown,
            drag_hold=args.drag_hold,
            swipe_frames=args.frames,
            cursor_margin=args.cursor_margin,
            stable_frames=args.stable_frames,
        )
        self.engine = InteractionEngine(cfg, self.input)

        self.current_gesture: GestureType | None = None
        self.current_confidence = 0.0
        self.current_quality = 0.0
        self.current_landmarks = None
        self.current_pinch: float | None = None
        self.current_mode = InteractionMode.READY
        self.current_status = "Ready"
        self.current_active = False

    def close(self) -> None:
        self.engine.close()
        self.tracker.close()

    def run(self) -> None:
        cap = open_camera(self.args.camera, self.args.warmup, self.args.camera_width, self.args.camera_height)
        grabber = FrameGrabber(cap)
        print(f"Camera {self.args.camera} ready. Press Q or Esc to quit.")

        fps = 0.0
        frame_times: list[float] = []
        start_ms = int(time.monotonic() * 1000)

        try:
            while True:
                t0 = time.monotonic()
                ok, frame = grabber.read()
                if not ok or frame is None:
                    time.sleep(0.005)
                    continue

                frame = cv2.flip(frame, 1)
                now = time.monotonic()
                hand = self.tracker.process(frame, self.args.pinch_threshold, int(now * 1000) - start_ms)

                if hand:
                    self.current_gesture = hand.stable_gesture
                    self.current_confidence = hand.confidence
                    self.current_quality = hand.quality
                    self.current_landmarks = hand.landmarks
                    self.current_pinch = hand.pinch
                    HandTracker.draw_landmarks(frame, hand.landmarks, hand.finger_scores)
                    result = self.engine.update(hand, now)
                else:
                    self.current_gesture = None
                    self.current_confidence = 0.0
                    self.current_quality = 0.0
                    self.current_landmarks = None
                    self.current_pinch = None
                    result = self.engine.update(None, now)

                self.current_mode = result.mode
                self.current_status = result.status
                self.current_active = result.active

                draw_overlay(
                    frame,
                    mode=result.mode,
                    status=result.status,
                    hint=MODE_HINTS.get(result.mode, ""),
                    active=result.active,
                    fps=fps,
                    gesture=self.current_gesture,
                    confidence=self.current_confidence,
                    quality=self.current_quality,
                    landmarks=self.current_landmarks,
                    pinch=self.current_pinch,
                )

                cv2.imshow("Gesture Control", frame)
                if (cv2.waitKey(1) & 0xFF) in (ord("q"), ord("Q"), 27):
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
    parser.add_argument("--list-cameras", action="store_true")
    parser.add_argument("--camera", type=int, default=1)
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--process-width", type=int, default=None)
    parser.add_argument("--model-quality", choices=("fast", "accurate"), default="accurate")
    parser.add_argument("--landmark-alpha", type=float, default=None)
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--scroll", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.085)
    parser.add_argument("--nav-threshold", type=float, default=0.13)
    parser.add_argument("--frames", type=int, default=7)
    parser.add_argument("--cooldown", type=float, default=0.22)
    parser.add_argument("--nav-cooldown", type=float, default=0.8)
    parser.add_argument("--click-cooldown", type=float, default=0.45)
    parser.add_argument("--right-click-cooldown", type=float, default=0.8)
    parser.add_argument("--double-click-cooldown", type=float, default=0.8)
    parser.add_argument("--pause-hold", type=float, default=0.6)
    parser.add_argument("--pause-cooldown", type=float, default=1.2)
    parser.add_argument("--drag-hold", type=float, default=0.38)
    parser.add_argument("--pinch-threshold", type=float, default=0.35)
    parser.add_argument("--cursor-margin", type=float, default=0.12)
    parser.add_argument("--smooth-alpha", type=float, default=0.55)
    parser.add_argument("--stable-frames", type=int, default=3)
    parser.add_argument("--warmup", type=float, default=3.0)
    parser.add_argument("--min-confidence", type=float, default=0.48)
    parser.add_argument("--vote-window", type=int, default=9)
    parser.add_argument("--detection-confidence", type=float, default=0.80)
    parser.add_argument("--tracking-confidence", type=float, default=0.80)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.list_cameras:
        list_cameras()
        return
    GestureApp(args).run()


if __name__ == "__main__":
    main()
