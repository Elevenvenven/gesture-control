from __future__ import annotations

from collections import deque

import cv2

from .actions import ActionMapper
from .mediapipe_hand import MediaPipeHandTracker
from .model import IntentModel


def run_realtime(args) -> None:
    model = IntentModel.load(args.model)
    threshold = model.threshold if args.threshold is None else args.threshold
    mapper = ActionMapper(args.config, execute=args.execute)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open camera index {args.camera}.")

    tracker = MediaPipeHandTracker()
    buffer = deque(maxlen=model.feature_config.window_frames)
    last_prediction = "waiting..."

    print("Realtime gesture intent control")
    print(f"Mode: {'EXECUTE' if args.execute else 'DRY RUN'}")
    print("Press q to quit.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue

            frame = cv2.flip(frame, 1)
            obs = tracker.process(frame)
            tracker.draw(frame, obs)

            if obs is not None:
                buffer.append(obs.landmarks)
            else:
                buffer.clear()

            if len(buffer) == buffer.maxlen:
                prediction = model.predict(list(buffer), threshold=threshold)
                last_prediction = f"{prediction.label} ({prediction.confidence:.2f})"
                if args.show_probs:
                    top = sorted(
                        prediction.probabilities.items(),
                        key=lambda item: item[1],
                        reverse=True,
                    )[:3]
                    print(" | ".join(f"{k}:{v:.2f}" for k, v in top))

                if prediction.label != "unknown":
                    mapper.maybe_execute(prediction.label, prediction.confidence)
                    # Keep overlapping windows for smoothness but avoid executing every frame.
                    for _ in range(max(1, buffer.maxlen // 3)):
                        if buffer:
                            buffer.popleft()

            _draw_overlay(frame, last_prediction, len(buffer), buffer.maxlen, args.execute, mapper.paused)
            cv2.imshow("gesture intent control", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break
    finally:
        tracker.close()
        cap.release()
        cv2.destroyAllWindows()


def _draw_overlay(frame, prediction: str, size: int, max_size: int, execute: bool, paused: bool) -> None:
    mode = "EXECUTE" if execute else "DRY RUN"
    if paused:
        mode += " / PAUSED"
    lines = [
        f"mode: {mode}",
        f"prediction: {prediction}",
        f"buffer: {size}/{max_size}",
        "q/esc quit",
    ]
    y = 32
    for line in lines:
        cv2.putText(
            frame,
            line,
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        y += 32

