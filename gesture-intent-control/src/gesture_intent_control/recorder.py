from __future__ import annotations

import time
from pathlib import Path

import cv2

from .dataset import save_clip
from .mediapipe_hand import MediaPipeHandTracker


def record_dataset(args) -> None:
    out_dir = Path(args.out_dir) / args.label
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open camera index {args.camera}.")

    tracker = MediaPipeHandTracker()
    recorded = 0
    print(f"Recording label: {args.label}")
    print("Press SPACE to record one clip. Press q to quit.")

    try:
        while recorded < args.repeats:
            ok, frame = cap.read()
            if not ok:
                continue
            frame = cv2.flip(frame, 1)
            obs = tracker.process(frame)
            tracker.draw(frame, obs)

            _put_text(frame, f"label: {args.label}", 20, 32)
            _put_text(frame, f"clips: {recorded}/{args.repeats}", 20, 64)
            _put_text(frame, "SPACE record | q quit", 20, 96)
            _put_text(frame, "hand: yes" if obs else "hand: no", 20, 128)
            cv2.imshow("gesture recorder", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key != ord(" "):
                continue

            frames = _record_one_clip(cap, tracker, args.seconds, args.label, recorded + 1, args.repeats)
            if len(frames) < 3:
                print("Skipped: too few hand frames.")
                continue

            ts = int(time.time() * 1000)
            path = out_dir / f"{args.label}_{ts}.json"
            save_clip(
                path,
                args.label,
                frames,
                meta={
                    "seconds": args.seconds,
                    "created_at_ms": ts,
                    "camera": args.camera,
                },
            )
            recorded += 1
            print(f"Saved {path}")
    finally:
        tracker.close()
        cap.release()
        cv2.destroyAllWindows()

    print(f"Done. Recorded {recorded} clips for label '{args.label}'.")


def _record_one_clip(cap, tracker: MediaPipeHandTracker, seconds: float, label: str, index: int, total: int):
    frames = []
    start = time.monotonic()
    while time.monotonic() - start < seconds:
        ok, frame = cap.read()
        if not ok:
            continue
        frame = cv2.flip(frame, 1)
        obs = tracker.process(frame)
        if obs is not None:
            frames.append(obs.landmarks)
        tracker.draw(frame, obs)

        remaining = max(0.0, seconds - (time.monotonic() - start))
        _put_text(frame, f"REC {label} {index}/{total}", 20, 32, color=(60, 60, 255))
        _put_text(frame, f"{remaining:.1f}s", 20, 64, color=(60, 60, 255))
        cv2.imshow("gesture recorder", frame)
        cv2.waitKey(1)
    return frames


def _put_text(frame, text: str, x: int, y: int, color=(255, 255, 255)) -> None:
    cv2.putText(
        frame,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        color,
        2,
        cv2.LINE_AA,
    )

