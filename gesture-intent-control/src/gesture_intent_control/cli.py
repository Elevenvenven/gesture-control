from __future__ import annotations

import argparse

from .realtime import run_realtime
from .recorder import record_dataset
from .train import train_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gesture_intent_control",
        description="Train a personal hand gesture intent model for desktop control.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    record = subparsers.add_parser("record", help="Record labeled gesture clips.")
    record.add_argument("--label", required=True, help="Intent label, e.g. click or nav_back.")
    record.add_argument("--seconds", type=float, default=1.2, help="Clip length in seconds.")
    record.add_argument("--repeats", type=int, default=20, help="Number of clips to record.")
    record.add_argument("--camera", type=int, default=0, help="OpenCV camera index.")
    record.add_argument("--out-dir", default="data/raw", help="Dataset output directory.")

    train = subparsers.add_parser("train", help="Train intent classifier.")
    train.add_argument("--data-dir", default="data/raw", help="Dataset directory.")
    train.add_argument("--model-out", default="models/intent_model.joblib", help="Output model path.")
    train.add_argument("--window-frames", type=int, default=32, help="Frames sampled per clip.")
    train.add_argument("--threshold", type=float, default=0.58, help="Default confidence threshold.")

    run = subparsers.add_parser("run", help="Run realtime inference.")
    run.add_argument("--model", default="models/intent_model.joblib", help="Trained model path.")
    run.add_argument("--config", default="configs/intents.yaml", help="Intent action config.")
    run.add_argument("--camera", type=int, default=0, help="OpenCV camera index.")
    run.add_argument("--threshold", type=float, default=None, help="Override model threshold.")
    run.add_argument("--execute", action="store_true", help="Actually control the computer.")
    run.add_argument("--show-probs", action="store_true", help="Print top probabilities.")

    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "record":
        record_dataset(args)
    elif args.command == "train":
        train_model(args)
    elif args.command == "run":
        run_realtime(args)
    else:
        raise SystemExit(f"Unknown command: {args.command}")

