from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .features import FeatureConfig, sequence_to_vector


def save_clip(path: Path, label: str, frames: list[np.ndarray], meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "label": label,
        "meta": meta,
        "frames": [np.asarray(frame, dtype=np.float32).tolist() for frame in frames],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def load_clip(path: Path) -> tuple[str, list[np.ndarray]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    label = str(payload["label"])
    frames = [np.asarray(frame, dtype=np.float32) for frame in payload["frames"]]
    return label, frames


def load_dataset(data_dir: str | Path, config: FeatureConfig) -> tuple[np.ndarray, np.ndarray, list[Path]]:
    data_path = Path(data_dir)
    files = sorted(data_path.glob("*/*.json"))
    if not files:
        raise FileNotFoundError(
            f"No clips found in {data_path}. Record data first with: "
            "python -m gesture_intent_control record --label click"
        )

    vectors: list[np.ndarray] = []
    labels: list[str] = []
    used_files: list[Path] = []
    skipped: list[tuple[Path, str]] = []

    for path in files:
        try:
            label, frames = load_clip(path)
            vectors.append(sequence_to_vector(frames, config))
            labels.append(label)
            used_files.append(path)
        except Exception as exc:  # noqa: BLE001 - keep dataset loading forgiving.
            skipped.append((path, str(exc)))

    if not vectors:
        details = "\n".join(f"- {path}: {err}" for path, err in skipped[:10])
        raise RuntimeError(f"All clips failed to load.\n{details}")

    if skipped:
        print(f"Skipped {len(skipped)} broken clips.")

    return np.stack(vectors, axis=0), np.asarray(labels), used_files

