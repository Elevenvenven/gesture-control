from __future__ import annotations

from dataclasses import dataclass

import numpy as np


LANDMARKS = 21
COORDS = 3


@dataclass
class FeatureConfig:
    window_frames: int = 32


def normalize_landmarks(landmarks: np.ndarray) -> np.ndarray:
    """Normalize hand landmarks to wrist-relative, scale-stable coordinates."""

    arr = np.asarray(landmarks, dtype=np.float32).reshape(LANDMARKS, COORDS)
    wrist = arr[0].copy()
    centered = arr - wrist

    # Scale by the largest wrist-to-landmark distance so hand size matters less.
    scale = float(np.linalg.norm(centered[:, :2], axis=1).max())
    if scale < 1e-6:
        scale = 1.0
    return centered / scale


def resample_sequence(frames: list[np.ndarray], target_frames: int) -> np.ndarray:
    if not frames:
        raise ValueError("Cannot featurize an empty gesture clip.")

    normalized = np.stack([normalize_landmarks(frame) for frame in frames], axis=0)
    if len(normalized) == target_frames:
        return normalized

    old_idx = np.linspace(0.0, 1.0, num=len(normalized), dtype=np.float32)
    new_idx = np.linspace(0.0, 1.0, num=target_frames, dtype=np.float32)
    out = np.empty((target_frames, LANDMARKS, COORDS), dtype=np.float32)

    for landmark_idx in range(LANDMARKS):
        for coord_idx in range(COORDS):
            out[:, landmark_idx, coord_idx] = np.interp(
                new_idx,
                old_idx,
                normalized[:, landmark_idx, coord_idx],
            )
    return out


def sequence_to_vector(frames: list[np.ndarray], config: FeatureConfig) -> np.ndarray:
    """Turn a variable-length landmark sequence into a fixed vector.

    The vector intentionally mixes sampled trajectory and summary statistics.
    That keeps the first model small and useful with modest personal data.
    """

    seq = resample_sequence(frames, config.window_frames)
    flat = seq.reshape(config.window_frames, LANDMARKS * COORDS)

    velocity = np.diff(flat, axis=0, prepend=flat[:1])
    acceleration = np.diff(velocity, axis=0, prepend=velocity[:1])

    summary_parts = [
        flat.mean(axis=0),
        flat.std(axis=0),
        flat.min(axis=0),
        flat.max(axis=0),
        velocity.mean(axis=0),
        velocity.std(axis=0),
        acceleration.mean(axis=0),
        acceleration.std(axis=0),
    ]

    # A few semantic distances are very useful for pinches/fists.
    semantic = _semantic_features(seq)

    vector = np.concatenate(
        [
            flat.reshape(-1),
            velocity.reshape(-1),
            np.concatenate(summary_parts),
            semantic,
        ]
    )
    return vector.astype(np.float32)


def _semantic_features(seq: np.ndarray) -> np.ndarray:
    thumb_tip = seq[:, 4, :2]
    index_tip = seq[:, 8, :2]
    middle_tip = seq[:, 12, :2]
    ring_tip = seq[:, 16, :2]
    pinky_tip = seq[:, 20, :2]
    wrist = seq[:, 0, :2]

    distances = np.stack(
        [
            np.linalg.norm(thumb_tip - index_tip, axis=1),
            np.linalg.norm(index_tip - middle_tip, axis=1),
            np.linalg.norm(index_tip - wrist, axis=1),
            np.linalg.norm(middle_tip - wrist, axis=1),
            np.linalg.norm(ring_tip - wrist, axis=1),
            np.linalg.norm(pinky_tip - wrist, axis=1),
        ],
        axis=1,
    )
    center = seq[:, :, :2].mean(axis=1)
    center_velocity = np.diff(center, axis=0, prepend=center[:1])

    return np.concatenate(
        [
            distances.mean(axis=0),
            distances.std(axis=0),
            distances.min(axis=0),
            distances.max(axis=0),
            center[0],
            center[-1],
            center[-1] - center[0],
            center_velocity.mean(axis=0),
            center_velocity.std(axis=0),
        ]
    ).astype(np.float32)

