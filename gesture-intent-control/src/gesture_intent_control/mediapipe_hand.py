from __future__ import annotations

from dataclasses import dataclass

import cv2
import mediapipe as mp
import numpy as np


@dataclass
class HandObservation:
    landmarks: np.ndarray
    handedness: str
    score: float

    @property
    def center(self) -> tuple[float, float]:
        xy = self.landmarks[:, :2]
        return float(xy[:, 0].mean()), float(xy[:, 1].mean())


class MediaPipeHandTracker:
    """Small wrapper around MediaPipe Hands.

    Landmarks are normalized image coordinates with shape (21, 3).
    """

    def __init__(
        self,
        max_num_hands: int = 1,
        min_detection_confidence: float = 0.55,
        min_tracking_confidence: float = 0.55,
    ) -> None:
        self._hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            model_complexity=1,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def process(self, frame_bgr: np.ndarray) -> HandObservation | None:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = self._hands.process(rgb)
        if not result.multi_hand_landmarks:
            return None

        hand_landmarks = result.multi_hand_landmarks[0]
        points = np.array(
            [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark],
            dtype=np.float32,
        )

        handedness = "Unknown"
        score = 0.0
        if result.multi_handedness:
            cls = result.multi_handedness[0].classification[0]
            handedness = cls.label
            score = float(cls.score)

        return HandObservation(landmarks=points, handedness=handedness, score=score)

    def draw(self, frame_bgr: np.ndarray, observation: HandObservation | None) -> None:
        if observation is None:
            return
        height, width = frame_bgr.shape[:2]
        for idx, (x, y, _z) in enumerate(observation.landmarks):
            px = int(x * width)
            py = int(y * height)
            color = (0, 230, 255) if idx in {4, 8, 12, 16, 20} else (60, 220, 60)
            cv2.circle(frame_bgr, (px, py), 4, color, -1)

        wrist = observation.landmarks[0]
        for idx in [4, 8, 12, 16, 20]:
            tip = observation.landmarks[idx]
            cv2.line(
                frame_bgr,
                (int(wrist[0] * width), int(wrist[1] * height)),
                (int(tip[0] * width), int(tip[1] * height)),
                (80, 160, 255),
                1,
            )

    def close(self) -> None:
        self._hands.close()

