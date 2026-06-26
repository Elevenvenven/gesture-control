"""Hand tracking and gesture classification."""

from __future__ import annotations

import math
import ssl
import urllib.request
from collections import Counter, deque
from dataclasses import dataclass, field
from enum import Enum
from math import hypot
from pathlib import Path
from types import SimpleNamespace

import certifi
import cv2
import mediapipe as mp
from mediapipe.python.solutions import hands as mp_hands
from mediapipe.tasks.python import vision

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
MODEL_DIR = Path(__file__).resolve().parent / "models"
MODEL_PATH = MODEL_DIR / "hand_landmarker.task"

QUALITY_PRESETS = {
    "fast": {"process_width": 480, "mincutoff": 1.4, "beta": 0.01},
    "accurate": {"process_width": 720, "mincutoff": 1.0, "beta": 0.006},
}

FINGER_DEFS = {
    "index": (8, 7, 6, 5),
    "middle": (12, 11, 10, 9),
    "ring": (16, 15, 14, 13),
    "pinky": (20, 19, 18, 17),
}


class GestureType(Enum):
    NONE = "none"
    OPEN_PALM = "open_palm"
    INDEX_ONLY = "index_only"
    TWO_FINGERS = "two_fingers"
    THREE_FINGERS = "three_fingers"
    FIST = "fist"
    PINCH = "pinch"
    OTHER = "other"


@dataclass
class FingerScore:
    extended: float
    curled: float


@dataclass
class HandFrame:
    landmarks: list
    gesture: GestureType
    stable_gesture: GestureType
    confidence: float
    palm_score: int
    up_count: int
    center_x: float
    center_y: float
    pinch: float
    index_only: bool
    is_pinching: bool
    is_closed_fist: bool
    finger_scores: dict[str, FingerScore]
    handedness: str


@dataclass
class SwipeState:
    maxlen: int = 7
    history: deque = field(init=False)

    def __post_init__(self) -> None:
        self.history = deque(maxlen=self.maxlen)

    def clear(self) -> None:
        self.history.clear()

    def append(self, x: float, y: float) -> None:
        self.history.append((x, y))

    @property
    def ready(self) -> bool:
        return len(self.history) == self.history.maxlen

    def delta(self) -> tuple[float, float]:
        if not self.ready:
            return 0.0, 0.0
        dx = self.history[-1][0] - self.history[0][0]
        dy = self.history[-1][1] - self.history[0][1]
        return dx, dy


def ensure_model() -> Path:
    if MODEL_PATH.exists():
        return MODEL_PATH
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading hand landmarker model to {MODEL_PATH} ...")
    context = ssl.create_default_context(cafile=certifi.where())
    with urllib.request.urlopen(MODEL_URL, context=context) as response, MODEL_PATH.open("wb") as out:
        out.write(response.read())
    print("Model download complete.")
    return MODEL_PATH


def _distance(a, b) -> float:
    return hypot(a.x - b.x, a.y - b.y, a.z - b.z)


def _dist2d(a, b) -> float:
    return hypot(a.x - b.x, a.y - b.y)


def hand_scale(landmarks) -> float:
    return max(_dist2d(landmarks[0], landmarks[9]), 0.001)


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def _smoothstep(value: float, edge0: float, edge1: float) -> float:
    if edge0 == edge1:
        return 1.0 if value >= edge1 else 0.0
    t = _clamp((value - edge0) / (edge1 - edge0))
    return t * t * (3.0 - 2.0 * t)


def _angle_at_joint(a, b, c) -> float:
    """Angle ABC in degrees (at vertex b)."""
    v1 = (a.x - b.x, a.y - b.y, a.z - b.z)
    v2 = (c.x - b.x, c.y - b.y, c.z - b.z)
    m1 = math.sqrt(sum(x * x for x in v1))
    m2 = math.sqrt(sum(x * x for x in v2))
    if m1 * m2 < 1e-8:
        return 0.0
    cos_v = _clamp(sum(x * y for x, y in zip(v1, v2)) / (m1 * m2), -1.0, 1.0)
    return math.degrees(math.acos(cos_v))


def _palm_normal_z(landmarks) -> float:
    """Signed palm normal component — encodes hand facing direction."""
    wrist, idx_mcp, pinky_mcp = landmarks[0], landmarks[5], landmarks[17]
    ax, ay = idx_mcp.x - wrist.x, idx_mcp.y - wrist.y
    bx, by = pinky_mcp.x - wrist.x, pinky_mcp.y - wrist.y
    return ax * by - ay * bx


def score_finger(landmarks, tip_id: int, pip_id: int, mcp_id: int, dip_id: int) -> FingerScore:
    tip, pip, mcp, dip = landmarks[tip_id], landmarks[pip_id], landmarks[mcp_id], landmarks[dip_id]
    wrist = landmarks[0]
    palm = landmarks[9]
    scale = hand_scale(landmarks)

    angle = _angle_at_joint(mcp, pip, tip)
    ext_angle = _smoothstep(angle, 155.0, 172.0)
    curl_angle = _smoothstep(145.0 - angle, 5.0, 25.0)

    tip_w = _dist2d(tip, wrist)
    pip_w = _dist2d(pip, wrist)
    mcp_w = _dist2d(mcp, wrist)
    ext_dist = _smoothstep(tip_w / max(pip_w, 1e-6), 1.06, 1.18) * _smoothstep(
        tip_w / max(mcp_w, 1e-6), 1.0, 1.12
    )

    ext_depth = _smoothstep(pip.z - tip.z, 0.008, 0.03)
    ext_y = _smoothstep(pip.y - tip.y, 0.004, 0.02)

    palm_side = _palm_normal_z(landmarks)
    if abs(palm_side) > 1e-4:
        side = 1.0 if palm_side > 0 else -1.0
        ext_palm = _smoothstep(side * (tip.x - pip.x), 0.0, 0.025)
    else:
        ext_palm = ext_y

    extended = _clamp(0.32 * ext_angle + 0.28 * ext_dist + 0.18 * ext_depth + 0.12 * ext_y + 0.10 * ext_palm)

    tip_to_palm = _dist2d(tip, palm) / scale
    curl_dist = _smoothstep(0.44 - tip_to_palm, 0.0, 0.12)
    curl_wrist = _smoothstep(pip_w - tip_w, 0.0, 0.02)
    curled = _clamp(max(curl_angle, curl_dist, curl_wrist, 1.0 - extended * 1.15))

    return FingerScore(extended=extended, curled=curled)


def analyze_fingers(landmarks) -> dict[str, FingerScore]:
    return {
        name: score_finger(landmarks, tip, dip, pip, mcp)
        for name, (tip, dip, pip, mcp) in FINGER_DEFS.items()
    }


def fist_confidence(landmarks, scores: dict[str, FingerScore]) -> float:
    scale = hand_scale(landmarks)
    palm = landmarks[9]

    if any(s.extended > 0.42 for s in scores.values()):
        return 0.0

    tip_scores = []
    for tip_id in (8, 12, 16, 20):
        proximity = 1.0 - _clamp(_dist2d(landmarks[tip_id], palm) / scale / 0.38)
        tip_scores.append(proximity)
    palm_score = sum(tip_scores) / len(tip_scores)

    thumb_tucked = 1.0 - _clamp(_dist2d(landmarks[4], landmarks[5]) / scale / 0.52)
    curl_avg = sum(s.curled for s in scores.values()) / len(scores)

    return _clamp(0.45 * palm_score + 0.25 * thumb_tucked + 0.30 * curl_avg)


def is_closed_fist(landmarks, scores: dict[str, FingerScore] | None = None) -> bool:
    if scores is None:
        scores = analyze_fingers(landmarks)
    return fist_confidence(landmarks, scores) >= 0.72


def pinch_distance(landmarks) -> float:
    scale = hand_scale(landmarks)
    return _distance(landmarks[4], landmarks[8]) / scale


def thumb_extended_for_pinch(landmarks) -> float:
    return score_finger(landmarks, 4, 3, 2, 1).extended


def normalized_hand_center(landmarks) -> tuple[float, float]:
    wrist = landmarks[0]
    middle_mcp = landmarks[9]
    return (wrist.x + middle_mcp.x) / 2.0, (wrist.y + middle_mcp.y) / 2.0


def open_palm_score(scores: dict[str, FingerScore]) -> int:
    return sum(1 for s in scores.values() if s.extended >= 0.58)


def classify_gesture(
    landmarks,
    scores: dict[str, FingerScore],
    is_pinching: bool,
) -> tuple[GestureType, float]:
    if is_pinching:
        return GestureType.PINCH, 0.92

    fist_conf = fist_confidence(landmarks, scores)
    if fist_conf >= 0.72:
        return GestureType.FIST, fist_conf

    idx = scores["index"].extended
    mid = scores["middle"].extended
    ring = scores["ring"].extended
    pinky = scores["pinky"].extended
    ring_c = scores["ring"].curled
    pinky_c = scores["pinky"].curled
    mid_c = scores["middle"].curled

    if idx >= 0.58 and mid >= 0.58 and ring_c >= 0.50 and pinky_c >= 0.50 and ring < 0.50:
        conf = min(idx, mid, ring_c, pinky_c)
        return GestureType.TWO_FINGERS, conf

    if idx >= 0.55 and mid >= 0.55 and ring >= 0.50 and pinky_c >= 0.50 and pinky < 0.48:
        conf = min(idx, mid, ring, pinky_c)
        return GestureType.THREE_FINGERS, conf

    if idx >= 0.58 and mid_c >= 0.52 and ring_c >= 0.52 and pinky_c >= 0.52 and mid < 0.45:
        conf = min(idx, mid_c, ring_c, pinky_c)
        return GestureType.INDEX_ONLY, conf

    palm_count = open_palm_score(scores)
    if palm_count >= 4:
        conf = min(s.extended for s in scores.values())
        return GestureType.OPEN_PALM, conf

    return GestureType.OTHER, 0.2


class OneEuroFilter:
    """Adaptive low-pass filter — smooth when slow, responsive when fast."""

    def __init__(self, freq: float = 30.0, mincutoff: float = 1.0, beta: float = 0.007) -> None:
        self.freq = freq
        self.mincutoff = mincutoff
        self.beta = beta
        self._x: float | None = None
        self._dx = 0.0

    def reset(self) -> None:
        self._x = None
        self._dx = 0.0

    def _alpha(self, cutoff: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau * self.freq)

    def filter(self, x: float) -> float:
        if self._x is None:
            self._x = x
            return x
        dx = (x - self._x) * self.freq
        dx_hat = self._dx + self._alpha(1.0) * (dx - self._dx)
        cutoff = self.mincutoff + self.beta * abs(dx_hat)
        x_hat = self._x + self._alpha(cutoff) * (x - self._x)
        self._x = x_hat
        self._dx = dx_hat
        return x_hat


class LandmarkSmoother:
    """Per-landmark One Euro filtering."""

    def __init__(self, mincutoff: float = 1.0, beta: float = 0.006, freq: float = 30.0) -> None:
        n = 21 * 3
        self._filters = [OneEuroFilter(freq, mincutoff, beta) for _ in range(n)]

    def reset(self) -> None:
        for f in self._filters:
            f.reset()

    def apply(self, landmarks) -> list:
        coords = []
        for i, lm in enumerate(landmarks):
            base = i * 3
            x = self._filters[base].filter(lm.x)
            y = self._filters[base + 1].filter(lm.y)
            z = self._filters[base + 2].filter(lm.z)
            coords.append(SimpleNamespace(x=x, y=y, z=z))
        return coords


class PinchDetector:
    def __init__(self, threshold: float, release_ratio: float = 1.2) -> None:
        self.threshold = threshold
        self.release_threshold = threshold * release_ratio
        self._active = False
        self._frames = 0

    def update(self, pinch: float, thumb_extended: float) -> bool:
        want_pinch = pinch <= self.threshold and thumb_extended >= 0.25
        if self._active:
            if pinch > self.release_threshold:
                self._frames = max(0, self._frames - 1)
                if self._frames == 0:
                    self._active = False
            else:
                self._frames = min(3, self._frames + 1)
        elif want_pinch:
            self._frames += 1
            if self._frames >= 2:
                self._active = True
        else:
            self._frames = 0
        return self._active

    def reset(self) -> None:
        self._active = False
        self._frames = 0


class GestureVoter:
    """Weighted temporal voting for stable gesture output."""

    def __init__(self, window: int = 7) -> None:
        self._window = window
        self._history: deque[tuple[GestureType, float]] = deque(maxlen=window)

    def reset(self) -> None:
        self._history.clear()

    def update(self, gesture: GestureType, confidence: float) -> tuple[GestureType, float]:
        self._history.append((gesture, confidence))
        if not self._history:
            return GestureType.NONE, 0.0

        weights: Counter = Counter()
        for g, c in self._history:
            weights[g] += c

        best, total_w = weights.most_common(1)[0]
        share = total_w / max(sum(weights.values()), 1e-6)

        recent = [g for g, _ in list(self._history)[-3:]]
        if recent.count(best) < 2 and len(self._history) >= 3:
            return GestureType.OTHER, share * 0.5

        return best, _clamp(share)


class HandTracker:
    def __init__(
        self,
        detection_confidence: float = 0.78,
        tracking_confidence: float = 0.78,
        max_hands: int = 1,
        model_quality: str = "accurate",
        process_width: int | None = None,
        landmark_alpha: float | None = None,
        use_gpu: bool = False,
        vote_window: int = 7,
    ) -> None:
        preset = QUALITY_PRESETS.get(model_quality, QUALITY_PRESETS["accurate"])
        if process_width is None:
            process_width = preset["process_width"]
        mincutoff = preset["mincutoff"]
        beta = preset["beta"]
        if landmark_alpha is not None:
            mincutoff = 0.5 + landmark_alpha

        model_path = ensure_model()
        base = mp.tasks.BaseOptions(model_asset_path=str(model_path))
        if use_gpu:
            try:
                base = mp.tasks.BaseOptions(model_asset_path=str(model_path), delegate=mp.tasks.BaseOptions.Delegate.GPU)
            except Exception:
                pass

        options = vision.HandLandmarkerOptions(
            base_options=base,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=max_hands,
            min_hand_detection_confidence=detection_confidence,
            min_hand_presence_confidence=tracking_confidence,
            min_tracking_confidence=tracking_confidence,
        )
        try:
            self._landmarker = vision.HandLandmarker.create_from_options(options)
        except Exception:
            base = mp.tasks.BaseOptions(model_asset_path=str(model_path))
            options = vision.HandLandmarkerOptions(
                base_options=base,
                running_mode=vision.RunningMode.VIDEO,
                num_hands=max_hands,
                min_hand_detection_confidence=detection_confidence,
                min_hand_presence_confidence=tracking_confidence,
                min_tracking_confidence=tracking_confidence,
            )
            self._landmarker = vision.HandLandmarker.create_from_options(options)

        self._process_width = process_width
        self._smoother = LandmarkSmoother(mincutoff=mincutoff, beta=beta)
        self._pinch = PinchDetector(threshold=0.35)
        self._voter = GestureVoter(window=vote_window)
        self._last_hand: HandFrame | None = None
        self._miss_count = 0
        self._hold_frames = 3

    def close(self) -> None:
        self._landmarker.close()

    def set_pinch_threshold(self, threshold: float) -> None:
        self._pinch.threshold = threshold
        self._pinch.release_threshold = threshold * 1.2

    def process(self, frame_bgr, pinch_threshold: float, timestamp_ms: int) -> HandFrame | None:
        self.set_pinch_threshold(pinch_threshold)

        h, w = frame_bgr.shape[:2]
        scale = self._process_width / w
        proc_h = max(int(h * scale), 1)
        if w != self._process_width:
            small = cv2.resize(frame_bgr, (self._process_width, proc_h), interpolation=cv2.INTER_LINEAR)
        else:
            small = frame_bgr

        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)

        if not result.hand_landmarks:
            self._miss_count += 1
            if self._last_hand is not None and self._miss_count <= self._hold_frames:
                return self._last_hand
            self._smoother.reset()
            self._pinch.reset()
            self._voter.reset()
            self._last_hand = None
            return None

        self._miss_count = 0
        landmarks = self._smoother.apply(result.hand_landmarks[0])
        scores = analyze_fingers(landmarks)
        pinch = pinch_distance(landmarks)
        thumb_ext = thumb_extended_for_pinch(landmarks)
        is_pinching = self._pinch.update(pinch, thumb_ext)
        closed_fist = is_closed_fist(landmarks, scores)
        center_x, center_y = normalized_hand_center(landmarks)

        gesture, conf = classify_gesture(landmarks, scores, is_pinching)
        stable, stable_conf = self._voter.update(gesture, conf)

        handedness = "Unknown"
        if result.handedness:
            handedness = result.handedness[0][0].category_name

        hand = HandFrame(
            landmarks=landmarks,
            gesture=gesture,
            stable_gesture=stable,
            confidence=stable_conf,
            palm_score=open_palm_score(scores),
            up_count=sum(1 for s in scores.values() if s.extended >= 0.58),
            center_x=center_x,
            center_y=center_y,
            pinch=pinch,
            index_only=gesture == GestureType.INDEX_ONLY,
            is_pinching=is_pinching,
            is_closed_fist=closed_fist,
            finger_scores=scores,
            handedness=handedness,
        )
        self._last_hand = hand
        return hand

    @staticmethod
    def draw_landmarks(frame_bgr, landmarks, finger_scores: dict[str, FingerScore] | None = None) -> None:
        h, w = frame_bgr.shape[:2]
        for conn in mp_hands.HAND_CONNECTIONS:
            x0, y0 = int(landmarks[conn[0]].x * w), int(landmarks[conn[0]].y * h)
            x1, y1 = int(landmarks[conn[1]].x * w), int(landmarks[conn[1]].y * h)
            cv2.line(frame_bgr, (x0, y0), (x1, y1), (0, 200, 80), 2)
        for i, lm in enumerate(landmarks):
            cx, cy = int(lm.x * w), int(lm.y * h)
            color = (0, 140, 255)
            if finger_scores and i in (8, 12, 16, 20):
                color = (0, 255, 120)
            cv2.circle(frame_bgr, (cx, cy), 4, color, -1)


class GestureStabilizer:
    def __init__(self, frames: int = 3) -> None:
        self.frames = frames
        self._current: GestureType = GestureType.NONE
        self._count = 0

    def update(self, gesture: GestureType) -> GestureType | None:
        if gesture == self._current:
            self._count += 1
        else:
            self._current = gesture
            self._count = 1
        if self._count >= self.frames:
            return gesture
        return None
