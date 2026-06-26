"""Hand tracking and gesture classification (v4)."""

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
    "accurate": {"process_width": 960, "mincutoff": 0.85, "beta": 0.005},
}

FINGER_DEFS = {
    "index": (8, 7, 6, 5),
    "middle": (12, 11, 10, 9),
    "ring": (16, 15, 14, 13),
    "pinky": (20, 19, 18, 17),
}

THUMB_DEF = (4, 3, 2, 1)


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
    quality: float
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


@dataclass
class _PalmBasis:
    scale: float
    origin: tuple[float, float, float]
    x_axis: tuple[float, float, float]
    y_axis: tuple[float, float, float]
    z_axis: tuple[float, float, float]


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


def _vec(a, b) -> tuple[float, float, float]:
    return (a.x - b.x, a.y - b.y, a.z - b.z)


def _dot(u, v) -> float:
    return u[0] * v[0] + u[1] * v[1] + u[2] * v[2]


def _cross(u, v) -> tuple[float, float, float]:
    return (
        u[1] * v[2] - u[2] * v[1],
        u[2] * v[0] - u[0] * v[2],
        u[0] * v[1] - u[1] * v[0],
    )


def _norm(v) -> float:
    return math.sqrt(_dot(v, v))


def _unit(v):
    n = _norm(v)
    if n < 1e-8:
        return (0.0, 0.0, 0.0)
    return (v[0] / n, v[1] / n, v[2] / n)


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def _smoothstep(value: float, edge0: float, edge1: float) -> float:
    if edge0 == edge1:
        return 1.0 if value >= edge1 else 0.0
    t = _clamp((value - edge0) / (edge1 - edge0))
    return t * t * (3.0 - 2.0 * t)


def _distance(a, b) -> float:
    return hypot(a.x - b.x, a.y - b.y, a.z - b.z)


def _dist2d(a, b) -> float:
    return hypot(a.x - b.x, a.y - b.y)


def hand_scale(landmarks) -> float:
    return max(_dist2d(landmarks[0], landmarks[9]), 0.001)


def _angle_at_joint(a, b, c) -> float:
    v1 = _vec(a, b)
    v2 = _vec(c, b)
    m1, m2 = _norm(v1), _norm(v2)
    if m1 * m2 < 1e-8:
        return 0.0
    cos_v = _clamp(_dot(v1, v2) / (m1 * m2), -1.0, 1.0)
    return math.degrees(math.acos(cos_v))


def _palm_basis(landmarks) -> _PalmBasis | None:
    wrist, index_mcp, pinky_mcp, middle_mcp = landmarks[0], landmarks[5], landmarks[17], landmarks[9]
    scale = hand_scale(landmarks)
    x_axis = _unit(_vec(wrist, middle_mcp))
    if _norm(x_axis) < 1e-6:
        return None
    y_raw = _vec(index_mcp, pinky_mcp)
    z_axis = _unit(_cross(x_axis, y_raw))
    if _norm(z_axis) < 1e-6:
        return None
    y_axis = _unit(_cross(z_axis, x_axis))
    return _PalmBasis(
        scale=scale,
        origin=(wrist.x, wrist.y, wrist.z),
        x_axis=x_axis,
        y_axis=y_axis,
        z_axis=z_axis,
    )


def _to_palm_local(lm, basis: _PalmBasis) -> tuple[float, float, float]:
    v = (lm.x - basis.origin[0], lm.y - basis.origin[1], lm.z - basis.origin[2])
    return (
        _dot(v, basis.x_axis) / basis.scale,
        _dot(v, basis.y_axis) / basis.scale,
        _dot(v, basis.z_axis) / basis.scale,
    )


def hand_quality(landmarks) -> float:
    scale = hand_scale(landmarks)
    if scale < 0.04:
        return 0.0
    if scale > 0.55:
        return 0.55

    in_bounds = 0
    for lm in landmarks:
        if 0.01 <= lm.x <= 0.99 and 0.01 <= lm.y <= 0.99:
            in_bounds += 1
    bounds_score = in_bounds / 21.0

    basis = _palm_basis(landmarks)
    palm_flat = 0.7
    if basis and abs(basis.z_axis[2]) > 0.15:
        palm_flat = _clamp(abs(basis.z_axis[2]) * 2.5, 0.4, 1.0)

    size_score = _smoothstep(scale, 0.06, 0.12) * (1.0 - _smoothstep(scale, 0.38, 0.5))
    return _clamp(0.45 * bounds_score + 0.35 * palm_flat + 0.20 * size_score)


def score_finger(
    landmarks,
    world_landmarks,
    basis: _PalmBasis | None,
    tip_id: int,
    pip_id: int,
    mcp_id: int,
    dip_id: int,
) -> FingerScore:
    tip, pip, mcp = landmarks[tip_id], landmarks[pip_id], landmarks[mcp_id]
    w_tip, w_pip, w_mcp = world_landmarks[tip_id], world_landmarks[pip_id], world_landmarks[mcp_id]
    wrist_w = world_landmarks[0]
    palm = landmarks[9]
    scale = hand_scale(landmarks)

    angle = _angle_at_joint(w_mcp, w_pip, w_tip)
    ext_angle = _smoothstep(angle, 158.0, 175.0)
    curl_angle = _smoothstep(152.0 - angle, 8.0, 30.0)

    tip_w = _distance(w_tip, wrist_w)
    pip_w = _distance(w_pip, wrist_w)
    mcp_w = _distance(w_mcp, wrist_w)
    ext_world = _smoothstep(tip_w / max(pip_w, 1e-6), 1.08, 1.22) * _smoothstep(
        tip_w / max(mcp_w, 1e-6), 1.02, 1.15
    )

    ext_depth = _smoothstep(w_pip.z - w_tip.z, 0.006, 0.025)
    ext_y = _smoothstep(pip.y - tip.y, 0.003, 0.018)

    ext_palm = ext_y
    if basis is not None:
        tip_l = _to_palm_local(tip, basis)
        pip_l = _to_palm_local(pip, basis)
        ext_palm = _smoothstep(tip_l[0] - pip_l[0], 0.06, 0.18)

    extended = _clamp(
        0.28 * ext_angle + 0.26 * ext_world + 0.16 * ext_palm + 0.16 * ext_depth + 0.14 * ext_y
    )

    tip_to_palm = _dist2d(tip, palm) / scale
    curl_dist = _smoothstep(0.40 - tip_to_palm, 0.0, 0.10)
    curl_world = _smoothstep(pip_w - tip_w, 0.0, 0.015)
    curled = _clamp(max(curl_angle, curl_dist, curl_world, 1.0 - extended * 1.2))

    return FingerScore(extended=extended, curled=curled)


def analyze_fingers(landmarks, world_landmarks) -> dict[str, FingerScore]:
    basis = _palm_basis(landmarks)
    scores = {
        name: score_finger(landmarks, world_landmarks, basis, tip, dip, pip, mcp)
        for name, (tip, dip, pip, mcp) in FINGER_DEFS.items()
    }
    scores["thumb"] = score_finger(
        landmarks, world_landmarks, basis, THUMB_DEF[0], THUMB_DEF[1], THUMB_DEF[2], THUMB_DEF[3]
    )
    return scores


class FingerScoreSmoother:
    """Temporal EMA on finger scores — reduces single-frame flicker."""

    def __init__(self, alpha: float = 0.58) -> None:
        self.alpha = alpha
        self._prev: dict[str, FingerScore] | None = None

    def reset(self) -> None:
        self._prev = None

    def apply(self, scores: dict[str, FingerScore]) -> dict[str, FingerScore]:
        if self._prev is None:
            self._prev = scores
            return scores
        out: dict[str, FingerScore] = {}
        for name, s in scores.items():
            p = self._prev[name]
            out[name] = FingerScore(
                extended=self.alpha * s.extended + (1 - self.alpha) * p.extended,
                curled=self.alpha * s.curled + (1 - self.alpha) * p.curled,
            )
        self._prev = out
        return out


def fist_confidence(landmarks, scores: dict[str, FingerScore]) -> float:
    scale = hand_scale(landmarks)
    palm = landmarks[9]

    finger_keys = ("index", "middle", "ring", "pinky")
    if any(scores[k].extended > 0.38 for k in finger_keys):
        return 0.0

    tip_ids = (8, 12, 16, 20)
    tip_scores = [1.0 - _clamp(_dist2d(landmarks[t], palm) / scale / 0.35) for t in tip_ids]
    palm_score = sum(tip_scores) / len(tip_scores)

    thumb_tucked = 1.0 - _clamp(_dist2d(landmarks[4], landmarks[5]) / scale / 0.48)
    curl_avg = sum(scores[k].curled for k in finger_keys) / len(finger_keys)
    thumb_curled = scores["thumb"].curled

    return _clamp(0.40 * palm_score + 0.22 * thumb_tucked + 0.28 * curl_avg + 0.10 * thumb_curled)


def is_closed_fist(landmarks, scores: dict[str, FingerScore]) -> bool:
    return fist_confidence(landmarks, scores) >= 0.76


def pinch_distance(landmarks) -> float:
    return _distance(landmarks[4], landmarks[8]) / hand_scale(landmarks)


def pinch_quality(landmarks, world_landmarks, scores: dict[str, FingerScore]) -> float:
    dist_score = 1.0 - _clamp(pinch_distance(landmarks) / 0.42)
    thumb_score = scores["thumb"].extended
    index_score = scores["index"].extended

    w_tip, w_idx = world_landmarks[4], world_landmarks[8]
    w_thumb_base = world_landmarks[2]
    approach = _angle_at_joint(w_thumb_base, w_tip, w_idx)
    angle_score = _smoothstep(approach, 15.0, 45.0)

    return _clamp(0.40 * dist_score + 0.30 * thumb_score + 0.18 * index_score + 0.12 * angle_score)


def normalized_hand_center(landmarks) -> tuple[float, float]:
    wrist = landmarks[0]
    middle_mcp = landmarks[9]
    return (wrist.x + middle_mcp.x) / 2.0, (wrist.y + middle_mcp.y) / 2.0


def open_palm_score(scores: dict[str, FingerScore]) -> int:
    return sum(1 for k in ("index", "middle", "ring", "pinky") if scores[k].extended >= 0.55)


class GestureMatcher:
    """Score every gesture template; pick winner only if clearly ahead."""

    MARGIN = 0.14

    @classmethod
    def classify(
        cls,
        landmarks,
        scores: dict[str, FingerScore],
        is_pinching: bool,
        pinch_q: float,
        quality: float,
    ) -> tuple[GestureType, float]:
        if quality < 0.35:
            return GestureType.OTHER, 0.1

        candidates: dict[GestureType, float] = {}

        if is_pinching:
            candidates[GestureType.PINCH] = 0.55 + 0.45 * pinch_q

        fist_conf = fist_confidence(landmarks, scores)
        if fist_conf > 0.5:
            candidates[GestureType.FIST] = fist_conf

        idx = scores["index"].extended
        mid = scores["middle"].extended
        ring = scores["ring"].extended
        pinky = scores["pinky"].extended
        mid_c = scores["middle"].curled
        ring_c = scores["ring"].curled
        pinky_c = scores["pinky"].curled

        two = min(idx, mid, ring_c, pinky_c) * _smoothstep(idx - ring, 0.08, 0.22)
        if idx >= 0.52 and mid >= 0.52:
            candidates[GestureType.TWO_FINGERS] = two

        three = min(idx, mid, ring, pinky_c) * _smoothstep(ring - pinky, 0.06, 0.18)
        if idx >= 0.50 and mid >= 0.50 and ring >= 0.45:
            candidates[GestureType.THREE_FINGERS] = three

        index_only = min(idx, mid_c, ring_c, pinky_c) * _smoothstep(idx - max(mid, ring, pinky), 0.10, 0.28)
        if idx >= 0.52:
            candidates[GestureType.INDEX_ONLY] = index_only

        palm_vals = [scores[k].extended for k in ("index", "middle", "ring", "pinky")]
        if min(palm_vals) >= 0.50:
            candidates[GestureType.OPEN_PALM] = min(palm_vals)

        if not candidates:
            return GestureType.OTHER, 0.15 * quality

        ranked = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
        best_g, best_s = ranked[0]
        second_s = ranked[1][1] if len(ranked) > 1 else 0.0

        if best_s - second_s < cls.MARGIN and best_g not in (GestureType.PINCH, GestureType.FIST):
            return GestureType.OTHER, best_s * 0.45 * quality

        return best_g, _clamp(best_s * (0.6 + 0.4 * quality))


class OneEuroFilter:
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
    def __init__(self, mincutoff: float = 1.0, beta: float = 0.006, freq: float = 30.0) -> None:
        self._filters = [OneEuroFilter(freq, mincutoff, beta) for _ in range(21 * 3)]

    def reset(self) -> None:
        for f in self._filters:
            f.reset()

    def apply(self, landmarks) -> list:
        out = []
        for i, lm in enumerate(landmarks):
            b = i * 3
            out.append(
                SimpleNamespace(
                    x=self._filters[b].filter(lm.x),
                    y=self._filters[b + 1].filter(lm.y),
                    z=self._filters[b + 2].filter(lm.z),
                )
            )
        return out


class PinchDetector:
    def __init__(self, threshold: float, release_ratio: float = 1.22) -> None:
        self.threshold = threshold
        self.release_threshold = threshold * release_ratio
        self._active = False
        self._frames = 0

    def update(self, pinch: float, pinch_q: float) -> bool:
        want = pinch <= self.threshold and pinch_q >= 0.48
        if self._active:
            if pinch > self.release_threshold or pinch_q < 0.30:
                self._frames = max(0, self._frames - 1)
                if self._frames == 0:
                    self._active = False
            else:
                self._frames = min(4, self._frames + 1)
        elif want:
            self._frames += 1
            if self._frames >= 3:
                self._active = True
        else:
            self._frames = 0
        return self._active

    def reset(self) -> None:
        self._active = False
        self._frames = 0


class GestureVoter:
    """Weighted voting with hysteresis — resists rapid gesture switching."""

    def __init__(self, window: int = 9) -> None:
        self._history: deque[tuple[GestureType, float]] = deque(maxlen=window)
        self._locked: GestureType | None = None

    def reset(self) -> None:
        self._history.clear()
        self._locked = None

    def update(self, gesture: GestureType, confidence: float) -> tuple[GestureType, float]:
        self._history.append((gesture, confidence))
        if not self._history:
            return GestureType.NONE, 0.0

        weights: Counter = Counter()
        for g, c in self._history:
            weights[g] += c

        best, best_w = weights.most_common(1)[0]
        total = sum(weights.values())
        share = best_w / max(total, 1e-6)

        recent = [g for g, _ in list(self._history)[-4:]]
        if recent.count(best) < 2 and len(self._history) >= 4:
            return GestureType.OTHER, share * 0.4

        if self._locked and self._locked != best:
            locked_w = weights.get(self._locked, 0.0)
            if best_w < locked_w * 1.4 and share < 0.62:
                best = self._locked
                share = locked_w / max(total, 1e-6)

        if share >= 0.50 and best not in (GestureType.OTHER, GestureType.NONE):
            self._locked = best

        return best, _clamp(share)


class HandTracker:
    def __init__(
        self,
        detection_confidence: float = 0.80,
        tracking_confidence: float = 0.80,
        max_hands: int = 1,
        model_quality: str = "accurate",
        process_width: int | None = None,
        landmark_alpha: float | None = None,
        use_gpu: bool = False,
        vote_window: int = 9,
    ) -> None:
        preset = QUALITY_PRESETS.get(model_quality, QUALITY_PRESETS["accurate"])
        if process_width is None:
            process_width = preset["process_width"]
        mincutoff = preset["mincutoff"]
        beta = preset["beta"]
        if landmark_alpha is not None:
            mincutoff = 0.4 + landmark_alpha * 0.8

        model_path = ensure_model()
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
        self._score_smoother = FingerScoreSmoother(alpha=0.58)
        self._pinch = PinchDetector(threshold=0.34)
        self._voter = GestureVoter(window=vote_window)
        self._last_hand: HandFrame | None = None
        self._miss_count = 0
        self._hold_frames = 3

    def close(self) -> None:
        self._landmarker.close()

    def set_pinch_threshold(self, threshold: float) -> None:
        self._pinch.threshold = threshold
        self._pinch.release_threshold = threshold * 1.22

    def process(self, frame_bgr, pinch_threshold: float, timestamp_ms: int) -> HandFrame | None:
        self.set_pinch_threshold(pinch_threshold)

        h, w = frame_bgr.shape[:2]
        proc_h = max(int(h * self._process_width / w), 1)
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
            self._score_smoother.reset()
            self._pinch.reset()
            self._voter.reset()
            self._last_hand = None
            return None

        self._miss_count = 0
        landmarks = self._smoother.apply(result.hand_landmarks[0])
        world = result.hand_world_landmarks[0] if result.hand_world_landmarks else landmarks

        raw_scores = analyze_fingers(landmarks, world)
        scores = self._score_smoother.apply(raw_scores)
        quality = hand_quality(landmarks)

        pinch = pinch_distance(landmarks)
        pinch_q = pinch_quality(landmarks, world, scores)
        is_pinching = self._pinch.update(pinch, pinch_q)
        closed_fist = is_closed_fist(landmarks, scores)
        center_x, center_y = normalized_hand_center(landmarks)

        gesture, conf = GestureMatcher.classify(landmarks, scores, is_pinching, pinch_q, quality)
        stable, stable_conf = self._voter.update(gesture, conf * quality)

        handedness = "Unknown"
        if result.handedness:
            handedness = result.handedness[0][0].category_name

        hand = HandFrame(
            landmarks=landmarks,
            gesture=gesture,
            stable_gesture=stable,
            confidence=stable_conf,
            quality=quality,
            palm_score=open_palm_score(scores),
            up_count=sum(1 for k in ("index", "middle", "ring", "pinky") if scores[k].extended >= 0.55),
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

        tip_map = {"index": 8, "middle": 12, "ring": 16, "pinky": 20, "thumb": 4}
        for i, lm in enumerate(landmarks):
            cx, cy = int(lm.x * w), int(lm.y * h)
            color = (0, 140, 255)
            if finger_scores:
                for name, tip_id in tip_map.items():
                    if i == tip_id:
                        ext = finger_scores[name].extended
                        g = int(80 + ext * 175)
                        color = (0, g, 255 - int(ext * 100))
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
