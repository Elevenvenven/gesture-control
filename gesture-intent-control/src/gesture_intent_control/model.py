from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np

from .features import FeatureConfig, sequence_to_vector


@dataclass
class Prediction:
    label: str
    confidence: float
    probabilities: dict[str, float]


class IntentModel:
    def __init__(
        self,
        classifier,
        labels: list[str],
        feature_config: FeatureConfig,
        threshold: float,
    ) -> None:
        self.classifier = classifier
        self.labels = labels
        self.feature_config = feature_config
        self.threshold = threshold

    def predict(self, frames: list[np.ndarray], threshold: float | None = None) -> Prediction:
        vector = sequence_to_vector(frames, self.feature_config).reshape(1, -1)
        probabilities_arr = self.classifier.predict_proba(vector)[0]
        classes = list(self.classifier.classes_)
        probabilities = {
            str(label): float(prob) for label, prob in zip(classes, probabilities_arr, strict=True)
        }
        best_idx = int(np.argmax(probabilities_arr))
        label = str(classes[best_idx])
        confidence = float(probabilities_arr[best_idx])
        cutoff = self.threshold if threshold is None else threshold
        if confidence < cutoff:
            label = "unknown"
        return Prediction(label=label, confidence=confidence, probabilities=probabilities)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "classifier": self.classifier,
                "labels": self.labels,
                "feature_config": self.feature_config,
                "threshold": self.threshold,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> "IntentModel":
        payload = joblib.load(path)
        return cls(
            classifier=payload["classifier"],
            labels=list(payload["labels"]),
            feature_config=payload["feature_config"],
            threshold=float(payload["threshold"]),
        )

