from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .dataset import load_dataset
from .features import FeatureConfig
from .model import IntentModel


def train_model(args) -> None:
    config = FeatureConfig(window_frames=args.window_frames)
    x, y, files = load_dataset(args.data_dir, config)
    counts = Counter(y)

    print(f"Loaded {len(files)} clips.")
    print("Label counts:")
    for label, count in sorted(counts.items()):
        print(f"  {label}: {count}")

    if len(counts) < 2:
        raise SystemExit("Need at least two labels to train a classifier.")

    can_stratify = all(count >= 2 for count in counts.values())
    test_size = 0.25 if len(y) >= 12 and can_stratify else None

    if test_size:
        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=test_size,
            random_state=42,
            stratify=y,
        )
    else:
        x_train, y_train = x, y
        x_test = y_test = None
        print("Not enough data for a reliable validation split; training on all clips.")

    classifier = make_pipeline(
        StandardScaler(),
        SVC(
            kernel="rbf",
            probability=True,
            class_weight="balanced",
            C=4.0,
            gamma="scale",
            random_state=42,
        ),
    )
    classifier.fit(x_train, y_train)

    if x_test is not None and y_test is not None:
        pred = classifier.predict(x_test)
        print("\nValidation report:")
        print(classification_report(y_test, pred, zero_division=0))

    labels = sorted(np.unique(y).tolist())
    model = IntentModel(
        classifier=classifier,
        labels=labels,
        feature_config=config,
        threshold=args.threshold,
    )
    model.save(args.model_out)
    print(f"\nSaved model: {Path(args.model_out).resolve()}")

