from __future__ import annotations

import numpy as np

from gesture_intent_control.features import FeatureConfig, sequence_to_vector


def main() -> None:
    frames = [np.random.rand(21, 3).astype("float32") for _ in range(18)]
    vector = sequence_to_vector(frames, FeatureConfig(window_frames=32))
    assert vector.ndim == 1
    assert vector.size > 0
    print(f"feature vector ok: shape={vector.shape}")


if __name__ == "__main__":
    main()

