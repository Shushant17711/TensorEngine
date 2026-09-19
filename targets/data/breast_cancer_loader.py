"""Loader for the real Wisconsin Breast Cancer dataset (see targets/data/README.md for
provenance) -- used for model_02/model_04 (8-wire tree circuits), since neither Iris (only
4 features) nor a synthetic task would let those get a real, honest accuracy comparison
(the same standard applied to model_03 in NOTES_WEEK5.md).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

_CSV_PATH = Path(__file__).resolve().parent / "breast_cancer.csv"

# The first 8 of the dataset's 10 "mean"-measurement columns (mean radius, texture,
# perimeter, area, smoothness, compactness, concavity, concave points) -- a natural,
# non-arbitrary choice (the leading, most commonly cited subset of this dataset's 30
# features), not selected by trying subsets and picking one that gives a nicer result.
N_FEATURES = 8


def load_breast_cancer_8(*, normalize: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Returns (X, y): X is (569, 8), y in {-1, +1} (malignant -> -1, benign -> +1,
    matching the source CSV's own 0/1 column order).
    """
    rows = _CSV_PATH.read_text().strip().split("\n")[1:]
    data = np.array([[float(v) for v in row.split(",")] for row in rows])
    x = data[:, :N_FEATURES]
    y = np.where(data[:, -1] == 0, -1.0, 1.0)  # 0=malignant, 1=benign per the source CSV

    if normalize:
        x = (x - x.min(axis=0)) / (x.max(axis=0) - x.min(axis=0))

    return x, y
