"""Loader for the real Iris dataset (see targets/data/README.md for provenance) --
replaces the earlier ad hoc synthetic tasks used for model_03/model_05's first attempt at
a reproduction (see NOTES_WEEK5.md for why that was a real gap, not a stylistic choice).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

_CSV_PATH = Path(__file__).resolve().parent / "iris.csv"

CLASS_NAMES = {0: "setosa", 1: "versicolor", 2: "virginica"}


def load_iris_binary(
    class_a: int = 1, class_b: int = 2, *, normalize: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (X, y) for the two named classes only (default: versicolor vs virginica --
    the classically non-trivial pair; setosa is linearly separable from the other two by
    a single feature at 100%, which would silently reintroduce the "task doesn't need
    entanglement" confound diagnosed in NOTES_WEEK4.md).

    ``y`` is in {-1, +1} (class_a -> -1, class_b -> +1), matching the convention used
    throughout this repo's sweep scripts.

    ``normalize=True`` (default) rescales each feature to [0, 1] via min-max over the
    selected two-class subset, suitable for angle encoding (``RY(pi * x)``-style).
    """
    rows = _CSV_PATH.read_text().strip().split("\n")[1:]
    data = np.array([[float(v) for v in row.split(",")] for row in rows])
    features, labels = data[:, :4], data[:, 4].astype(int)

    mask = (labels == class_a) | (labels == class_b)
    x = features[mask]
    y = np.where(labels[mask] == class_a, -1.0, 1.0)

    if normalize:
        x = (x - x.min(axis=0)) / (x.max(axis=0) - x.min(axis=0))

    return x, y


def train_test_split(x: np.ndarray, y: np.ndarray, *, test_frac: float = 0.3, seed: int = 0):
    rng = np.random.default_rng(seed)
    n = len(x)
    idx = rng.permutation(n)
    n_test = int(n * test_frac)
    test_idx, train_idx = idx[:n_test], idx[n_test:]
    return x[train_idx], y[train_idx], x[test_idx], y[test_idx]
