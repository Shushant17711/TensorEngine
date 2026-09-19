# Real datasets used by target-model reproductions

## `iris.csv`

Fetched 2026-09-19 from
`https://raw.githubusercontent.com/scikit-learn/scikit-learn/main/sklearn/datasets/data/iris.csv`
(scikit-learn's own bundled copy of the classic Fisher/Anderson Iris dataset — chosen as a
stable, canonical source rather than re-deriving it, since this is the same data file
`sklearn.datasets.load_iris()` reads). 150 samples, 4 features (sepal length/width, petal
length/width, cm), 3 classes (0=setosa, 1=versicolor, 2=virginica), 50 samples per class.

Used because Schuld, Bocharov, Svore, Wiebe, *Circuit-centric quantum classifiers* (Phys.
Rev. A 101, 032308, arXiv:1804.00633) explicitly benchmarks on ternary Iris classification
(confirmed via a citing paper's description — see PREREGISTRATION.md/NOTES_WEEK5.md for the
verification trail) — replacing the earlier synthetic, made-up task used for
`targets/model_05_circuit_centric` and (for a fair same-footing comparison, since it has
the same 4-feature/4-wire shape) `targets/model_03_mps_classifier`.

**Class pair used for the binary reproductions in this repo:** versicolor (1) vs. virginica
(2) — the classically non-trivial pair (setosa is linearly separable from the other two
with 100% accuracy by a single feature, which would silently reintroduce the same
"task doesn't need entanglement" confound diagnosed in NOTES_WEEK4.md for model_03's
original synthetic task). The paper itself does full ternary classification; this repo
does the harder binary sub-problem as a first, honest step — see each model's
`original_reported.md` for what would be needed to close that remaining gap.

## `breast_cancer.csv`

Fetched 2026-09-19 from
`https://raw.githubusercontent.com/scikit-learn/scikit-learn/main/sklearn/datasets/data/breast_cancer.csv`
(scikit-learn's own bundled copy of the Wisconsin Diagnostic Breast Cancer dataset). 569
samples, 30 features, binary label (0=malignant, 1=benign).

Used for `targets/model_02_qcnn_pooling` and `targets/model_04_ttn_classifier` (both
8-wire tree circuits) — Iris only has 4 features, and inventing a synthetic 8-feature task
would repeat the exact mistake NOTES_WEEK5.md corrected for model_03. `breast_cancer_loader.py`
selects the first 8 of the dataset's 10 "mean"-measurement columns (mean radius, texture,
perimeter, area, smoothness, compactness, concavity, concave points) — the leading,
commonly-cited subset of this well-known dataset's 30 features, chosen for being a natural,
non-arbitrary subset, not by trying options and keeping whichever gave a nicer result.
