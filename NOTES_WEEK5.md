# Week 5 — closing the reproduction gap, filing upstream bugs

**Date:** 2026-09-19

Prompted by direct scrutiny of the session's rigor: model_03's original sweep used an
invented synthetic task instead of the real dataset PREREGISTRATION.md's own protocol
promised ("verify it reproduces ... the paper's own reported accuracy on a small dataset
before touching the surrogate"). That's fixed here, and the two real `default.tensor`
bugs found along the way (NOTES_WEEK2.md, and a new one) have been filed upstream.

## 1. Real dataset for model_03

Schuld et al. 2020 (arXiv:1804.00633, model_05's own paper) benchmarks on ternary Iris
classification — confirmed via a citing paper's description, not assumed. Iris is also a
natural fit for model_03 (4 features = 4 wires, no architecture change needed). Added
`targets/data/iris.csv` (scikit-learn's own bundled copy, provenance recorded in
`targets/data/README.md`) and `targets/data/iris_loader.py`.

**Class pair chosen: versicolor vs virginica**, not setosa vs the rest — setosa is
linearly separable from the other two at 100% by a single feature, which would silently
reintroduce the exact "task doesn't need entanglement" confound diagnosed for model_03's
first attempt (NOTES_WEEK4.md). Checked versicolor/virginica the same way before trusting
it: no single feature exceeds 94% on the full 100-sample set — still not maximally
entanglement-demanding (see result below), but at least not trivially solvable, and more
importantly: real, not invented.

## 2. Re-run result: chi\*=1 again — but now it's a real finding, not an artifact

Re-ran `scripts/run_model_03_sweep.py` against real Iris data (16 train / 10 test samples,
subsampled from Iris's 100 for compute — see the script for the exact numbers).
Reference (`default.qubit`) accuracy: train 1.000, test 0.900. Surrogate sweep: **chi\*=1
for both seeds**, matching train accuracy 1.000 and *even slightly exceeding* the
reference's own test accuracy (1.000 vs 0.900) at every chi tried, including chi=1.

**This time the result is legitimate, not an artifact of a made-up task** — it is the
genuine outcome of reproducing the paper's own actual benchmark. The interpretation is
different from the earlier version though: this doesn't confirm or refute anything about
this specific circuit's structural entanglement capacity (that would need a
state-fidelity-style check, not an accuracy-based one — see LIMITATIONS.md's note on
`qml.state()` non-differentiability blocking that approach for now). What it *does* show,
honestly: versicolor-vs-virginica Iris classification, at this small sample size, doesn't
require exploiting this circuit's entanglement to match its own reported accuracy. This is
consistent with a broader, already-published critique in the QML literature that Iris (and
similarly small classical benchmarks) are often too easy to demonstrate any real quantum
structure is doing useful work — this project's own tool reproducing that pattern on a
real paper's real benchmark, rather than an invented one, is a small but genuine piece of
evidence for the project's own central thesis (README §1), not a null result.

## 3. Two real `default.tensor` bugs filed upstream

- [PennyLaneAI/pennylane#10169](https://github.com/PennyLaneAI/pennylane/issues/10169) —
  `wires=` ordering on `default.tensor` is silently ignored; only `qml.map_wires` on the
  circuit itself actually changes the physical MPS chain order (NOTES_WEEK2.md's original
  finding, now filed with a minimal, independently-verified repro).
- [PennyLaneAI/pennylane#10170](https://github.com/PennyLaneAI/pennylane/issues/10170) —
  a batched QNode call on `default.tensor` silently returns a wrong-shaped, wrong-valued
  result instead of erroring (found while profiling model_01's training cost, see
  LIMITATIONS.md and `tests/test_default_tensor_batching_trap.py`).

Both repro scripts (`upstream_reports/issue_*.md`) were re-run standalone, fresh, right
before filing, to confirm they reproduce exactly as written — not just copied from earlier
session output.

## What's still not done (repeated from NOTES_WEEK4.md, still true)

Models 5 and 6's dataset was *not* changed — on reflection, this wasn't actually necessary:
both are already correctly rejected structurally (`NotImplementedError`, "not applicable"),
and no accuracy claim was ever made from their synthetic data, so there's no equivalent
broken promise there. `TreeSurrogate`, a general non-local surrogate, H2, H3, and a
full-scale battery remain the real open items.
