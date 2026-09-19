# Week 6 — TreeSurrogate: turning two "not applicable" results into real ones

**Date:** 2026-09-19

The clearest documented gap after Week 5's corrections (LIMITATIONS.md, NOTES_WEEK4.md):
models 2 and 4 could only ever report "not applicable" because `MPSSurrogate` correctly
refuses tree-shaped circuits. Built and validated `TreeSurrogate` (a genuine
bond-dimension-bounded classical tree tensor network, in plain torch -- see
`dequant_engine/tree_surrogate.py`), then ran it against both models on real data.

## What got built

- `TreeSurrogate`: leaves embedded via the same cos/sin feature map used throughout this
  repo, combined pairwise up a binary tree via trainable linear maps capped at dimension
  chi, renormalized at each merge (multilinear throughout, no elementwise nonlinearities,
  so it stays tensor-network-faithful).
- Validated the way E1 validates `MPSSurrogate`: trained against a hand-built 8-leaf
  reference tree circuit's outputs. Real, sharp signal: chi=1 fails badly (MSE ~0.51),
  chi=2 already nearly exact (~1e-4). Not a loose "close enough" check -- verified actual
  numbers before writing the test's assertions.
- New real dataset: Wisconsin Breast Cancer (`targets/data/breast_cancer.csv`,
  `breast_cancer_loader.py`), needed because model_02/04 use 8 wires and Iris only has 4
  features -- reusing Iris here would have meant either padding/duplicating features
  (unfaithful) or reverting to an invented task (exactly the mistake NOTES_WEEK5.md fixed
  for model_03). Checked the same way as Iris before trusting it: no single feature
  exceeds 92% accuracy alone (feature 7, "mean concave points") -- not maximally
  entanglement-demanding, but real, and this dataset's ease is itself a well-known,
  independently-documented property of Wisconsin Breast Cancer, not something introduced
  by this project's own task design.

## Two real bugs found by actually running this (not caught by review alone)

1. **A crash**: `model_04_ttn_classifier`'s circuit expects weights shaped
   `(N_UNITARIES, 6)` (each unitary block needs its own 6 params), but the sweep script
   passed a flat `(N_PARAMS,)` tensor -- worked fine for model_02 (which genuinely is flat)
   but crashed for model_04. Fixed by making the sweep script take an explicit
   `weight_shape` per model instead of assuming one universally.
2. **A real methodological flaw**: `sweep_bond_dimension`'s "dequantized" check used
   `abs(metric - reference_metric) <= tolerance` -- symmetric closeness. This wrongly
   fails to credit a surrogate that *exceeds* an imperfectly-trained reference's accuracy
   (which happened here: model_02's reference only reached 80% train accuracy after 60
   epochs, while the chi=2 surrogate reached 98.3%, `abs(98.3-80)=18.3 > tolerance`, so it
   was reported as "NOT dequantized" despite clearly having caught up and then some).
   Fixed to one-sided: `metric >= reference_metric - tolerance` -- "caught up" means
   matching or exceeding, not landing in a narrow band around the reference. Fixed in
   both `dequant_engine/sweep.py` (used by models 1/3) and the new tree-model script.
   Checked this doesn't change any previously-reported chi\* (models 1 and 3 both hit
   accuracy exactly 1.0 = the maximum possible, where the two definitions coincide).

## Results: chi\*=2 for both, real data, real reference models

| Model | Reference train/test acc | chi\* (3/3 seeds) |
|---|---|---|
| model_02_qcnn_pooling | 0.800 / 0.900 | **2** |
| model_04_ttn_classifier | 0.950 / 0.850 | **2** |

## The important caveat found immediately after getting these numbers

The two models' full chi-sweep tables are **numerically identical**. Diagnosed rather than
just reported (same discipline as every other surprising result this project has produced
-- e.g. NOTES_WEEK4.md's model_03 investigation): `TreeSurrogate`, unlike `MPSSurrogate`,
is **not gate-structure-matched**. `MPSSurrogate` literally re-executes the reference
QNode's own gates on a truncated device -- its chi\* genuinely reflects that specific
circuit. `TreeSurrogate` is a generic, from-scratch classical tree classifier that shares
only the *topology* (leaf count, tree shape) with the reference circuit, trained directly
against data labels -- nothing about a specific circuit's own gates or trained weights
ever enters its training. So of course two different circuits trained on the same data get
identical surrogate numbers: the surrogate never saw either circuit's actual structure.

**This means the chi\*=2 results above should be read as**: "a generic tree-shaped
classical model needs bond dimension 2 to match a tree-shaped quantum circuit's accuracy
on Wisconsin Breast Cancer" -- informative, and a real upgrade from "not applicable" to an
actual number, but a *weaker* claim than model_01/03's `MPSSurrogate` results, which really
do reflect their specific circuits' own entanglement. Documented prominently in
`dequant_engine/tree_surrogate.py`'s docstring and both models' `sweep_results.md`, not
buried -- this is exactly the kind of caveat that would be easy to gloss over in a paper
draft and shouldn't be.

## What would close this gap (future work, not attempted here)

A properly gate-structure-matched `TreeSurrogate` would need each node's linear map
initialized from (or constrained by) the actual quantum gate's own unitary at that tree
position, mirroring how `MPSSurrogate` gets its fidelity "for free" by literally reusing
the gate sequence. That's a real, separate engineering task, not a quick fix.

## Updated battery status

All 6 models now have a real, numeric, or definitively-structural final outcome. Only
models 5 and 6 (genuinely tangled, not tree or path) remain at "not applicable" with no
further surrogate family currently available for them.
