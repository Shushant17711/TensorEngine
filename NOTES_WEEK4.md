# Week 4 (started early) — preregistered battery: 4 models implemented and traced

**Date:** 2026-09-19

Following the preregistration commit (PREREGISTRATION.md), implemented and structurally
verified all four preregistered models before running any training — the same discipline
used for models 1/2, now applied at battery scale.

## What got built

| Model | Paper | Verified topology | Outcome |
|---|---|---|---|
| model_03 | Huggins et al. 2019 (arXiv:1803.11537), MPS-circuit variant | Linear chain (confirmed) | Sweep running — testing χ\*≈2 self-referential prediction |
| model_04 | Same paper, tree-circuit variant | Tree, n-1 edges (confirmed) | **Not applicable** (TTN) |
| model_05 | Schuld et al. 2020 (arXiv:1804.00633), via `qml.StronglyEntanglingLayers` | Genuinely tangled (confirmed, revised from "uncertain") | **Not applicable** (dense, no relabeling helps) |
| model_06 | Havlicek et al. 2019 (arXiv:1804.11326), ZZ feature map | Genuinely tangled, all-to-all (confirmed) | **Not applicable**, matches original prediction exactly |

## Methodology notes worth keeping

- **Verify citations by fetching the actual paper text, not memory.** Caught one real
  error this way: an early draft attributed an "MPS-ansatz variant" to Grant et al. 2018,
  which turned out (on fetching the paper) to only contain TTN and MERA variants. Corrected
  to Huggins et al. 2019 before any model was built against the wrong citation.
- **Trace structure through the parser before writing predictions into the permanent
  record, and correct predictions when the trace disagrees — visibly, not silently.**
  model_05 was originally marked "uncertain" in the preregistration; once actually traced,
  it was clearly non-path-decomposable, and the document was updated to say so *before* any
  training was attempted against it, with the correction itself left visible in the file
  rather than the original guess being erased.
- **Prefer a paper's own reference implementation over hand-rolled gates when one exists
  and is verifiably tied to the paper** (`qml.StronglyEntanglingLayers`, confirmed via
  PennyLane's own docs to implement Schuld et al.'s design) — reduces the chance of a
  transcription bug relative to reading a circuit diagram and re-deriving gates by hand.
- **Time a representative slice before committing to a full multi-seed run.** model_03's
  originally-drafted 6-wire version was projected (from a 3-epoch timing probe) to take
  over an hour for the full protocol; scaled to 4 wires and 2 seeds *before* running the
  real sweep, based on the timing probe alone — not on any accuracy or χ\* result.

## Battery outcome shape so far

Three of four new models (4, 5, 6) land as "not applicable" — the current MPS-only
matching layer's scope (path/ring/tree, per LIMITATIONS.md) doesn't cover dense or
non-tree-non-path graphs. This is itself informative: real published small-scale QML
ansätze skew toward either simple chains (rare, mostly toy/pedagogical circuits like
model_01/03) or deliberately expressive, densely-entangled designs (models 5/6) — the
"path-decomposable" middle ground this tool's MPS surrogate covers may be a narrower slice
of real published architectures than the original spec anticipated. This is worth stating
plainly in the eventual paper rather than treating a battery skewed toward "not applicable"
as a shortcoming of this session's model choices — see PREREGISTRATION.md's outcome
predictions, made before any of this was known, which correctly anticipated most of it.

## Update: model_03's sweep finished, and the result overturned a prediction — usefully

χ\*=1, not the predicted χ\*≈2 (see `targets/model_03_mps_classifier/sweep_results.md`
for the full diagnosis). Root cause, confirmed by checking: the synthetic classification
task used for model_03 (linear-threshold-style, unlike model_01's parity task) doesn't
actually require the circuit's entanglement to solve — a chi=1 (product-state) surrogate
can already reach the reference accuracy via a "mean-field" (best-rank-1-approximation)
effect that still carries some signal from upstream qubits even without preserving real
correlations. Parity (model_01) provably can't be solved this way (chi=1 measured exactly
chance, 0.5), which is why model_01's χ\*=2 is a cleaner, more structurally meaningful
number than model_03's χ\*=1 turned out to be.

**The methodological lesson generalizes beyond model_03**: χ\* as measured by task
accuracy conflates two different things — the ansatz's own structural bond dimension, and
how much correlation the *task* actually demands. Getting an honest, structurally
meaningful χ\* requires a task known (or checkable) to need genuine multi-qubit
correlation, not just any plausible-looking synthetic label. This is worth stating
explicitly in `PREREGISTRATION.md`-style protocol notes for any future battery expansion,
and is arguably a more interesting, generalizable finding than a clean chi*≈2 confirmation
would have been — exactly the kind of thing spec §1 means by treating every legitimate
outcome (including a falsified prediction, diagnosed rather than buried) as reportable.

## What's still open
- The general non-local surrogate (SWAP network) that would let models 5/6 produce a real
  H1 accuracy comparison instead of stopping at "not applicable" — flagged in
  `targets/model_06_zz_feature_map/sweep_results.md` as plausibly a genuine H1 "not
  dequantized" case being mislabeled as a tooling gap, which is exactly the kind of
  ambiguity that extension would resolve.
- `TreeSurrogate` for models 2/4 (a much more tractable near-term extension than the
  general SWAP-network case, since TTN contraction is well-supported by `quimb`).
- H2 (structural prediction) now has real data points to work with for the first time:
  6 models total with known topology classes, though still far short of enough data for a
  meaningful regression — that remains a later-week task once the battery is larger.
