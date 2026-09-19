# Preregistration — target model battery

**Committed: 2026-09-19.**

## Integrity note on model_01 and model_02

`targets/model_01_vqc_chain` and `targets/model_02_qcnn_pooling` were built and run
*before* this document — they are **pilot / tool-development models**, not blind battery
members, and their results (already known: model_01 dequantizes at χ\*=2, model_02 is a
confirmed "not applicable" tree-topology case) do not count as E2 findings. They were
built specifically to stress-test the matching layer (Week 2/3, see NOTES_WEEK2.md,
NOTES_WEEK3.md) before committing to a real battery, exactly the kind of engineering
validation E1 exists for — but including them in the "real" battery below would be
indistinguishable from cherry-picking two results I already knew going in. They may be
*referenced* for context but the battery this document preregisters is the four models
below, none of which have been run yet as of this commit.

## The preregistered battery (models 3-6, not yet run)

Chosen to intentionally span the outcome space (spec §4.3: don't pick a battery that's
all one outcome), each with a stated prediction made *before* running anything:

| # | Model | Source | Entangler shape (predicted from paper's circuit diagram, not yet traced through the parser) | Predicted outcome |
|---|---|---|---|---|
| 3 | MPS-ansatz classifier | Grant et al., *Hierarchical quantum classifiers*, npj Quantum Information 4:65 (2018), arXiv:1804.03680 — the paper's own linear/MPS-shaped circuit variant | Linear chain (by construction — the paper explicitly designs this variant to mirror an MPS) | **Dequantizes easily**, confirmatory alongside model_01 |
| 4 | Tree-ansatz classifier | Same paper (Grant et al. 2018) — the paper's own tree-shaped circuit variant | Binary tree | **Not applicable** (TTN shape), confirmatory alongside model_02 -- and a cleaner citation for it than a QCNN, since this paper explicitly frames its own circuit as tree-tensor-network-based |
| 5 | Circuit-centric classifier | Schuld, Bocharov, Svore, Wiebe, *Circuit-centric quantum classifiers*, Phys. Rev. A 101, 032308 (2020), arXiv:1804.00633 | "Brick"-pattern layers of pairwise unitaries (offset nearest-neighbor blocks) — not yet traced; may turn out path-decomposable-after-reorder (like the model_01_vqc_chain synthetic case) or may not | **Uncertain** — this is deliberately the "don't already know the answer" model |
| 6 | ZZ feature map classifier | Havlicek et al., *Supervised learning with quantum-enhanced feature spaces*, Nature 567:209 (2019), arXiv:1804.11326 | All-to-all `ZZFeatureMap`-style entangling layer (every qubit pair gets an entangling gate) | **Expected to resist** (genuinely non-path-decomposable — likely the "wheel"-type case from NOTES_WEEK3.md, needing a real SWAP network our matching layer doesn't build; may land as "not applicable" for a different reason than models 2/4: not a tree, but densely tangled) |

## What "running the sweep" means for each (committed now, before any run)

- Reimplement the paper's circuit faithfully (same gate structure, same encoding), verify
  it reproduces (or comes reasonably close to) the paper's own reported accuracy on a
  small dataset before touching the surrogate — same discipline as model_01's use of
  `default.qubit` as ground truth.
- Run `parser.parse_qnode` first and record the topology classification *before* deciding
  whether `MPSSurrogate` can even be attempted — this is itself a result (a confirmed
  "not applicable" for models 4 and possibly 6 is a legitimate, informative outcome, not a
  failure to fix).
- Where `MPSSurrogate` applies: χ ∈ {1,2,4,8}, 3 seeds, from-scratch training per χ, same
  loss/optimizer/epoch budget as the reference model — the protocol tightened in
  `scripts/run_model_01_sweep.py` (NOTES_WEEK3.md).

## Known compute constraint (LIMITATIONS.md) affecting this battery's size

Model_01 alone (3 seeds × 3 χ values × 100 epochs × 16 samples, parameter-shift-only
training — `default.tensor` supports neither backprop nor adjoint differentiation, see
LIMITATIONS.md) took **~13 minutes**. Four more models at similar or larger scale puts
this battery at something like an hour or more of compute, likely more once model 5/6's
larger parameter counts and dataset sizes are accounted for. This is a real planning input,
not padding: **the battery size (4 new models, not the spec's full 6-10) is deliberately
kept small enough to actually run to completion in this project's compute budget**, rather
than preregistering an aspirational list that then quietly shrinks under time pressure
later. Expanding it (toward the spec's 6-10) is a legitimate thing to do in a *later*,
separately-dated preregistration addendum — never by editing this list after seeing
results.
