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

| # | Model | Source | Entangler shape (predicted from the paper's own description, verified by fetching its text — not yet traced through our parser) | Predicted outcome |
|---|---|---|---|---|
| 3 | MPS discriminative classifier | Huggins, Patil, Mitchell, Whaley, Stoudenmire, *Towards quantum machine learning with tensor networks*, Quantum Sci. Technol. 4, 024001 (2019), arXiv:1803.11537 — the paper's own MPS-circuit variant (full, non-qubit-efficient version) | Sequential chain: unitary₁ on (q1,q2) keeps one qubit forward as the "bond," unitary₂ on (bond,q3), ... through q_N — literally a linear chain by construction, and the paper explicitly parameterizes a bond dimension V (in *qubits* carried forward) analogous to our χ | **Dequantizes easily**, and specifically at **χ\* ≈ 2^V** (Schmidt rank of V qubits, not V itself — for the simplest V=1 implementation, this predicts χ\*≈2, directly comparable to model_01's χ\*=2) — a sharper, self-referential prediction than model_01's, since this circuit already *is* an MPS ansatz by construction |
| 4 | Tree tensor network classifier | Same paper (Huggins et al. 2019) — the paper's own tree-circuit variant, 8 inputs → 3 layers (4+2+1 unitaries) merging equal-sized qubit pairs hierarchically | Binary tree (verified: pairs (1,2),(3,4),(5,6),(7,8) → 2 survivors merge → 1 final) | **Not applicable** (TTN shape), confirmatory alongside model_02 — same paper as model 3, so this is a clean within-paper MPS-vs-tree comparison, exactly this tool's central question |
| 5 | Circuit-centric classifier | Schuld, Bocharov, Svore, Wiebe, *Circuit-centric quantum classifiers*, Phys. Rev. A 101, 032308 (2020), arXiv:1804.00633 — implemented via `qml.StronglyEntanglingLayers`, which PennyLane's own docs confirm is "inspired by" this exact paper (verified, not assumed) | ~~"Brick"-pattern, not yet traced~~ **UPDATE (traced before any training, see model_05/reimplementation.py docstring):** entangler connects wire i to (i+r) mod M with r varying per layer — traced for 4-6 wires/2-3 layers and confirmed **not path-decomposable, not a ring, not a tree** (genuinely tangled: 3+ distinct partners on some wires, cycles present) | **Expected to resist** (revised from "uncertain" now that it's traced — same bucket as model 6, but for a structurally different reason: accumulated multi-range entanglement across layers, not one all-to-all round) |
| 6 | ZZ feature map classifier | Havlicek et al., *Supervised learning with quantum-enhanced feature spaces*, Nature 567:209 (2019), arXiv:1804.11326 | All-to-all `IsingZZ` entangling layer, verified from the paper's own formula (`U_Phi(x)` sums over all size-≤2 subsets) — traced (before training): all 6 pairs among 4 wires present, confirmed genuinely tangled | **CONFIRMED "expected to resist"** (structurally traced, matches the original prediction exactly — genuinely non-path-decomposable, not a ring, not a tree) |

**Correction note (before any of these were run):** an earlier draft of this document
misattributed an "MPS-ansatz variant" to Grant et al. 2018 — verified by fetching that
paper's actual text that it only describes TTN and MERA variants, no MPS one. Corrected to
Huggins et al. 2019 (arXiv:1803.11537), which explicitly proposes both MPS and tree
circuit variants in one paper (confirmed by fetching its text before writing the table
above) — a better, cleaner citation for models 3/4 than the original mixed pairing. This
correction was made before running any sweep against models 3-6, so it does not compromise
the preregistration's anti-cherry-picking purpose; it is recorded here rather than silently
edited so the correction itself is auditable.

## What "running the sweep" means for each (committed now, before any run)

- Reimplement the paper's circuit faithfully (same gate structure, same encoding), verify
  it reproduces (or comes reasonably close to) the paper's own reported accuracy on a
  small dataset before touching the surrogate — same discipline as model_01's use of
  `default.qubit` as ground truth.
- Run `parser.parse_qnode` first and record the topology classification *before* deciding
  whether `MPSSurrogate` can even be attempted — this is itself a result (a confirmed
  "not applicable" for models 4 and possibly 6 is a legitimate, informative outcome, not a
  failure to fix).
- Where `MPSSurrogate` applies: χ ∈ {1,2,4,8}, 3 seeds where compute allows (see below —
  model 3's larger parameter count required dropping to 2 seeds and χ∈{1,2,4}, decided
  from a timing probe *before* running the real sweep, not from any chi\* result), from-
  scratch training per χ, same loss/optimizer/epoch budget as the reference model — the
  protocol tightened in `scripts/run_model_01_sweep.py` (NOTES_WEEK3.md).

**Per-model protocol deviations (recorded here as they happen, before seeing results):**
- **Model 3**: timed at ~2.7s/epoch for a 4-wire, 3-unitary (18-param) version — the
  originally-drafted 6-wire version timed at ~6.9s/epoch, which would have put the full
  3-seed × 3-χ × 80-epoch run at over an hour. Scaled the circuit down to 4 wires (doesn't
  change the chain topology or the χ\*≈2 prediction being tested) and to 2 seeds / 50
  epochs, keeping runtime to roughly 8-10 minutes. This is a compute-driven adjustment made
  from a timing probe, before running the actual sweep against real data.

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
