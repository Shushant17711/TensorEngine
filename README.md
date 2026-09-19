# DequantEngine: An Automated Tensor-Network Baseline Generator for PennyLane QNNs

> Given a trained quantum classifier, automatically construct the closest matched classical
> tensor-network surrogate, sweep its bond dimension, and report the exact point where the
> classical model catches up — the "dequantization threshold." A measuring instrument, not an
> advantage claim.

---

## 0. Current status (updated as work progresses)

- **Week 1** — `default.tensor` capability check: done, [`NOTES_WEEK1.md`](NOTES_WEEK1.md).
- **Week 2** — matching layer built and fixed (it was silently unused at first — a real
  bug): done, [`NOTES_WEEK2.md`](NOTES_WEEK2.md). Exact graph-theoretic classification of
  circuit topologies (path / ring / tree / genuinely tangled), each with its own handling.
- **Week 3 (started early)** — first two target models: [`NOTES_WEEK3.md`](NOTES_WEEK3.md).
  `targets/model_01_vqc_chain` runs end-to-end (**χ\* = 2**, see its `sweep_results.md`).
  `targets/model_02_qcnn_pooling` is a confirmed, legitimate **"not applicable"** result —
  QCNN pooling forms a tree-tensor-network, not an MPS, and the tool correctly says so
  rather than forcing a bad fit.
- **Preregistration committed** — [`PREREGISTRATION.md`](PREREGISTRATION.md): 4 new models,
  each with a stated outcome prediction. Explicitly excludes model_01/02 from the "real"
  battery (they were pilot models used to build the matching layer, not blind picks).
  All 4 now implemented and topology-traced (structural predictions confirmed *before* any
  training, per the preregistration's own discipline):
  - **model_03** (Huggins et al. 2019 MPS-circuit variant) — confirmed linear chain,
    dequantizes at **χ\*=1**, not the predicted χ\*≈2. First measured against an invented
    synthetic task (a real gap, caught under review — see NOTES_WEEK5.md) and re-run
    against the real Iris dataset (versicolor vs virginica); the result held. Interpreted
    honestly: Iris at this scale doesn't need this circuit's entanglement to match its own
    accuracy — a genuine instance of this project's own central thesis (§1), not a null
    result. See its `sweep_results.md`, NOTES_WEEK4.md, and NOTES_WEEK5.md.
  - **model_04** (Huggins et al. 2019 tree-circuit variant) — confirmed tree (TTN), "not
    applicable", same paper as model_03 for a clean within-paper MPS-vs-tree comparison.
  - **model_05** (Schuld et al. 2020, via `qml.StronglyEntanglingLayers`) — confirmed
    genuinely tangled; prediction revised from "uncertain" to "expected to resist" once
    traced.
  - **model_06** (Havlicek et al. 2019 ZZ feature map) — confirmed genuinely tangled
    (all 6 pairs among 4 wires), matching the original prediction exactly.
- **Battery status**: all 4 preregistered models have a final recorded outcome — see
  [`results/chi_star_table.md`](results/chi_star_table.md) for the consolidated table.
- **Two real `default.tensor` bugs found and filed upstream** (not just documented
  locally): [PennyLaneAI/pennylane#10169](https://github.com/PennyLaneAI/pennylane/issues/10169)
  (wire-order silently ignored) and
  [PennyLaneAI/pennylane#10170](https://github.com/PennyLaneAI/pennylane/issues/10170)
  (batched call silently returns a wrong-shaped result) — see `upstream_reports/` for the
  standalone, independently-verified repro scripts.
- **Not yet done**: H2 (needs more data points than this small battery gives), H3
  (trainability/memory comparison), a `TreeSurrogate` for the tree-shaped models, a
  general non-local surrogate for the tangled ones, paper write-up.

## 1. Honest framing, up front

This project makes **no claim that quantum models are fake or that dequantization "wins."**
The opposite framing has already burned researchers: a May 2026 analysis of a QCNN-on-MNIST
paper found every "quantum" model was a classical simulation, the comparison cost roughly
1,200× the runtime to save 50 MB of memory, and the paper wasn't fraudulent — it was ordinary
science under ordinary incentives. The tool built here exists to make that kind of comparison
routine and honest, not to manufacture a takedown. Every result gets reported as one of three
outcomes: **dequantized** (classical surrogate matches), **not dequantized** (quantum model
keeps an edge), or **not applicable** (surrogate family doesn't cover this circuit class). All
three are legitimate, publishable outcomes.

## 2. The actual gap — read this before writing code

**PennyLane already has a `default.tensor` device backed by tensor-network contraction (via
`quimb`), and matrix-product-state simulation of quantum circuits is not itself new.** The
2026 dequantization literature — the PRX Quantum result that QCNNs are effectively classically
simulable via low-bodyness Pauli propagation and tensor networks, and the broader "Quantum
Outpost" tracking of papers where MPS/PEPS reproduced claimed quantum utility results in
laptop-time — establishes that MPS surrogates for shallow, low-entanglement circuits are a
known, working technique.

**So the gap is not "can a tensor network simulate this circuit."** It's tooling and reporting:

- No automated middleware takes an arbitrary user-supplied PennyLane `QNode`, infers a matched
  classical MPS architecture from its gate structure, and trains it under an identical protocol
  to the quantum model — this currently requires bespoke code per paper.
- No one systematically sweeps bond dimension `χ` across a battery of published small-scale
  QML models and reports the distribution of dequantization thresholds — existing benchmarking
  suites (e.g. the Bowles/Ahmed/Schuld QML benchmarking package) compare model *classes* on
  standard datasets; they don't retrofit a matched surrogate onto an arbitrary trained model and
  report where it catches up.

**Week 1 checkpoint — done, see [`NOTES_WEEK1.md`](NOTES_WEEK1.md).** Verified empirically
(not just from docs) on this machine: `default.tensor(method="mps", max_bond_dim=χ)` exposes
χ as a real, working, tunable parameter (truncated results converge to the exact statevector
result as χ grows), and gradient-based training through it via `interface="torch"` +
`diff_method="parameter-shift"` works cleanly, at roughly 0.14s/step for a 4-qubit/8-parameter
circuit on CPU. **Conclusion: circuit-to-MPS simulation for locally-entangled circuits is not
this project's contribution — `default.tensor` already does it.**

**Week 2 finding — see [`NOTES_WEEK2.md`](NOTES_WEEK2.md), this sharpened the scope further.**
`default.tensor`'s bond truncation is keyed to numeric wire-label order and is *not*
auto-optimized — an 8-qubit circuit entangling wire pairs `(0,7),(1,6),(2,5),(3,4)` needs
χ≈8 for exact results under its natural wire order, but is exactly reproduced at χ=2 once
the wires are relabeled (`qml.map_wires`) so each entangled pair is physically adjacent. So
the ordering layer isn't a rare-case addendum — **without it, a reported χ\* is an artifact
of how a circuit's author happened to number its wires, not a real measurement of its
entanglement structure.** This project's actual scope is:

- the **matching/ordering layer**: an exact graph check (is the entanglement graph a
  disjoint union of simple paths — every wire degree ≤ 2, no cycles?) plus a `qml.map_wires`
  relabeling when one exists, applied to *every* circuit, not just visibly-scrambled ones —
  this is load-bearing for a fair χ\*, confirmed necessary even for "simple" circuits,
- the **χ-sweep and χ\* extraction pipeline** (`default.tensor` gives you one χ at a time;
  nothing sweeps it and reports a threshold),
- the **audit-battery + preregistration discipline** across published models,
- **structural prediction (H2)** and **trainability/memory reporting (H3)**, both new.

## 3. Claims you are testing

> **H1.** Across a battery of published small-scale QNN/QCNN classifiers, there exists a bond
> dimension `χ*` below which the MPS surrogate underperforms the quantum model and above which
> it matches or exceeds it. Report the distribution of `χ*` across model types, not a single
> number.

> **H2.** `χ*` is predictable in advance from the circuit's own structure — specifically, from
> its entangling-gate count and topology — without ever training the surrogate. If true, this
> turns the tool into a fast screening pass: estimate `χ*` from the circuit description alone,
> and only run the expensive training comparison on circuits where the estimate is ambiguous.

> **H3.** Where `χ*` is small (say, `χ* ≤ 8`), the classical surrogate also trains faster and
> more stably than the quantum model — fewer barren-plateau-like optimization failures. This
> gives a second, independent reason to prefer the surrogate in that regime, beyond raw accuracy
> parity.

## 4. Method

### 4.1 The matching layer

```
PennyLane QNode
    → parse operation sequence (gates, wires, parameter structure)
    → map each gate to its MPS tensor-network equivalent
      (single-qubit rotations → local tensors; two-qubit entanglers →
       bond-dimension-truncated SVD contraction)
    → construct a torch/quimb MPS model with χ as a free hyperparameter
    → train under the SAME loss, optimizer, learning rate, and epoch
      budget as the original quantum model
```

Use `quimb`'s MPS machinery for the tensor contractions; wrap it in a PyTorch `nn.Module` so
gradient-based training is identical to the quantum model's training loop. The matching step —
inferring a sensible MPS topology from an arbitrary gate sequence — is the actual engineering
work. Circuits with all-to-all entanglement (not a 1D chain) need a canonical qubit ordering
chosen to minimize truncation error; document your ordering heuristic and its failure cases.

### 4.2 The sweep

For each target model, train the surrogate at `χ ∈ {1, 2, 4, 8, 16, 32, 64}` (cap based on
VRAM), plot accuracy vs. `χ`, and record `χ*` as the smallest bond dimension within some
tolerance (e.g. 1 percentage point) of the quantum model's accuracy.

### 4.3 The audit battery

Apply the tool to 6–10 published small-scale QML classifiers (reimplemented from their papers,
same discipline as a reproducibility study): verify you can reproduce their reported accuracy
first, then run the surrogate sweep. **Preregister the model list before running any sweep** —
commit it to the repo with a timestamp — so you can't be accused of cherry-picking easy targets.
Include at least one or two models you expect to resist dequantization (deep, highly entangled
ansätze) alongside the shallow ones you expect to fold easily; a battery that's all one outcome
teaches nothing.

## 5. Experiments

- **E1 — Validation.** One hand-built toy circuit, known analytically to be MPS-representable
  at `χ=2`. Confirm the pipeline finds `χ*=2`. If it doesn't, the matching layer has a bug —
  fix before touching real models.
- **E2 — Main sweep.** The full battery, 3 seeds each, `χ*` table as the headline result.
- **E3 — Structural prediction (H2).** Regress `χ*` against circuit-level features (entangling
  gate count, max entanglement-generating depth, connectivity graph properties). Report R² and,
  more importantly, where the prediction fails badly — that failure case is itself informative
  about what the structural heuristic misses.
- **E4 — Trainability (H3).** For each model, log gradient variance and training stability
  (fraction of seeds that converge) for both the quantum model and its surrogate at `χ*`.

## 6. Metrics

- Accuracy vs. `χ` curve per model (the core plot)
- `χ*` per model, with 3-seed confidence interval
- Wall-clock training time, quantum vs. surrogate at matched accuracy
- GPU memory footprint, quantum (statevector) vs. surrogate (MPS) — this is often the more
  dramatic number: MPS memory scales roughly `O(n · χ² · d)` against the statevector's
  `O(2^n)`, so even a "not dequantized" case (accuracy doesn't match) can show a striking
  memory-cost comparison worth reporting separately.

## 7. Repository structure

```
dequant-engine/
├── README.md
├── LIMITATIONS.md
├── PREREGISTRATION.md          # target model list, committed before E2
├── dequant_engine/
│   ├── parser.py                # QNode → gate sequence → MPS topology
│   ├── mps_surrogate.py         # torch-wrapped quimb MPS model
│   ├── sweep.py                 # χ sweep + χ* extraction
│   └── structural_features.py   # circuit-level features for H2
├── targets/
│   ├── model_01_qcnn_mnist/
│   │   ├── reimplementation.py
│   │   ├── original_reported.md
│   │   └── sweep_results.md
│   └── ...
├── scripts/reproduce.sh
├── results/
│   ├── chi_star_table.md
│   ├── chi_accuracy_curves.png
│   ├── structural_prediction.png
│   └── memory_comparison.png
└── paper/main.tex
```

## 8. Timeline (9 weeks)

| Week | Work |
|---|---|
| 1 | Evaluate `default.tensor`'s real capabilities. Decide the actual scope of your contribution. |
| 2 | Build the parser + matching layer. Validate on the E1 toy circuit. |
| 3 | Select and preregister the target model battery. Reimplement 2–3 models, confirm reproduction. |
| 4 | Reimplement remaining models. |
| 5–6 | Run the full χ sweep across the battery, 3 seeds each. |
| 7 | Structural prediction analysis (H2). |
| 8 | Trainability comparison (H3), memory footprint analysis. |
| 9 | Write-up, package release. |

## 9. Compute

MPS simulation is dramatically cheaper than statevector simulation for the low-`χ` regime
this tool targets — feasible on the RTX 4060's 8GB VRAM well past where statevector simulation
would need it. The quantum-model side follows the density-matrix (10–13 qubit) or adjoint-diff
(20–24 qubit) ceilings depending on whether the target models are noiseless or noisy.

## 10. Failure modes

- **`default.tensor` already does most of this.** Addressed in §2 — pivot your contribution
  claim accordingly, don't discover this in week 6.
- **Deep, highly entangled circuits need `χ` too large to be practical.** This is not a tool
  failure — it's a "not dequantized" result, and it's the interesting half of the audit. Report
  it as such.
- **Reproduction failures.** If you can't reproduce a target paper's reported accuracy, don't
  silently drop it — report the reproduction failure rate across the battery as its own number,
  same discipline as the SurrogateBench audit idea.
- **Cherry-picking accusation.** Mitigated by preregistration (§4.3) — do not skip this step.

## 11. References

- PRX Quantum 10.1103/8qt9-72ts — *Quantum Convolutional Neural Networks are Effectively
  Classically Simulable*
- postquantum.com's May 2026 analysis of the QCNN-on-MNIST reproducibility issue (the cautionary
  framing example in §1)
- Bowles, Ahmed, Schuld — QML benchmarking package (the nearest existing tool; read this first
  to avoid duplicating it)
- `quimb` documentation — MPS construction and contraction
- PennyLane `default.tensor` device documentation — verify current capabilities before starting
- Shin, Teo, Jeong, *Dequantizing quantum machine learning models using tensor networks*, Phys.
  Rev. Research 6, 023218 (2024)
