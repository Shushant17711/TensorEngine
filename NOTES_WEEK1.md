# Week 1 checkpoint — `default.tensor` capability evaluation

**Date:** 2026-09-19
**Script:** `scripts/week1_spike.py` (`uv run python scripts/week1_spike.py`)
**Environment:** Python 3.12, PennyLane 0.45.1, quimb 1.15.0, torch 2.14.0 (CPU), pinned via `uv.lock`.

The spec (§2, §8-Week-1) is explicit that this checkpoint gates everything else: find out
how much of the "engine" `default.tensor` already provides before designing the matching
layer. Results below are from an actual run on this machine, not just documentation.

## Q1 — Is bond dimension (`chi`) a real, working, tunable parameter?

Probed a 4-wire, 2-layer circuit (RY/RZ per qubit + nearest-neighbor CNOT ladder per
layer — the same family as the planned E1 toy circuit) on `default.qubit` (exact) and on
`default.tensor(method="mps", max_bond_dim=chi)`:

| device | `<Z0 Z3>` | \|diff from exact\| |
|---|---|---|
| `default.qubit` (exact) | +0.326384 | — |
| `default.tensor`, χ=1 | +0.042558 | 2.84e-1 |
| `default.tensor`, χ=2 | +0.298866 | 2.75e-2 |
| `default.tensor`, χ=4 | +0.326384 | 1.11e-15 |
| `default.tensor`, χ=None (unlimited) | +0.326384 | 5.55e-17 |

**Verdict: PASS.** `max_bond_dim` genuinely truncates and genuinely converges to the exact
statevector result as χ grows — for this 4-qubit circuit χ=4 is already numerically exact,
which is expected (max bipartite Schmidt rank across a 4-qubit chain is 2^2=4). This
directly confirms `default.tensor` is a working MPS simulator with χ as a first-class knob,
not just documentation claiming so.

## Q2 — Does gradient-based training work cleanly through it?

Wrapped the same circuit with `interface="torch"`, `diff_method="parameter-shift"`, at
`max_bond_dim=2`, and ran 15 Adam steps against a fixed target expectation value:

- Loss trajectory: `0.0325 → 0.0003 → 0.0113 → ... → 0.0028` (noisy but clearly minimizing
  around the target — this is expected shape for a low-χ *truncated* surrogate chasing a
  target set by the untruncated circuit, not monotone convergence to zero).
- Wall-clock: **2.14s total / 0.143s per step** for a 4-qubit, 8-parameter circuit on CPU,
  parameter-shift (which itself needs 2 circuit evals per parameter — 16 evals/step here).

**Verdict: PASS**, with a caveat to track going forward: parameter-shift cost scales
linearly in parameter count (2 evals/param/step), so training cost for realistic target
models (tens to low-hundreds of parameters) needs to be watched — this is a *speed*
consideration for the sweep design (§4.2), not a correctness blocker.

## Q3 — What circuit family was actually exercised?

Only the shallow, nearest-neighbor-entangled family needed for Week 2/E1 was probed here.
All-to-all / long-range-entangled circuits (the harder case the spec §4.1 flags — "circuits
with all-to-all entanglement need a canonical qubit ordering") are **not yet tested** and
remain an open Week-2/3 question once real target models are reimplemented.

## Scope decision

`default.tensor` already supplies a real, differentiable, χ-tunable MPS simulator for
locally-entangled circuits, trainable with the exact same PennyLane/torch training loop
used for the quantum model. Per the spec's own instruction (§2, "if it already does 80%
... your contribution shifts to the matching, sweeping, and reporting layer"), **this
project's actual contribution is scoped down to:**

1. **The matching/ordering layer** for circuits that are *not* already 1D-local (canonical
   qubit ordering to minimize truncation error for all-to-all entanglers) — genuinely open,
   not solved by `default.tensor` itself.
2. **The χ-sweep and χ\* extraction pipeline** (`sweep.py`) — `default.tensor` gives you one
   χ at a time; nothing sweeps it and reports a threshold automatically.
3. **The audit battery + preregistration discipline** (§4.3) — reproducing published models
   and running the sweep across all of them systematically.
4. **Structural prediction (H2)** — regressing χ\* against circuit features. Entirely new.
5. **Trainability/memory comparison (H3)** — new reporting layer on top of existing devices.

For circuits `default.tensor` already handles well (local/1D entanglement), the "MPS
surrogate model" *is* `default.tensor` itself at bounded χ, re-using the same QNode — no
separate hand-rolled quimb+torch tensor network is needed for that case. A hand-rolled
`quimb`+torch model is kept as a fallback path in `mps_surrogate.py` for circuit families
`default.tensor` can't represent cleanly (to be determined once real target models from the
audit battery are reimplemented in Week 3).

**README §2 has been updated** to reflect this — it no longer treats circuit→MPS simulation
as this project's own invention; that already exists. The contribution is the layer on top.
