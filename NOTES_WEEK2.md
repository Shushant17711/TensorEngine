# Week 2 finding — the ordering layer is load-bearing, not optional

**Date:** 2026-09-19

Week 1 tentatively concluded that the matching/ordering layer only matters for the "hard"
all-to-all case, since `default.tensor` already handles arbitrary gate connectivity without
erroring (see NOTES_WEEK1.md, Q3). Digging in further before building real target models
surfaced a more important and more precise fact:

## The finding

`default.tensor`'s bond-dimension truncation is keyed to **numeric wire-label order**, not
to any order you declare via the device's `wires=` argument, and not auto-optimized by the
`"auto-mps"` contraction method. Concretely, on an 8-qubit circuit entangling wire pairs
`(0,7), (1,6), (2,5), (3,4)` (four independent, near-maximally-entangled pairs):

| setup | `max_bond_dim` | \|diff from exact\| |
|---|---|---|
| natural wire order (0..7) | 2 | 7.4e-2 |
| natural wire order (0..7) | 4 | 7.4e-2 |
| **relabeled** so each pair is adjacent (`qml.map_wires`) | **2** | **3.9e-16** |

Passing a custom `wires=[0,7,1,6,2,5,3,4]` list to the *device* changes nothing (verified —
identical results to the natural order at every χ tried). Only relabeling the wires *inside
the circuit itself* (via `qml.map_wires`, applied to every operation and the measurement)
changes what the physical MPS chain actually looks like.

## Why this matters more than Week 1's note suggested

This isn't a "handles it or doesn't" correctness question — `default.tensor` will happily
run *any* wire order and converge to the exact answer as χ→∞ regardless. The problem is
**fairness of the reported χ\***. A circuit whose entanglement pattern is genuinely simple
(here: 4 independent Bell-like pairs, which any reasonable measure would call
"low-complexity") gets measured as needing χ≈8 under a bad wire order and χ=2 under a good
one. Since H1/H2 are entirely about the distribution and predictability of χ\*, an
unprincipled default wire order would systematically bias every reported threshold upward —
making circuits look harder to dequantize than they are, for no reason connected to their
actual entanglement structure. **The ordering layer is what keeps χ\* an honest measurement
instead of an artifact of how a circuit's wires happened to be labeled by whoever wrote it.**

## What changed in the code

- `parser.py`: replaced the natural-order-span heuristic for "is this circuit local" with
  an exact graph check — the entanglement graph is *path-decomposable* (some linear order
  makes every entangling gate act on adjacent wires) iff every wire has degree ≤ 2 and the
  graph has no cycles. This is a necessary and sufficient condition, not a heuristic guess.
  `CircuitTopology` now reports both `max_entangling_span` (natural order — mostly an
  artifact) and `achievable_span` (post-reordering — the structurally meaningful number).
- `mps_surrogate.py`: `MPSSurrogate` now actually applies the proposed relabeling via
  `qml.map_wires` before building the device/QNode, rather than computing a `wire_order`
  and never using it (a real latent bug the E1 toy circuit didn't expose, since it happened
  to already be naturally local). Genuinely non-path-decomposable circuits (a wire with 3+
  distinct entangling partners, or an entangling cycle) still correctly raise
  `NotImplementedError` — this is a provable structural fact, not a heuristic miss, so
  there is no accuracy loss in refusing rather than guessing.
- New tests (`tests/test_week2_reordering.py`): the 8-qubit long-range-pairs circuit above
  as a regression test (naive order fails at χ=2, `MPSSurrogate` succeeds), plus a
  degree-3 "star" circuit confirmed to be correctly rejected rather than silently
  mishandled.

## Follow-up: distinguishing rings from genuinely-tangled circuits

Rechecking against a real PennyLane template (`qml.BasicEntanglerLayers`, whose default
entangler is a *ring* of CNOTs — `(0,1),(1,2),(2,3),(3,0)` — not a chain) showed the
degree/cycle check needed one more distinction. A ring has degree exactly 2 everywhere
(so it isn't the "genuinely tangled" degree-≥3 case) but does contain one cycle, so it
still isn't path-decomposable as originally defined. Rather than lump it in with the harder
case, `parser.py` now classifies it separately (`CircuitTopology.ring_wraparound_edges`):
cutting any one edge of a ring turns it into a path, so a ring is "one unavoidable
long-range edge away" from fully local — much closer to representable than a true
degree-≥3 tangle, and worth reporting as such since ring/periodic-boundary entanglers are
extremely common in real published ansätze (this is exactly the sort of thing the Week 3
audit battery would otherwise hit immediately and uninformatively). `MPSSurrogate` still
refuses to build a surrogate for rings (no SWAP-network support yet), but now says exactly
which single edge is the problem instead of a generic "not path-decomposable" message.

Also fixed while testing this: `parser.parse_qnode` previously only ever saw whatever
literal gates were queued, so a circuit built from a PennyLane template (e.g.
`qml.BasicEntanglerLayers`, `qml.StronglyEntanglingLayers`) would show up as one giant
opaque templated operation with no wires-pairs information at all. `parse_qnode` now runs
`qml.transforms.decompose(tape, stopping_condition=lambda op: len(op.wires) <= 2)` first,
so the entanglement graph reflects the actual gates being applied regardless of whether
the circuit was hand-written gate-by-gate or built from a template — this matters a lot
once real target models (built from templates, as most published ansätze are) enter the
audit battery.

## Revised scope statement (supersedes NOTES_WEEK1.md's tentative one)

The ordering/matching layer is not a minor addendum for a rare edge case — it is required
for *any* real target circuit whose author didn't happen to write entangling gates in
already-adjacent wire order (which, empirically, most published ansätze don't bother to
guarantee, since it doesn't matter for exact simulation). This raises its priority for the
Week 3 target-model reimplementation work: expect real published ansätze to need this path
regularly, not just as a rare fallback.
