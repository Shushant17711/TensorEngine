# Week 3 — first two target models, and the QCNN/TTN finding

**Date:** 2026-09-19

Started the target-model reimplementation work a session early (originally Week 3 on the
spec's own timeline) by stress-testing the matching layer against two architecturally
different, real-style models before committing to a preregistered battery. This surfaced
one more structural finding worth locking in before the battery is chosen.

## model_01_vqc_chain — first fully end-to-end result

A Farhi & Neven (2018) / Mitarai et al. (2018)-style angle-encoded VQC: alternating
rotation layers and a **linear chain** of CNOTs (deliberately not a ring — see
NOTES_WEEK2.md), trained to classify bitstring parity. `parser.parse_qnode` confirms it's
already 1D-local; `MPSSurrogate` runs it directly. First genuine end-to-end sweep:

| chi | accuracy |
|---|---|
| 1 | 0.500 (chance -- parity is a maximally entanglement-sensitive task) |
| 2 | 1.000 |
| 4 | 1.000 |

**chi\* = 2.** (`scripts/run_model_01_sweep.py`, results in
`targets/model_01_vqc_chain/sweep_results.md`.) Reference (`default.qubit`) accuracy: 1.000.
Note this run fine-tunes the surrogate from the trained reference weights rather than
training from scratch at each chi — a reasonable first check, but the more rigorous E2
protocol (spec section 5) is a from-scratch comparison per chi; that's the remaining
Week 3/4 methodological tightening before this counts as a preregistration-ready result.

## model_02_qcnn_pooling — a real "not applicable" outcome, found before training anything

A Cong-Choi-Lukin (2019)-style QCNN: 8 wires reduced to 1 via 3 rounds of
convolution+pooling. Running it through `parser.parse_qnode` **before** attempting any
training revealed its entanglement graph is an exact tree: 8 nodes, 7 edges, connected,
acyclic (each pooling round's surviving wire accumulates one entangling partner per round
it participates in). `MPSSurrogate` correctly refuses to build a surrogate for it.

This is not a parser gap — it's the well-known fact that **QCNNs are naturally tree tensor
networks (TTNs), not matrix product states.** Per the spec's own outcome taxonomy (section
1: dequantized / not dequantized / **not applicable** — "surrogate family doesn't cover
this circuit class"), this is a legitimate, informative "not applicable" result, and it's
exactly the kind of model the spec asks to include deliberately (section 4.3: include
"models you expect to resist dequantization ... alongside the shallow ones you expect to
fold easily"). Reported as such rather than forced through a bad-fit encoding.

## Parser extension: distinguishing degree-3+ graphs by whether they're acyclic

Testing model_02 required extending the ring-vs-tangled distinction from NOTES_WEEK2.md
one step further. A degree-3+ node doesn't automatically mean "hopelessly tangled" — if the
graph is still acyclic (a tree, as QCNN pooling and, interestingly, a simple star-shaped
entangler both are), it's a **different, well-understood** case (TTN-shaped), not the same
bucket as an actually cyclic, densely-connected graph. `CircuitTopology` now has three
mutually exclusive non-path outcomes instead of one: `ring_wraparound_edges` (one bad edge),
`is_tree` (acyclic, TTN-shaped, needs a `TreeSurrogate` some future week), and the residual
"genuinely tangled" case (cyclic AND degree >= 3 — confirmed with a dedicated test, a "wheel"
graph: a 4-cycle plus one chord). Locked in with tests in `tests/test_week2_reordering.py`
and `tests/test_model_02_qcnn_pooling.py`.

## Where this leaves the preregistration

Two real architecture families are now runnable/diagnosable through the pipeline:
path-decomposable ansätze (model_01's family) end-to-end, and QCNN-style pooling
architectures as a documented, expected "not applicable" case. `PREREGISTRATION.md` is
still deliberately empty (Week 3's actual battery commit is still pending real paper
citations and the from-scratch training protocol tightening above) — but the matching
layer itself is now in much better shape to classify whatever the battery ends up
containing, rather than crashing uninformatively on the first real ansatz that isn't a bare
CNOT chain.
