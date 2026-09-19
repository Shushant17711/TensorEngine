# Known limitations

Living document — updated as real limitations are found, not written speculatively.

## `default.tensor` cannot differentiate a `qml.state()` return (any diff_method)

Confirmed empirically while writing `tests/test_e1_toy_circuit.py` (2026-09-19):

- `diff_method="best"` (which resolves to parameter-shift here) raises
  `"Computing the gradient of circuits that return the state with the parameter-shift rule
  gradient transform is not supported, as it is a hardware-compatible method."`
- `diff_method="backprop"` and `diff_method="adjoint"` both raise
  `"Device <default.tensor ...> does not support {method} with requested circuit."`

**Practical effect:** any surrogate objective that needs to backprop through a full
statevector (e.g. training directly against state fidelity to a reference state) cannot
currently be trained through `default.tensor`. This is not a blocker for the project's
actual sweep protocol (spec §4.2), which trains against `qml.expval` measurements — those
differentiate fine via parameter-shift (confirmed in `scripts/week1_spike.py`, see
`NOTES_WEEK1.md`). It only affects state-fidelity-style validation checks; worked around in
`test_sweep_recovers_chi_star_equal_2` by evaluating the (fixed, untrained) reference
weights directly rather than training against a state-fidelity loss.

## `parser.py`'s qubit-reordering heuristic

See `dequant_engine.parser.FAILURE_MODES` for the known failure cases of the matching
layer. As of Week 3, non-path-decomposable circuits are classified into three distinct
buckets rather than one blanket rejection, each needing a different future extension:

1. **Ring** (one cycle, degree ≤ 2 everywhere) — e.g. `qml.BasicEntanglerLayers`'s default
   entangler. One unavoidable long-range edge per ring. `MPSSurrogate` refuses; a future
   fix could special-case a single long-range MPO-style coupling for just that edge.
2. **Tree** (acyclic, some degree ≥ 3) — e.g. QCNN convolution+pooling (confirmed
   empirically on `targets/model_02_qcnn_pooling`, see NOTES_WEEK3.md), and even a plain
   "star" entangler. This is structurally a tree-tensor-network (TTN) shape, not an MPS
   one — `MPSSurrogate` correctly refuses rather than forcing a bad-fit path encoding. The
   right fix is a **`TreeSurrogate`** class (parallel to `MPSSurrogate`, contracting a TTN
   via `quimb`'s tree-tensor support instead of an MPS) — not yet implemented; a natural
   Week 4+ addition once the audit battery includes more QCNN-family models.
3. **Genuinely tangled** (a cycle present AND degree ≥ 3 somewhere) — no relabeling or
   single-edge-cut analysis helps; would need an explicit mid-circuit SWAP network.
   Confirmed distinct from cases 1/2 with a dedicated test (a 4-cycle plus one chord,
   "wheel" graph). Not addressed — expected to be rare in practice for small-scale
   published ansätze, but the audit battery hasn't confirmed that yet.
