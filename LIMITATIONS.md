# Known limitations

Living document — updated as real limitations are found, not written speculatively.

## Fixed: `sweep_bond_dimension`'s "dequantized" check was a symmetric closeness band

Found (and fixed) in Week 6 (NOTES_WEEK6.md) while running `TreeSurrogate` against
model_02: `abs(metric - reference_metric) <= tolerance` wrongly fails to credit a
surrogate that *exceeds* an imperfectly-trained reference model's accuracy. Now one-sided:
`metric >= reference_metric - tolerance`. Fixed in both `dequant_engine/sweep.py` and
`scripts/run_tree_models_sweep.py`; checked it doesn't change any previously-reported
chi\* (models 1 and 3 both hit the maximum possible accuracy, 1.0, where the two
definitions coincide). Kept here rather than deleted once fixed, per this file's own
"living document" convention of recording what was actually found.

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

## `default.tensor` does NOT support adjoint/backprop, and silently mishandles batched calls

**Filed upstream:**
[PennyLaneAI/pennylane#10169](https://github.com/PennyLaneAI/pennylane/issues/10169)
(wire-order ignored) and
[PennyLaneAI/pennylane#10170](https://github.com/PennyLaneAI/pennylane/issues/10170)
(batched-call correctness bug) — full repro scripts in `upstream_reports/`.

Confirmed empirically while investigating why `scripts/run_model_01_sweep.py` took ~13
minutes for one small model (2026-09-19):

- **No adjoint or backprop differentiation at all**, even for a plain `qml.expval`
  return: both raise `"Device ... does not support {method} with requested circuit."`
  `diff_method="best"` silently falls back to parameter-shift. This means every gradient
  step costs 2 circuit evaluations *per trainable parameter* (parameter-shift's cost),
  with no faster alternative available on this device today -- a real, load-bearing
  compute constraint on how large a preregistered battery (spec §4.3, 6-10 models x 3
  seeds x up to 7 chi values each) can practically be, not just an implementation detail.
- **Passing a batch of data points to a single QNode call silently returns the wrong
  shape instead of erroring or broadcasting correctly.** Tested directly: 5 data points
  passed as `qn(weights, xs)` with `xs.shape == (5, 2)` returned a length-2 result that
  didn't match any of the 5 per-sample values computed via a Python loop (`qn(weights,
  xi)` for each `xi`) -- not a slow-but-correct broadcast, an outright wrong answer with
  no error raised. **Do not attempt to vectorize `MPSSurrogate`/`sweep.py` calls over a
  data batch dimension in a single QNode call on this device** -- the per-sample Python
  loop used throughout this codebase is slower but is the only currently-verified-correct
  approach. Revisit if a future PennyLane/quimb version documents real batching support
  for `default.tensor`.

**Practical effect on the timeline:** the ~13-minute run for one model (3 seeds x 3 chi
values x 100 epochs x 16 samples, all serial parameter-shift) suggests the full
preregistered battery (Week 5-6 on the original timeline) needs either a smaller battery,
fewer seeds/chi values, fewer epochs, or accepting a multi-hour run -- a real planning
input for Week 3's preregistration commit, not a hypothetical concern.

## `parser.py`'s qubit-reordering heuristic

The reason this module's relabeling (via `qml.map_wires`) is necessary at all, rather
than passing a reordered `wires=` list to the device, is itself a `default.tensor` bug —
filed as [PennyLaneAI/pennylane#10169](https://github.com/PennyLaneAI/pennylane/issues/10169)
(see NOTES_WEEK2.md for the original discovery).

See `dequant_engine.parser.FAILURE_MODES` for the known failure cases of the matching
layer. As of Week 3, non-path-decomposable circuits are classified into three distinct
buckets rather than one blanket rejection, each needing a different future extension:

1. **Ring** (one cycle, degree ≤ 2 everywhere) — e.g. `qml.BasicEntanglerLayers`'s default
   entangler. One unavoidable long-range edge per ring. `MPSSurrogate` refuses; a future
   fix could special-case a single long-range MPO-style coupling for just that edge.
2. **Tree** (acyclic, some degree ≥ 3) — e.g. QCNN convolution+pooling (confirmed
   empirically on `targets/model_02_qcnn_pooling`, see NOTES_WEEK3.md), and even a plain
   "star" entangler. This is structurally a tree-tensor-network (TTN) shape, not an MPS
   one — `MPSSurrogate` correctly refuses rather than forcing a bad-fit path encoding.
   **Implemented (Week 6, NOTES_WEEK6.md): `dequant_engine/tree_surrogate.py`** — built
   directly in torch rather than through `quimb`/`default.tensor` (checked and confirmed
   `default.tensor`'s `method="tn"` has no bond-dimension truncation knob to build on, see
   below). Validated the way E1 validates `MPSSurrogate`, and run against models 2/4 on
   real data (chi\*=2 both). **Important caveat, found immediately after getting that
   result and documented prominently, not buried**: unlike `MPSSurrogate`,
   `TreeSurrogate` is *not gate-structure-matched* — it's a generic tree-shaped classical
   classifier trained on data labels, sharing only the topology with the reference
   circuit, not its specific gates. Two different circuits trained on the same data get
   numerically identical `TreeSurrogate` results (confirmed: models 2 and 4's sweep
   tables are identical). A properly gate-matched version (each node's map initialized
   from/constrained by the actual quantum gate at that tree position) remains open.
   **Checked (2026-09-19) and worth recording so nobody re-checks it expecting a
   shortcut**: `default.tensor` also has a `method="tn"` (general Tensor Network) mode
   alongside `"mps"`, which sounded like it might give a TTN surrogate for free. It does
   not — `"tn"` is for *exact* large-scale simulation via a smart contraction ordering
   (`quimb`/`cotengra`), with no `max_bond_dim`-equivalent truncation knob (that kwarg is
   documented as MPS-method-specific), which is why `TreeSurrogate` was built independently
   in plain torch instead.
3. **Genuinely tangled** (a cycle present AND degree ≥ 3 somewhere) — no relabeling or
   single-edge-cut analysis helps; would need an explicit mid-circuit SWAP network.
   Confirmed distinct from cases 1/2 with a dedicated test (a 4-cycle plus one chord,
   "wheel" graph). Not addressed — expected to be rare in practice for small-scale
   published ansätze, but the audit battery hasn't confirmed that yet.
