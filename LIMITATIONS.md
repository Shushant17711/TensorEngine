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

See `dequant_engine.parser.FAILURE_MODES` for the three known failure cases of the
greedy nearest-neighbor-chain ordering heuristic (dense all-to-all entanglement,
per-layer-varying entanglement patterns, and non-optimality vs. an exact linear-arrangement
solver). Not yet exercised against any real target model — `MPSSurrogate` currently refuses
to build a surrogate at all for any circuit `parser.parse_qnode` judges non-1D-local
(raises `NotImplementedError`); building the general case is deferred until a real target
model in the Week 3 audit battery actually needs it (see README §2/timeline).
