# model_03_mps_classifier — sweep results

Reference (default.qubit) accuracy: **1.000** (final training loss 0.3585, lr=0.2, epochs=50)

**Prediction check (PREREGISTRATION.md model 3): chi\* ≈ 2 expected (circuit's own bond dimension V=1 => Schmidt rank cap 2^V=2).**

chi sweep, 2 seeds ([0, 1]), from-scratch training per chi:

| chi | mean accuracy | range |
|---|---|---|
| 1 | 1.000 | 1.000–1.000 |
| 2 | 1.000 | 1.000–1.000 |
| 4 | 1.000 | 1.000–1.000 |

**chi* per seed: [1, 1] (2/2 seeds dequantized within the range tried)**

## The prediction was not confirmed — and the reason is itself the finding

χ\*=1, not χ\*≈2 as predicted. Diagnosed rather than just reported (same discipline as
E1's "if it doesn't match, find the bug before moving on"): checked whether the synthetic
dataset (`make_binary_dataset` — label = sign of first-half-sum minus second-half-sum) can
already be solved by a simple threshold on a single raw feature. It nearly can: feature 1
alone gets 91.7% accuracy with one threshold, on this particular 12-sample draw.

**This reveals a real distinction the original prediction conflated:** "χ\* needed to
match the reference model's *task accuracy*" and "χ\* needed to exactly reproduce the
reference *circuit's full state/behavior*" are different quantities, and only coincide when
the task specifically requires exploiting the circuit's entanglement. model_01's parity
task forces this — parity is providably unsolvable by any product state (chi=1 measured
exactly chance, 0.5), so χ\*=2 there really does reflect the circuit's entangling
structure. This session's synthetic linear-threshold-style task for model_03 does *not*
force it — a chi=1 (product-state) surrogate can already get most of the way there via
each step's chi=1 truncation still being the *best rank-1 (mean-field) approximation* of
the true post-gate state, which is a function of upstream qubits' data even though it
discards their genuine correlations. That's enough signal for an easy, correlated,
small-sample synthetic task, even though it would fail on a task that specifically needs
correlations (like parity).

**Correct interpretation:** this is not a refutation of the paper's own MPS-circuit
architecture, and not evidence against the "path-decomposable circuits dequantize easily"
finding from model_01 — it's evidence that **χ\* is task-dependent, not purely a property
of the ansatz's own structure**, and that an entanglement-insensitive task will always
report a lower (weaker, less informative) χ\* than the ansatz's structural bond dimension
would suggest. This is a real methodological lesson for the eventual audit battery (spec
§4.3): **task choice matters as much as circuit choice** for getting an honest χ\* — a
future battery should prefer tasks known or checkable to require genuine multi-qubit
correlation (like parity), not just any linearly-flavored synthetic label, when the goal
is measuring the circuit's *representational* dequantization threshold rather than its
*task-specific* one. See NOTES_WEEK4.md for the broader writeup.
