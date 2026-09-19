# model_05_circuit_centric — result

**Outcome: NOT APPLICABLE** (spec §1's third legitimate outcome category — "surrogate
family doesn't cover this circuit class").

`qml.StronglyEntanglingLayers`' default entangling pattern (wire i to (i+r) mod M, r
varying per layer) was traced through `parser.parse_qnode` for 4-6 wires / 2-3 layers
*before* any training was attempted (PREREGISTRATION.md's original "uncertain" prediction
was revised to "expected to resist" once this trace was done — see that file's model 5
entry). The combined entanglement graph across layers has wires with 3+ distinct entangling
partners AND contains cycles — neither a path, a ring, nor a tree. `MPSSurrogate` correctly
refuses (`tests/test_model_05_circuit_centric.py`).

No χ sweep was run — the matching layer's current scope (path / ring / tree, per
LIMITATIONS.md) doesn't cover this circuit class. A general non-local surrogate (explicit
mid-circuit SWAP network) is the correct extension, not attempted this session.
