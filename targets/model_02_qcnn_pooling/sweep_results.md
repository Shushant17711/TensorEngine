# model_02_qcnn_pooling — result

**Outcome: NOT APPLICABLE** (spec §1's third legitimate outcome category — "surrogate
family doesn't cover this circuit class").

`parser.parse_qnode` traced this circuit's entanglement graph and found it to be an exact
tree: 8 nodes, 7 edges, connected, acyclic (see NOTES_WEEK3.md). `MPSSurrogate` correctly
refuses to build a surrogate — building one would force a tree-tensor-network-shaped
circuit into a path (MPS) encoding, which would need a bond dimension inflated for no
reason connected to the circuit's actual structure, exactly the kind of unfair measurement
the matching layer exists to prevent (see NOTES_WEEK2.md).

No χ sweep was run — there is nothing to sweep. This is the correct, honest outcome, not a
tool failure: QCNN-style pooling architectures are naturally tree tensor networks, not
matrix product states. A future `TreeSurrogate` (see LIMITATIONS.md) is the right way to
extend coverage to this circuit class, not forcing this one through `MPSSurrogate`.

Confirmed with `tests/test_model_02_qcnn_pooling.py`.
