# model_04_ttn_classifier — result

**Outcome: NOT APPLICABLE** (spec §1's third legitimate outcome category — "surrogate
family doesn't cover this circuit class").

Traced through `parser.parse_qnode` before any training: 8 wires, 7 unique entangling
edges, connected, acyclic — an exact spanning tree, exactly as predicted from the paper's
own description of its tree-circuit variant (Huggins et al. 2019, arXiv:1803.11537).
`MPSSurrogate` correctly refuses (`tests/test_model_03_04_huggins.py`).

Notably, this is the *same paper* as model_03's MPS-circuit variant — the two together
give a clean within-paper MPS-vs-tree comparison, which is exactly this tool's central
question (does an MPS surrogate cover a given circuit's structure, or not). Pairing them
confirms both directions cleanly: model_03 (chain) is covered and dequantizes; model_04
(tree) is correctly identified as out of scope for an MPS-only matching layer.
