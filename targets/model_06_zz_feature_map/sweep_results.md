# model_06_zz_feature_map — result

**Outcome: NOT APPLICABLE** (spec §1's third legitimate outcome category — "surrogate
family doesn't cover this circuit class").

The paper's own `U_Phi(x)` formula (verified by fetching the paper's text) sums over all
size-≤2 qubit subsets — an all-to-all ZZ-interaction pattern once generalized beyond the
paper's trivial n=2 hardware demo. Traced through `parser.parse_qnode` *before* any
training: all 6 pairs among 4 wires present as entangling edges — genuinely tangled (not
path, not ring, not tree). `MPSSurrogate` correctly refuses
(`tests/test_model_06_zz_feature_map.py`), exactly matching the original preregistered
prediction ("expected to resist").

No χ sweep was run. Unlike model_02/model_04 (tree-shaped — a specific, well-understood
non-MPS family with a known extension path, `TreeSurrogate`), this circuit's density means
even a future general SWAP-network-based surrogate would likely need a bond dimension
large enough to be practically uninteresting as a "classical catches up" result — this is
plausibly closer to a genuine H1 "not dequantized" case than a "not applicable" tooling
gap, though confirming that distinction properly would require actually building and
running a general non-local surrogate, which is out of scope this session.
