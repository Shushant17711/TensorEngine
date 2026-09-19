"""Circuit-level structural features for H2 (spec section 3/5-E3): predicting chi* from
circuit structure alone, without training the surrogate.

Stub for now -- this is Week 7 work (see README timeline) and needs a real battery of
trained (circuit, chi*) pairs from the audit battery (Week 3-6) before a regression is
meaningful. Left here as a placeholder with the intended interface so ``sweep.py`` results
have an obvious place to feed into once that data exists.
"""

from __future__ import annotations

from dequant_engine.parser import CircuitTopology


def extract_features(topology: CircuitTopology) -> dict[str, float]:
    """Candidate structural features for the H2 regression (entangling-gate count, span,
    connectivity). Not yet validated against any real chi* data -- see module docstring.
    """
    n_entangling = len(topology.entangling_pairs)
    return {
        "n_wires": float(topology.n_wires),
        "n_entangling_gates": float(n_entangling),
        "n_params": float(topology.n_params),
        "max_entangling_span": float(topology.max_entangling_span),
        "is_1d_local": float(topology.is_1d_local),
        "entangling_density": n_entangling / max(topology.n_wires, 1),
    }
