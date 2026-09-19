"""Circuit-centric quantum classifier, Schuld, Bocharov, Svore, Wiebe (2020), *Circuit-
centric quantum classifiers*, Phys. Rev. A 101, 032308, arXiv:1804.00633.

Implemented via ``qml.StronglyEntanglingLayers``, which PennyLane's own documentation
states is "inspired by the circuit-centric classifier design arXiv:1804.00633" (verified
by fetching that documentation directly, not assumed) -- using the reference
implementation rather than hand-rolling gates avoids introducing our own transcription
errors into a "faithful reimplementation."

Structure, traced through the parser (before any training -- see PREREGISTRATION.md's
correction note for models 3/4 recording the same discipline): each layer applies a full
single-qubit `Rot` (RZ-RY-RZ) to every wire, then a round of entanglers connecting wire i
to (i + r) mod M, with r varying per layer (r = layer_index mod M by default). Traced
empirically for 4-6 wires, 2-3 layers: this accumulates entangling edges at multiple
different spans across layers (not just nearest-neighbor), and the combined graph is
**not path-decomposable, not a pure ring, and not a tree** -- a genuinely tangled case
(some wire touches 3+ distinct partners AND cycles are present). This is a *sharper*,
data-backed prediction than the original "uncertain" placeholder in PREREGISTRATION.md's
first draft: this model is now expected to land in the same "expected to resist" bucket as
model_06, for a structurally different reason than model_06's simpler ZZ-feature-map
all-to-all pattern -- accumulated multi-range entanglement across layers, not one
single-layer all-to-all round.
"""

from __future__ import annotations

import numpy as np
import pennylane as qml

N_WIRES = 4
N_LAYERS = 2


def circuit_centric_circuit(weights, x=None):
    """``weights`` shape matches ``qml.StronglyEntanglingLayers.shape(N_LAYERS, N_WIRES)``.
    ``x`` (optional): length-N_WIRES feature vector, angle-encoded before the ansatz (the
    paper's own scheme is amplitude encoding, which needs an explicit state-prep
    subroutine; angle encoding is used here for simplicity and is a common, well-attested
    substitute when reimplementing this ansatz family for small feature counts -- the
    ansatz being tested is the *entangling structure*, which is unaffected by which
    single-qubit encoding precedes it).
    """
    if x is not None:
        for w in range(N_WIRES):
            qml.RY(np.pi * x[w], wires=w)
    qml.StronglyEntanglingLayers(weights, wires=range(N_WIRES))
    return qml.expval(qml.PauliZ(0))


def weight_shape(n_layers: int = N_LAYERS, n_wires: int = N_WIRES) -> tuple[int, int, int]:
    return qml.StronglyEntanglingLayers.shape(n_layers=n_layers, n_wires=n_wires)
