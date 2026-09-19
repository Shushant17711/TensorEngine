"""A small variational quantum classifier (VQC) in the style of Farhi & Neven (2018) /
Mitarai et al.'s "quantum circuit learning" (2018) -- angle-encoded data, alternating
layers of single-qubit rotations and a *linear* (non-ring) chain of entanglers, trained to
classify parity of bit strings. Chosen deliberately with a chain (not ring) entangler
pattern: this is the common alternative to ``qml.BasicEntanglerLayers``' default ring (see
NOTES_WEEK2.md / NOTES_WEEK3.md) and is explicitly path-decomposable, so this model is
usable end-to-end through ``MPSSurrogate``/``sweep_bond_dimension`` right now -- the
audit battery's first genuinely runnable positive case (unlike model_02, which the
matching layer correctly refuses -- see its own reimplementation.py docstring).

Task: learn the parity of an n-bit binary string (Farhi & Neven's original benchmark task
for this kind of classifier) -- a small, well-defined, easily-labeled synthetic dataset,
not requiring any external data download.
"""

from __future__ import annotations

import numpy as np
import pennylane as qml

N_WIRES = 4
N_LAYERS = 2
N_PARAMS = N_LAYERS * N_WIRES  # one RY angle per wire per layer


def make_parity_dataset(n_samples: int, n_bits: int, seed: int = 0):
    """All-bitstring-covering (for small n_bits) or randomly-sampled parity dataset:
    label = XOR of all bits, the classic small-scale benchmark task for this ansatz.
    """
    rng = np.random.default_rng(seed)
    if n_samples >= 2**n_bits:
        x = np.array([[(i >> b) & 1 for b in range(n_bits)] for i in range(2**n_bits)], dtype=float)
    else:
        x = rng.integers(0, 2, size=(n_samples, n_bits)).astype(float)
    y = (x.sum(axis=1).astype(int) % 2).astype(float) * 2 - 1  # labels in {-1, +1}
    return x, y


def vqc_circuit(weights, x=None):
    """``weights`` shape (N_LAYERS, N_WIRES). ``x`` (optional): a length-N_WIRES bitstring
    encoded via RX rotations before the trainable layers; omit for structure-only tracing
    (parser.parse_qnode only needs the operation sequence, not real data).
    """
    if x is not None:
        for w in range(N_WIRES):
            qml.RX(np.pi * x[w], wires=w)
    for layer in range(N_LAYERS):
        for w in range(N_WIRES):
            qml.RY(weights[layer, w], wires=w)
        for w in range(N_WIRES - 1):  # linear chain -- NOT a ring (no wire N-1 -> 0 gate)
            qml.CNOT(wires=[w, w + 1])
    return qml.expval(qml.PauliZ(N_WIRES - 1))
