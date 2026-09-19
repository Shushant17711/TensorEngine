"""ZZ feature map classifier, Havlicek, Corcoles, Temme, Harrow, Kandala, Chow, Gambetta
(2019), *Supervised learning with quantum-enhanced feature spaces*, Nature 567:209,
arXiv:1804.11326.

Verified circuit structure (by fetching the paper's own text -- see PREREGISTRATION.md's
model 6 entry): the feature map is U_Phi(x) H^{otimes n} U_Phi(x) H^{otimes n} applied to
|0>^n (i.e. the H-wall + diagonal-encoding block repeated twice), where
U_Phi(x) = exp(i * sum_{S subseteq [n], |S|<=2} phi_S(x) * prod_{i in S} Z_i) -- single-
qubit Z-rotations plus pairwise ZZ-interactions. The paper's own n=2 hardware demo
restricted ZZ pairs to the chip's physical connectivity (trivially local for 2 qubits);
its general theoretical formula sums over *all* size-<=2 subsets, i.e. all-to-all ZZ terms
-- the version used here (n=4), since that's the version responsible for this feature
map's famous conjectured classical-hardness property, and it's the version needed to
actually test this project's "expected to resist" prediction (an all-to-all entangler is
exactly the harder, non-path-decomposable case NOTES_WEEK3.md's "wheel" graph models).

A small trainable single-qubit rotation + entangling readout layer is appended after the
(fixed, non-trainable) feature map, matching the paper's own "quantum variational
classifier" variant (Nature paper's second proposed classifier, alongside the kernel-
method one) -- trainable parameters are needed for this project's sweep protocol (spec
section 4.1: train the surrogate under the same loss/optimizer as the reference model),
which a pure fixed-feature-map kernel method doesn't provide on its own.
"""

from __future__ import annotations

import itertools

import numpy as np
import pennylane as qml

N_WIRES = 4
N_READOUT_PARAMS = N_WIRES * 2  # one RY + one RZ per wire in the trainable readout layer


def _feature_map_block(x):
    for i in range(N_WIRES):
        qml.Hadamard(wires=i)
    for i in range(N_WIRES):
        qml.RZ(2 * x[i], wires=i)  # phi_i(x) = x_i, per the paper's own example coefficients
    for i, j in itertools.combinations(range(N_WIRES), 2):
        phi_ij = (np.pi - x[i]) * (np.pi - x[j])  # the paper's own pairwise coefficient
        qml.IsingZZ(2 * phi_ij, wires=[i, j])  # all-to-all -- see module docstring


def zz_feature_map_circuit(weights, x=None):
    """``weights`` shape (N_READOUT_PARAMS,). ``x`` (optional): length-N_WIRES feature
    vector; the feature map itself is untrainable (per the paper), repeated twice.
    """
    if x is None:
        x = np.zeros(N_WIRES)  # structure-only trace (parser doesn't need real data)
    _feature_map_block(x)
    _feature_map_block(x)

    for i in range(N_WIRES):
        qml.RY(weights[2 * i], wires=i)
        qml.RZ(weights[2 * i + 1], wires=i)

    return qml.expval(qml.PauliZ(0))
