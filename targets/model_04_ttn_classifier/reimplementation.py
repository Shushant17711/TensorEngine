"""Tree tensor network (TTN) classifier, Huggins, Patil, Mitchell, Whaley, Stoudenmire
(2019), *Towards quantum machine learning with tensor networks*, Quantum Sci. Technol. 4,
024001, arXiv:1803.11537 -- same paper as model_03, its tree-circuit variant, used here for
a clean within-paper MPS-vs-tree comparison (this tool's central question).

Verified circuit structure (by fetching the paper's own text): 8 input qubits -> layer 1
applies 4 two-qubit unitaries on nearest-neighbor pairs (0,1),(2,3),(4,5),(6,7), keeping
one qubit from each pair forward (4 survivors) -> layer 2 applies 2 unitaries on pairs of
survivors -> layer 3 applies 1 final unitary -> 1 output qubit. This is architecturally
identical to targets/model_02_qcnn_pooling (same tree shape, same "TTN not MPS" structural
fact -- see NOTES_WEEK3.md) but with a *verified* published citation directly framing it as
a tree tensor network classifier, and from the same paper as model_03's MPS variant, rather
than the more loosely-analogous QCNN convolution+pooling construction used for model_02.
"""

from __future__ import annotations

import numpy as np
import pennylane as qml

from targets.model_03_mps_classifier.reimplementation import two_qubit_block

N_WIRES = 8
N_PARAMS_PER_UNITARY = 6
N_UNITARIES = N_WIRES - 1  # 4 + 2 + 1 = 7, same total as model_02 (any binary tree on N leaves)
N_PARAMS = N_UNITARIES * N_PARAMS_PER_UNITARY


def ttn_classifier_circuit(weights, x=None):
    """``weights`` shape (N_UNITARIES, N_PARAMS_PER_UNITARY), consumed in tree order
    (layer 1's 4 unitaries first, then layer 2's 2, then layer 3's 1).
    """
    if x is not None:
        for w in range(N_WIRES):
            qml.RY(np.pi * x[w], wires=w)

    active = list(range(N_WIRES))
    idx = 0
    while len(active) > 1:
        survivors = []
        for a, b in zip(active[0::2], active[1::2], strict=False):
            two_qubit_block(weights[idx], wires=[a, b])
            idx += 1
            survivors.append(b)  # keep the second qubit of each pair, same convention as model_03
        active = survivors

    return qml.expval(qml.PauliZ(active[0]))
