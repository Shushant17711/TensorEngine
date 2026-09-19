"""MPS discriminative classifier, Huggins, Patil, Mitchell, Whaley, Stoudenmire (2019),
*Towards quantum machine learning with tensor networks*, Quantum Sci. Technol. 4, 024001,
arXiv:1803.11537 -- the paper's own MPS-circuit variant (full, non-qubit-efficient version;
V=1 bond dimension, the simplest case, matching the paper's Fig. 7-style schematic).

Verified circuit structure (by fetching the paper's own text, not guessed -- see
PREREGISTRATION.md's correction note): each input feature is encoded onto its own qubit
via a single-qubit rotation, then a chain of two-qubit unitaries runs sequentially --
unitary_1 acts on (q0, q1), keeping q1 forward as the "bond" and discarding q0's further
role; unitary_2 acts on (q1, q2), keeping q2 forward; ... through unitary_{N-1} acting on
(q_{N-2}, q_{N-1}). The final qubit q_{N-1} is the readout. This is, by construction,
literally a linear chain -- the same topology class as model_01, but arrived at because the
circuit family itself *is* meant to realize a classical MPS, not because we picked a chain
entangler for convenience. That makes chi* here a self-referential check: since the
circuit's own "bond line" carries a single qubit at a time (V=1 qubit of bond), any cut's
Schmidt rank is at most 2^V = 2, so chi*=2 (not just "small") is a specific, falsifiable
prediction (see PREREGISTRATION.md model 3).
"""

from __future__ import annotations

import numpy as np
import pennylane as qml

N_WIRES = 4  # number of input features / qubits -- kept small deliberately: parameter-shift
# training through default.tensor is the dominant cost (LIMITATIONS.md), and the chi*~2
# prediction being tested here doesn't depend on wire count, only on the chain topology
# and the circuit's own bond dimension V=1
N_PARAMS_PER_UNITARY = 6  # 2 single-qubit rotations (RX,RY) per qubit either side of CNOT
N_UNITARIES = N_WIRES - 1
N_PARAMS = N_UNITARIES * N_PARAMS_PER_UNITARY


def two_qubit_block(params, wires):
    """A generic, sufficiently expressive parameterized 2-qubit block: local RX+RY on
    each qubit, a CNOT, then local RX+RY again -- the paper's "simple gate" construction
    ("two arbitrary single-qubit rotations followed by a CNOT"), doubled either side of
    the CNOT for a bit more expressivity while staying within the paper's spirit.
    """
    a, b = wires
    qml.RX(params[0], wires=a)
    qml.RY(params[1], wires=a)
    qml.RX(params[2], wires=b)
    qml.RY(params[3], wires=b)
    qml.CNOT(wires=[a, b])
    qml.RX(params[4], wires=b)
    qml.RY(params[5], wires=b)


def mps_classifier_circuit(weights, x=None):
    """``weights`` shape (N_UNITARIES, N_PARAMS_PER_UNITARY). ``x`` (optional): length
    N_WIRES feature vector in [0, 1], encoded via the paper's own feature map
    (cos/sin rotation, Eq. 1: psi = cos(pi/2 * x)|0> + sin(pi/2 * x)|1>).
    """
    if x is not None:
        for w in range(N_WIRES):
            qml.RY(np.pi * x[w], wires=w)  # equivalent single-qubit-rotation encoding

    for i in range(N_UNITARIES):
        two_qubit_block(weights[i], wires=[i, i + 1])

    return qml.expval(qml.PauliZ(N_WIRES - 1))  # final "bond" qubit is the readout
