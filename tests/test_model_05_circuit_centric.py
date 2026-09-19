"""Regression test for the preregistered model 5 (PREREGISTRATION.md): confirms the
StronglyEntanglingLayers-based circuit-centric classifier lands in the genuinely-tangled
bucket (not path, not ring, not tree), matching the traced prediction recorded before any
training was attempted.
"""

from __future__ import annotations

import numpy as np
import pennylane as qml
import pytest

from dequant_engine.mps_surrogate import MPSSurrogate
from dequant_engine.parser import parse_qnode
from targets.model_05_circuit_centric.reimplementation import (
    N_WIRES,
    circuit_centric_circuit,
    weight_shape,
)


def test_circuit_centric_is_genuinely_tangled_not_ring_or_tree():
    dev = qml.device("default.qubit", wires=N_WIRES)
    qnode = qml.QNode(circuit_centric_circuit, dev)
    weights = np.zeros(weight_shape())

    topology = parse_qnode(qnode, weights, np.zeros(N_WIRES))

    assert not topology.is_1d_local
    assert not topology.ring_wraparound_edges
    assert not topology.is_tree

    with pytest.raises(NotImplementedError, match="path-decomposable"):
        MPSSurrogate(circuit_centric_circuit, N_WIRES, 2, weights, np.zeros(N_WIRES))
