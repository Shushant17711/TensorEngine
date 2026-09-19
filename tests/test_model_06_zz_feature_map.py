"""Regression test for the preregistered model 6 (PREREGISTRATION.md): confirms the
all-to-all ZZ feature map lands in the genuinely-tangled bucket, matching the prediction.
"""

from __future__ import annotations

import numpy as np
import pennylane as qml
import pytest

from dequant_engine.mps_surrogate import MPSSurrogate
from dequant_engine.parser import parse_qnode
from targets.model_06_zz_feature_map.reimplementation import (
    N_READOUT_PARAMS,
    N_WIRES,
    zz_feature_map_circuit,
)


def test_zz_feature_map_is_genuinely_tangled():
    dev = qml.device("default.qubit", wires=N_WIRES)
    qnode = qml.QNode(zz_feature_map_circuit, dev)
    weights = np.zeros(N_READOUT_PARAMS)

    topology = parse_qnode(qnode, weights, np.zeros(N_WIRES))

    assert not topology.is_1d_local
    assert not topology.ring_wraparound_edges
    assert not topology.is_tree
    # All-to-all: every pair of the 4 wires should appear as an entangling edge.
    assert len(set(topology.entangling_pairs)) == 4 * 3 // 2

    with pytest.raises(NotImplementedError, match="path-decomposable"):
        MPSSurrogate(zz_feature_map_circuit, N_WIRES, 2, weights, np.zeros(N_WIRES))
