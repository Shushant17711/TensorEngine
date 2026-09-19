"""Regression test locking in the Week 3 finding for targets/model_02_qcnn_pooling: its
entanglement graph is a tree (TTN shape), correctly distinguished from both the
path-decomposable and the "genuinely tangled" (cyclic, degree>=3) cases, and MPSSurrogate
correctly refuses to build an MPS surrogate for it rather than silently mishandling it.
"""

from __future__ import annotations

import numpy as np
import pennylane as qml
import pytest

from dequant_engine.mps_surrogate import MPSSurrogate
from dequant_engine.parser import parse_qnode
from targets.model_02_qcnn_pooling.reimplementation import N_PARAMS, N_WIRES, qcnn_circuit


def test_qcnn_pooling_forms_a_tree():
    dev = qml.device("default.qubit", wires=N_WIRES)
    qnode = qml.QNode(qcnn_circuit, dev)
    weights = np.zeros(N_PARAMS)

    topology = parse_qnode(qnode, weights)

    assert not topology.is_1d_local
    assert not topology.ring_wraparound_edges  # not a ring
    assert topology.is_tree  # acyclic despite degree >= 3 somewhere
    # 8 wires, 7 unique entangling edges, connected, acyclic -- exactly a spanning tree.
    assert len(set(topology.entangling_pairs)) == N_WIRES - 1


def test_mps_surrogate_refuses_tree_with_specific_message():
    with pytest.raises(NotImplementedError, match="TREE"):
        MPSSurrogate(qcnn_circuit, N_WIRES, 2, np.zeros(N_PARAMS))
