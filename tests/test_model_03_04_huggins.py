"""Regression tests for the preregistered models 3/4 (PREREGISTRATION.md), both from
Huggins et al. 2019 (arXiv:1803.11537): confirms the predicted topology for each, made
*before* running any training, matches what the parser actually finds.
"""

from __future__ import annotations

import numpy as np
import pennylane as qml
import pytest

from dequant_engine.mps_surrogate import MPSSurrogate
from dequant_engine.parser import parse_qnode
from targets.model_03_mps_classifier.reimplementation import (
    N_PARAMS_PER_UNITARY as P3,
)
from targets.model_03_mps_classifier.reimplementation import (
    N_UNITARIES as U3,
)
from targets.model_03_mps_classifier.reimplementation import (
    N_WIRES as W3,
)
from targets.model_03_mps_classifier.reimplementation import (
    mps_classifier_circuit,
)
from targets.model_04_ttn_classifier.reimplementation import (
    N_PARAMS_PER_UNITARY as P4,
)
from targets.model_04_ttn_classifier.reimplementation import (
    N_UNITARIES as U4,
)
from targets.model_04_ttn_classifier.reimplementation import (
    N_WIRES as W4,
)
from targets.model_04_ttn_classifier.reimplementation import (
    ttn_classifier_circuit,
)


def test_model_03_is_a_linear_chain_as_predicted():
    dev = qml.device("default.qubit", wires=W3)
    qnode = qml.QNode(mps_classifier_circuit, dev)
    weights = np.zeros((U3, P3))

    topology = parse_qnode(qnode, weights, np.zeros(W3))

    assert topology.is_1d_local
    assert not topology.needs_reordering  # already a chain 0-1-2-...-(N-1) by construction
    assert len(topology.entangling_pairs) == W3 - 1

    # MPSSurrogate must accept it directly (no NotImplementedError).
    surrogate = MPSSurrogate(mps_classifier_circuit, W3, 2, weights, np.zeros(W3))
    assert surrogate.topology.is_1d_local


def test_model_04_is_a_tree_as_predicted():
    dev = qml.device("default.qubit", wires=W4)
    qnode = qml.QNode(ttn_classifier_circuit, dev)
    weights = np.zeros((U4, P4))

    topology = parse_qnode(qnode, weights, np.zeros(W4))

    assert not topology.is_1d_local
    assert topology.is_tree
    assert not topology.ring_wraparound_edges
    assert len(set(topology.entangling_pairs)) == W4 - 1  # spanning tree: n-1 edges

    with pytest.raises(NotImplementedError, match="TREE"):
        MPSSurrogate(ttn_classifier_circuit, W4, 2, weights, np.zeros(W4))
