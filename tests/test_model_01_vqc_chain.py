"""Regression test for the audit battery's first fully end-to-end result
(targets/model_01_vqc_chain, see scripts/run_model_01_sweep.py and its sweep_results.md).

Not a re-run of the full training script (that's a slower, standalone script covering the
"train the reference model from scratch" side too) -- this just locks in the structural
precondition the whole result depends on: the model's entangler pattern must stay
path-decomposable (a linear chain, not a ring), or MPSSurrogate would refuse it entirely
and the sweep result would no longer apply.
"""

from __future__ import annotations

import numpy as np
import pennylane as qml

from dequant_engine.parser import parse_qnode
from targets.model_01_vqc_chain.reimplementation import N_LAYERS, N_WIRES, vqc_circuit


def test_model_01_stays_path_decomposable():
    dev = qml.device("default.qubit", wires=N_WIRES)
    qnode = qml.QNode(vqc_circuit, dev)
    weights = np.zeros((N_LAYERS, N_WIRES))

    topology = parse_qnode(qnode, weights, np.zeros(N_WIRES))

    assert topology.is_1d_local
    assert not topology.needs_reordering  # already a chain, no ring-closing gate
    assert not topology.ring_wraparound_edges
