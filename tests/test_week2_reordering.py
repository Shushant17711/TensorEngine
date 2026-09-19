"""Week 2 finding: default.tensor's bond truncation is keyed to numeric wire-label order,
not to whatever the natural circuit wire order suggests -- so the matching layer's
reordering is a correctness-of-measurement issue (an unfair, inflated chi*), not just an
optimization. This test locks that fix in.

Circuit: 8 wires, RY per wire, then four *long-range* CNOTs pairing (0,7), (1,6), (2,5),
(3,4). Under the natural wire order this needs bond dimension ~8 for exact results (each
pair spans the whole chain); under a relabeling that puts each pair adjacent, it needs
only chi=2 (four independent nearly-maximally-entangled pairs, no cross-pair entanglement).
"""

from __future__ import annotations

import numpy as np
import pennylane as qml
import pytest
import torch

from dequant_engine.mps_surrogate import MPSSurrogate
from dequant_engine.parser import parse_qnode

N_WIRES = 8


def long_range_pairs_circuit(weights):
    for w in range(N_WIRES):
        qml.RY(weights[w], wires=w)
    qml.CNOT(wires=[0, 7])
    qml.CNOT(wires=[1, 6])
    qml.CNOT(wires=[2, 5])
    qml.CNOT(wires=[3, 4])
    return qml.expval(qml.PauliZ(0) @ qml.PauliZ(7))


def test_parser_finds_path_decomposable_reordering():
    dev = qml.device("default.qubit", wires=N_WIRES)
    qnode = qml.QNode(long_range_pairs_circuit, dev)
    weights = np.zeros(N_WIRES)

    topology = parse_qnode(qnode, weights)

    assert topology.max_entangling_span == 7  # natural order: worst case, (0,7)
    assert topology.is_1d_local  # degree <= 2 everywhere (each wire in exactly 1 pair), acyclic
    assert topology.needs_reordering
    assert topology.achievable_span == 1  # reordering makes every pair adjacent

    # Each entangled pair should end up adjacent under the proposed order.
    position = {wire: idx for idx, wire in enumerate(topology.wire_order)}
    for a, b in topology.entangling_pairs:
        assert abs(position[a] - position[b]) == 1


def test_reordering_recovers_exact_result_at_low_chi_natural_order_does_not():
    """The actual regression test for the Week 2 fix: MPSSurrogate must apply the
    relabeling internally so that chi=2 is genuinely sufficient here, matching exact
    default.qubit -- and, as a control, confirm the *unmapped* raw circuit at the same
    chi under the naive natural order is NOT sufficient (demonstrating the fix matters).
    """
    rng = np.random.default_rng(0)
    weights = torch.tensor(rng.uniform(0, 2 * np.pi, size=N_WIRES))

    dev_exact = qml.device("default.qubit", wires=N_WIRES)
    exact_val = float(qml.QNode(long_range_pairs_circuit, dev_exact)(weights))

    # Control: naive natural-order default.tensor at chi=2 should NOT match exact.
    dev_naive = qml.device("default.tensor", wires=N_WIRES, method="mps", max_bond_dim=2)
    naive_val = float(qml.QNode(long_range_pairs_circuit, dev_naive)(weights))
    assert abs(naive_val - exact_val) > 1e-3, (
        "naive natural-order chi=2 unexpectedly matched exact -- test circuit no longer "
        "demonstrates the ordering problem"
    )

    # MPSSurrogate (which applies the parser's relabeling) should match exact at chi=2.
    surrogate = MPSSurrogate(long_range_pairs_circuit, N_WIRES, 2, weights)
    surrogate_val = float(surrogate(weights))
    assert abs(surrogate_val - exact_val) < 1e-9


def star_circuit(weights):
    """Wire 0 entangled with 3 distinct partners -> degree 3 -> not path-decomposable.
    No linear ordering can make all three CNOTs act on adjacent wires simultaneously.
    """
    for i in range(4):
        qml.RY(weights[i], wires=i)
    qml.CNOT(wires=[0, 1])
    qml.CNOT(wires=[0, 2])
    qml.CNOT(wires=[0, 3])
    return qml.expval(qml.PauliZ(1))


def test_genuinely_nonlocal_circuit_is_rejected_not_silently_mishandled():
    dev = qml.device("default.qubit", wires=4)
    topology = parse_qnode(qml.QNode(star_circuit, dev), np.zeros(4))
    assert not topology.is_1d_local
    assert not topology.ring_wraparound_edges  # this is the "harder" case, not a ring

    with pytest.raises(NotImplementedError, match="path-decomposable"):
        MPSSurrogate(star_circuit, 4, 2, np.zeros(4))


def ring_entangler_circuit(weights):
    """qml.BasicEntanglerLayers' default entangler pattern: a ring of CNOTs
    (0,1),(1,2),(2,3),(3,0) -- degree exactly 2 everywhere, but one cycle, not a path.
    A very common real ansatz choice (hardware-efficient / periodic-boundary designs).
    """
    qml.BasicEntanglerLayers(weights, wires=range(4))
    return qml.expval(qml.PauliZ(0))


def test_ring_topology_is_distinguished_from_the_harder_nonlocal_case():
    dev = qml.device("default.qubit", wires=4)
    weights = np.zeros((2, 4))
    topology = parse_qnode(qml.QNode(ring_entangler_circuit, dev), weights)

    assert not topology.is_1d_local
    assert len(topology.ring_wraparound_edges) == 1  # one ring -> exactly one bad edge

    with pytest.raises(NotImplementedError, match="ring component"):
        MPSSurrogate(ring_entangler_circuit, 4, 2, weights)
