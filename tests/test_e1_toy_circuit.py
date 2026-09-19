"""E1 -- Validation (spec section 5): one hand-built toy circuit, known analytically to be
MPS-representable at chi=2. Confirm the pipeline finds chi*=2. Per the spec: "If it doesn't,
the matching layer has a bug -- fix before touching real models."

Toy circuit: a single layer of nearest-neighbor entanglers on a 1D chain (RY rotations +
one ladder of CNOTs, no long-range gates). A single ladder of two-qubit gates on an
initially product state builds at most one Schmidt-rank-2 bond per cut for a chain this
shallow (each CNOT can only double the Schmidt rank across the one bond it acts on, and
there's exactly one layer of them), so this circuit's *exact* state is representable with
bond dimension chi=2 everywhere -- chi=1 (product state) must lose fidelity, chi>=2 must
recover the exact circuit.
"""

from __future__ import annotations

import numpy as np
import pennylane as qml
import torch

from dequant_engine.mps_surrogate import MPSSurrogate
from dequant_engine.parser import parse_qnode
from dequant_engine.sweep import sweep_bond_dimension

N_WIRES = 4


def toy_circuit(weights):
    """RY per qubit, then one ladder of CNOTs, then measure a global observable that's
    sensitive to the entanglement the CNOT ladder introduces.
    """
    for w in range(N_WIRES):
        qml.RY(weights[w], wires=w)
    for w in range(N_WIRES - 1):
        qml.CNOT(wires=[w, w + 1])
    return qml.expval(qml.PauliZ(0) @ qml.PauliZ(N_WIRES - 1))


def test_parser_recognizes_toy_circuit_as_1d_local():
    dev = qml.device("default.qubit", wires=N_WIRES)
    qnode = qml.QNode(toy_circuit, dev)
    weights = np.zeros(N_WIRES)

    topology = parse_qnode(qnode, weights)

    assert topology.is_1d_local
    assert topology.max_entangling_span == 1
    assert topology.wire_order == list(range(N_WIRES))
    assert len(topology.entangling_pairs) == N_WIRES - 1


def test_chi1_surrogate_loses_fidelity_but_chi2_matches_exact():
    """Sanity check underlying the chi*=2 claim: chi=1 (product-state MPS) cannot
    reproduce the CNOT-ladder's entanglement, chi=2 can, chi=4 (already >= exact bond
    dimension for this depth) agrees with chi=2 to numerical precision.
    """
    rng = np.random.default_rng(0)
    weights = torch.tensor(rng.uniform(0, 2 * np.pi, size=N_WIRES))

    dev_exact = qml.device("default.qubit", wires=N_WIRES)
    exact_val = float(qml.QNode(toy_circuit, dev_exact)(weights))

    vals = {}
    for chi in (1, 2, 4):
        surrogate = MPSSurrogate(toy_circuit, N_WIRES, chi, weights)
        vals[chi] = float(surrogate(weights))

    assert abs(vals[1] - exact_val) > 1e-3, "chi=1 should NOT match the exact entangled state"
    assert abs(vals[2] - exact_val) < 1e-9, "chi=2 should already be numerically exact here"
    assert abs(vals[4] - vals[2]) < 1e-9, "chi=4 should agree with chi=2 (already saturated)"


def _toy_circuit_state(weights):
    """Same circuit as ``toy_circuit`` but returning the full statevector, so the sweep
    below can be trained against *state fidelity* rather than a single scalar
    expectation value. A single expectation value like ``<Z0 Z3>`` is a degenerate
    target for this purpose: it factorizes for a product state too (``<Z0><Z3>``), so a
    chi=1 (unentangled) surrogate can trivially hit almost any scalar target by choosing
    different rotation angles -- it would "pass" without ever reproducing the actual
    entangled state. Fidelity against the full state has no such loophole: it can only be
    driven to 1 by actually reproducing the entanglement structure, so it's the honest
    test of MPS *representability* that spec section 4.2's "accuracy vs chi" curve is
    supposed to be measuring.
    """
    for w in range(N_WIRES):
        qml.RY(weights[w], wires=w)
    for w in range(N_WIRES - 1):
        qml.CNOT(wires=[w, w + 1])
    return qml.state()


def test_sweep_recovers_chi_star_equal_2():
    """The full ``sweep_bond_dimension`` pipeline, exercised against a state-fidelity
    metric, confirms chi* = 2 -- the spec's required E1 gate ("If it doesn't, the
    matching layer has a bug -- fix before touching real models").

    Note on epochs=0 here: PennyLane's ``default.tensor`` currently cannot differentiate
    a ``qml.state()`` return with *any* diff_method (parameter-shift explicitly refuses
    state returns as "not hardware-compatible"; backprop/adjoint both raise
    "does not support ... with requested circuit" for this device) -- confirmed
    empirically while writing this test, and logged in LIMITATIONS.md. So this test
    checks *representability* (does the sweep pipeline correctly identify which chi
    reproduces a fixed, known-entangled reference state) rather than *trainability*
    against a state-fidelity objective, which is exactly what E1 (spec section 5) asks
    for: confirm the matching layer recovers chi*=2 on an analytically-known circuit.
    Trainability against a differentiable objective is validated separately in
    ``scripts/week1_spike.py`` and will apply to real target models, which train against
    ``qml.expval`` (differentiable) rather than raw state fidelity.
    """
    reference_weights = torch.tensor([0.9, 1.7, 2.3, 0.4])

    dev_exact = qml.device("default.qubit", wires=N_WIRES)
    reference_state = qml.QNode(_toy_circuit_state, dev_exact, interface="torch")(
        reference_weights
    ).detach()
    reference_metric = 1.0  # perfect fidelity is the target a low-enough chi should hit

    def fidelity(state_a: torch.Tensor, state_b: torch.Tensor) -> torch.Tensor:
        overlap = torch.vdot(state_a, state_b)
        return (overlap.conj() * overlap).real

    def make_loss_fn(surrogate: MPSSurrogate):
        # Never actually called (epochs=0 below) -- present only because
        # sweep_bond_dimension's interface requires one; see docstring above.
        def loss_fn(weights):
            return 1.0 - fidelity(reference_state, surrogate(weights))

        return loss_fn

    def metric_fn(surrogate: MPSSurrogate, trained_weights: torch.Tensor) -> float:
        pred_state = surrogate(trained_weights)
        return float(fidelity(reference_state, pred_state))

    result = sweep_bond_dimension(
        _toy_circuit_state,
        N_WIRES,
        reference_weights,  # epochs=0, so "training" is a no-op: this is the eval point
        make_loss_fn,
        metric_fn,
        reference_metric,
        chi_values=[1, 2, 4, 8],
        tolerance=1e-6,
        epochs=0,
        example_args=(reference_weights,),
    )

    assert result.chi_star == 2, result.summary()
