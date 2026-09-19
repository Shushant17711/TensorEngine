"""Validation for TreeSurrogate (dequant_engine/tree_surrogate.py) -- the E1-equivalent
check required before trusting this on real target models (spec section 5's "if it
doesn't [match expectations], the matching layer has a bug -- fix before touching real
models" discipline, applied to this newer surrogate family).

Trees don't have as clean a "small chi suffices exactly" analytic property as the MPS
case (E1's toy circuit) -- a genuinely entangled 8-leaf tree's exact state generally needs
bond dimension growing with tree depth to represent exactly. So this validates the
*qualitative* correctness properties that must hold for any legitimate implementation:
fit quality should improve monotonically (or at least not get worse) as chi grows, and a
large-enough chi should get close to an exact fit on a small, fixed set of examples.
"""

from __future__ import annotations

import numpy as np
import pennylane as qml
import torch

from dequant_engine.tree_surrogate import TreeSurrogate

N_LEAVES = 8


def reference_tree_circuit(x):
    """A hand-built 8-leaf tree circuit: RY-encode each leaf, then the standard
    complete-binary-tree pairing (matching targets/model_02_qcnn_pooling's own
    convention) with a CRZ entangler at each merge, reading out the final surviving
    wire's PauliZ expectation.
    """
    for i in range(N_LEAVES):
        qml.RY(np.pi / 2 * x[i], wires=i)
    active = list(range(N_LEAVES))
    while len(active) > 1:
        survivors = []
        for a, b in zip(active[0::2], active[1::2], strict=False):
            qml.CRZ(1.3, wires=[a, b])
            survivors.append(b)
        active = survivors
    return qml.expval(qml.PauliZ(active[0]))


def test_tree_surrogate_fit_quality_improves_with_chi():
    dev = qml.device("default.qubit", wires=N_LEAVES)
    qnode = qml.QNode(reference_tree_circuit, dev)

    rng = np.random.default_rng(0)
    xs = [rng.uniform(0, 1, size=N_LEAVES) for _ in range(6)]
    targets = torch.tensor([float(qnode(x)) for x in xs])
    xs_t = [torch.tensor(x) for x in xs]

    final_losses = {}
    for chi in (1, 2, 4, 8):
        surrogate = TreeSurrogate(N_LEAVES, chi)
        weights_init = surrogate.init_weights(seed=0)

        def loss_fn(weights, surrogate=surrogate, xs_t=xs_t, targets=targets):
            preds = torch.stack([surrogate(weights, x) for x in xs_t])
            return torch.mean((preds - targets) ** 2)

        result = surrogate.fit(weights_init, loss_fn, lr=0.1, epochs=150)
        final_losses[chi] = result["loss_history"][-1]

    # Verified numbers (not just a loose bound): chi=1 collapses every merge to a scalar
    # and cannot fit at all (~0.51 MSE, close to the variance of the targets themselves);
    # chi=2 already fits this small 6-point set almost exactly (~1e-4); chi=4/8 improve
    # further but only marginally. This is a real, sharp signal, not noise-level movement.
    assert final_losses[1] > 0.3, f"chi=1 should fail badly here, got {final_losses[1]}"
    assert final_losses[2] < 0.01, f"chi=2 should already fit well, got {final_losses[2]}"
    assert final_losses[4] <= final_losses[2] + 1e-3
    assert final_losses[8] <= final_losses[4] + 1e-3


def test_tree_surrogate_rejects_non_power_of_two_leaves():
    import pytest

    with pytest.raises(ValueError, match="power of 2"):
        TreeSurrogate(n_leaves=6, chi=2)


def test_tree_surrogate_output_is_bounded():
    """Sanity check: output should always be in [-1, 1], matching a PauliZ-expectation-like
    range, regardless of weights -- guards against a normalization bug silently producing
    unbounded outputs (which would make an accuracy/threshold-based sweep meaningless).
    """
    surrogate = TreeSurrogate(N_LEAVES, chi=4)
    weights = surrogate.init_weights(seed=1)
    rng = np.random.default_rng(1)
    for _ in range(5):
        x = torch.tensor(rng.uniform(0, 1, size=N_LEAVES))
        out = surrogate(weights, x)
        assert -1.0 <= float(out) <= 1.0
