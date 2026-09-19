"""Locks in a real correctness trap found while profiling scripts/run_model_01_sweep.py
(LIMITATIONS.md): passing a batch of data points to a single default.tensor QNode call
does NOT correctly vectorize over the batch -- it silently returns a wrong-shaped result
rather than raising an error. This test exists so nobody "optimizes" MPSSurrogate/sweep.py
into using batched calls without re-discovering this the hard way.
"""

from __future__ import annotations

import pennylane as qml
import torch


def circuit(weights, x):
    qml.RX(torch.pi * x[0], wires=0)
    qml.RX(torch.pi * x[1], wires=1)
    qml.RY(weights[0], wires=0)
    qml.RY(weights[1], wires=1)
    qml.CNOT(wires=[0, 1])
    return qml.expval(qml.PauliZ(1))


def test_batched_call_does_not_match_per_sample_loop():
    dev = qml.device("default.tensor", wires=2, method="mps", max_bond_dim=2)
    qnode = qml.QNode(circuit, dev, interface="torch", diff_method="best")

    weights = torch.tensor([0.3, 0.7])
    xs = torch.tensor(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 0.5]], dtype=torch.float64
    )

    per_sample = torch.stack([qnode(weights, xi) for xi in xs])
    assert per_sample.shape == (5,)

    batched = qnode(weights, xs)
    # This assertion documents the trap: batched output shape does not even match the
    # input batch size (5), confirming the batched call is not doing what it looks like
    # it should. If this ever starts passing (shape 5 and matching per_sample), the
    # underlying PennyLane/default.tensor behavior has changed -- revisit
    # LIMITATIONS.md and consider using batched calls for a real speedup.
    assert batched.shape != per_sample.shape or not torch.allclose(batched, per_sample)
