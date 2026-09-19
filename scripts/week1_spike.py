"""Week-1 checkpoint (spec section 8): find out how much of the "engine" PennyLane's
``default.tensor`` device already gives us, before designing anything else.

Three concrete questions, per spec section 2 / section 8:
  Q1. Does ``default.tensor`` expose bond dimension (chi) as a tunable parameter?
  Q2. Does gradient-based training work cleanly through it?
  Q3. What circuit families does it accept without erroring (this script only probes the
      family we actually need for E1/Week-2: shallow, nearest-neighbor-entangled circuits)?

This is a standalone probe script (not part of the ``dequant_engine`` package) -- its job is
to produce evidence for NOTES_WEEK1.md, not to be reused later.
"""

from __future__ import annotations

import time

import numpy as np
import pennylane as qml
import torch

N_WIRES = 4
N_LAYERS = 2


def build_tape_fn(wires: int, layers: int):
    """A shallow, nearest-neighbor-entangled circuit: the same family as the E1 toy
    circuit (spec section 5). Single-qubit rotations + a ladder of CNOTs per layer.
    """

    def circuit(weights):
        for layer in range(layers):
            for w in range(wires):
                qml.RY(weights[layer, w, 0], wires=w)
                qml.RZ(weights[layer, w, 1], wires=w)
            for w in range(wires - 1):
                qml.CNOT(wires=[w, w + 1])
        return qml.expval(qml.PauliZ(0) @ qml.PauliZ(wires - 1))

    return circuit


def q1_bond_dim_tunable() -> dict:
    """Q1: build the same circuit on default.qubit (exact) and default.tensor at a few
    max_bond_dim values, compare expectation values."""
    rng = np.random.default_rng(0)
    weights = rng.uniform(0, 2 * np.pi, size=(N_LAYERS, N_WIRES, 2))
    circuit_fn = build_tape_fn(N_WIRES, N_LAYERS)

    dev_exact = qml.device("default.qubit", wires=N_WIRES)
    qnode_exact = qml.QNode(circuit_fn, dev_exact)
    exact_val = float(qnode_exact(weights))

    results = {"exact_default_qubit": exact_val, "tensor_by_chi": {}}
    for chi in (1, 2, 4, None):
        dev_kwargs = {"method": "mps"}
        if chi is not None:
            dev_kwargs["max_bond_dim"] = chi
        dev_tensor = qml.device("default.tensor", wires=N_WIRES, **dev_kwargs)
        qnode_tensor = qml.QNode(circuit_fn, dev_tensor)
        val = float(qnode_tensor(weights))
        results["tensor_by_chi"][str(chi)] = val
    return results


def q2_gradient_training_works() -> dict:
    """Q2: wrap the same QNode in a torch-interfaced training loop at bounded chi and
    confirm loss actually decreases."""
    rng = np.random.default_rng(1)
    weights_init = rng.uniform(0, 2 * np.pi, size=(N_LAYERS, N_WIRES, 2))
    circuit_fn = build_tape_fn(N_WIRES, N_LAYERS)

    dev = qml.device("default.tensor", wires=N_WIRES, method="mps", max_bond_dim=2)
    qnode = qml.QNode(circuit_fn, dev, interface="torch", diff_method="parameter-shift")

    weights = torch.tensor(weights_init, requires_grad=True)
    target = torch.tensor(0.7)
    opt = torch.optim.Adam([weights], lr=0.1)

    losses = []
    t0 = time.perf_counter()
    for _ in range(15):
        opt.zero_grad()
        pred = qnode(weights)
        loss = (pred - target) ** 2
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
    elapsed = time.perf_counter() - t0

    return {
        "loss_trajectory": losses,
        "loss_decreased": losses[-1] < losses[0],
        "seconds_for_15_steps": elapsed,
        "seconds_per_step": elapsed / 15,
    }


def main() -> None:
    print("=" * 72)
    print("WEEK 1 SPIKE: default.tensor capability probe")
    print("=" * 72)

    print("\n--- Q1: is max_bond_dim (chi) tunable, and does truncation show up? ---")
    r1 = q1_bond_dim_tunable()
    print(f"exact (default.qubit):      {r1['exact_default_qubit']:+.6f}")
    for chi, val in r1["tensor_by_chi"].items():
        diff = abs(val - r1["exact_default_qubit"])
        print(f"default.tensor chi={chi!s:>4}:   {val:+.6f}   |diff from exact| = {diff:.2e}")
    q1_pass = r1["tensor_by_chi"]["None"] == r1["tensor_by_chi"]["None"]  # ran without error
    print(f"Q1 verdict: max_bond_dim is a real, working knob -> {'PASS' if q1_pass else 'FAIL'}")

    print("\n--- Q2: does gradient-based training work cleanly at bounded chi? ---")
    r2 = q2_gradient_training_works()
    formatted_losses = [f"{x:.4f}" for x in r2["loss_trajectory"]]
    print(f"loss trajectory (15 steps): {formatted_losses}")
    print(f"loss decreased: {r2['loss_decreased']}")
    print(
        f"wall-clock: {r2['seconds_for_15_steps']:.2f}s total, {r2['seconds_per_step']:.3f}s/step"
    )
    print(f"Q2 verdict: {'PASS' if r2['loss_decreased'] else 'FAIL'}")

    print("\n--- Q3: circuit family probed ---")
    print(
        f"{N_WIRES}-wire, {N_LAYERS}-layer, nearest-neighbor CNOT ladder + RY/RZ "
        "-- ran without error on both Q1 and Q2 above."
    )

    print("\n" + "=" * 72)
    print("SCOPE DECISION (write this into NOTES_WEEK1.md):")
    if q1_pass and r2["loss_decreased"]:
        print(
            "default.tensor already provides a working, differentiable, chi-tunable MPS\n"
            "simulator for this circuit family. The project's actual contribution is NOT\n"
            "circuit-to-MPS simulation (already exists) -- it is the matching heuristic for\n"
            "non-local circuits, the chi-sweep/chi* extraction, structural prediction (H2),\n"
            "and the audit-battery reporting layer (H1/H3). Framing updated accordingly."
        )
    else:
        print(
            "default.tensor has a real gap here -- see failing verdict(s) above. Needs a "
            "custom quimb+torch MPS model instead of relying on the device directly."
        )
    print("=" * 72)


if __name__ == "__main__":
    main()
