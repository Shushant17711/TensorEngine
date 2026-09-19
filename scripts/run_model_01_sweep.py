"""Train model_01_vqc_chain's reference quantum model, then sweep its MPS surrogate's
bond dimension and extract chi* -- the audit battery's first fully end-to-end run (spec
section 4.2/4.3, on a single model; the full preregistered battery is Week 3+ follow-up).

Writes targets/model_01_vqc_chain/sweep_results.md.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pennylane as qml
import torch

from dequant_engine.mps_surrogate import MPSSurrogate
from dequant_engine.sweep import sweep_bond_dimension
from targets.model_01_vqc_chain.reimplementation import (
    N_LAYERS,
    N_WIRES,
    make_parity_dataset,
    vqc_circuit,
)


def accuracy(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    return float((torch.sign(predictions) == torch.sign(targets)).float().mean())


def train_reference_model(x: torch.Tensor, y: torch.Tensor, *, lr: float, epochs: int) -> dict:
    """Same loss/optimizer/epoch discipline that will also be used for every surrogate --
    spec section 4.1's "train under the SAME loss, optimizer, learning rate, and epoch
    budget as the original quantum model" (this class instantiates that discipline for
    the reference side).
    """
    dev = qml.device("default.qubit", wires=N_WIRES)
    qnode = qml.QNode(vqc_circuit, dev, interface="torch", diff_method="backprop")

    weights = torch.zeros((N_LAYERS, N_WIRES), requires_grad=True)
    torch.nn.init.uniform_(weights, 0, 2 * torch.pi)
    optimizer = torch.optim.Adam([weights], lr=lr)

    history = []
    for _ in range(epochs):
        optimizer.zero_grad()
        preds = torch.stack([qnode(weights, xi) for xi in x])
        loss = torch.mean((preds - y) ** 2)
        loss.backward()
        optimizer.step()
        history.append(float(loss.detach()))

    final_preds = torch.stack([qnode(weights, xi) for xi in x])
    return {
        "weights": weights.detach(),
        "loss_history": history,
        "accuracy": accuracy(final_preds, y),
    }


def main() -> None:
    x_np, y_np = make_parity_dataset(n_samples=16, n_bits=N_WIRES, seed=0)
    x = torch.tensor(x_np)
    y = torch.tensor(y_np)

    lr, epochs = 0.2, 100
    ref = train_reference_model(x, y, lr=lr, epochs=epochs)
    print(f"Reference (default.qubit) accuracy after training: {ref['accuracy']:.3f}")
    print(f"Final training loss: {ref['loss_history'][-1]:.4f}")

    weights_init = ref["weights"].clone()  # start surrogates from the trained reference
    # point -- reasonable for this small-scale check; a from-scratch comparison (fresh
    # random init per chi) is the more rigorous E2 protocol and is Week 3+ follow-up.

    def make_loss_fn(surrogate: MPSSurrogate):
        def loss_fn(weights):
            preds = torch.stack([surrogate(weights, xi) for xi in x])
            return torch.mean((preds - y) ** 2)

        return loss_fn

    def metric_fn(surrogate: MPSSurrogate, trained_weights: torch.Tensor) -> float:
        preds = torch.stack([surrogate(trained_weights, xi) for xi in x])
        return accuracy(preds, y)

    result = sweep_bond_dimension(
        vqc_circuit,
        N_WIRES,
        weights_init,
        make_loss_fn,
        metric_fn,
        reference_metric=ref["accuracy"],
        chi_values=[1, 2, 4],
        tolerance=0.01,
        lr=lr,
        epochs=30,  # fine-tune from the reference point rather than retrain from scratch
        example_args=(weights_init, x_np[0]),
    )

    print(result.summary())

    out_path = (
        Path(__file__).resolve().parent.parent / "targets/model_01_vqc_chain/sweep_results.md"
    )
    out_path.write_text(
        "# model_01_vqc_chain — sweep results\n\n"
        f"Reference (default.qubit) accuracy: **{ref['accuracy']:.3f}** "
        f"(final training loss {ref['loss_history'][-1]:.4f}, lr={lr}, epochs={epochs})\n\n"
        "chi sweep (fine-tuned 30 epochs from the reference weights, same loss/optimizer):\n\n"
        "| chi | accuracy |\n|---|---|\n"
        + "\n".join(f"| {chi} | {result.metric_at_chi[chi]:.3f} |" for chi in result.chi_values)
        + f"\n\n**chi\\* = {result.chi_star}** "
        + ("(dequantized)" if result.dequantized else "(NOT dequantized within range tried)")
        + "\n"
    )
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
