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


SEEDS = (0, 1, 2)  # spec section 5/E2: report chi* with a multi-seed confidence interval


def main() -> None:
    x_np, y_np = make_parity_dataset(n_samples=16, n_bits=N_WIRES, seed=0)
    x = torch.tensor(x_np)
    y = torch.tensor(y_np)

    lr, epochs = 0.2, 100
    ref = train_reference_model(x, y, lr=lr, epochs=epochs)
    print(f"Reference (default.qubit) accuracy after training: {ref['accuracy']:.3f}")
    print(f"Final training loss: {ref['loss_history'][-1]:.4f}")

    def make_loss_fn(surrogate: MPSSurrogate):
        def loss_fn(weights):
            preds = torch.stack([surrogate(weights, xi) for xi in x])
            return torch.mean((preds - y) ** 2)

        return loss_fn

    def metric_fn(surrogate: MPSSurrogate, trained_weights: torch.Tensor) -> float:
        preds = torch.stack([surrogate(trained_weights, xi) for xi in x])
        return accuracy(preds, y)

    # From-scratch training per chi, same epoch budget as the reference model, across
    # multiple seeds -- the E2 protocol (spec section 5), tightening the earlier
    # fine-tune-from-the-reference-point shortcut noted in NOTES_WEEK3.md.
    chi_values = [1, 2, 4]
    per_seed_results = []
    for seed in SEEDS:
        torch.manual_seed(seed)
        weights_init = torch.empty((N_LAYERS, N_WIRES))
        torch.nn.init.uniform_(weights_init, 0, 2 * torch.pi)

        result = sweep_bond_dimension(
            vqc_circuit,
            N_WIRES,
            weights_init,
            make_loss_fn,
            metric_fn,
            reference_metric=ref["accuracy"],
            chi_values=chi_values,
            tolerance=0.01,
            lr=lr,
            epochs=epochs,
            example_args=(weights_init, x_np[0]),
        )
        per_seed_results.append(result)
        print(f"seed={seed}: {result.summary()}")

    chi_stars = [r.chi_star for r in per_seed_results if r.chi_star is not None]
    accuracy_by_chi = {chi: [r.metric_at_chi[chi] for r in per_seed_results] for chi in chi_values}

    out_path = (
        Path(__file__).resolve().parent.parent / "targets/model_01_vqc_chain/sweep_results.md"
    )
    rows = "\n".join(
        f"| {chi} | {sum(vals) / len(vals):.3f} | {min(vals):.3f}–{max(vals):.3f} |"
        for chi, vals in accuracy_by_chi.items()
    )
    chi_star_summary = (
        f"chi* per seed: {[r.chi_star for r in per_seed_results]} "
        f"({len(chi_stars)}/{len(SEEDS)} seeds dequantized within the range tried)"
    )
    out_path.write_text(
        "# model_01_vqc_chain — sweep results\n\n"
        f"Reference (default.qubit) accuracy: **{ref['accuracy']:.3f}** "
        f"(final training loss {ref['loss_history'][-1]:.4f}, lr={lr}, epochs={epochs})\n\n"
        f"chi sweep, {len(SEEDS)} seeds ({list(SEEDS)}), from-scratch training per chi "
        "(same loss/optimizer/epoch budget as the reference model):\n\n"
        "| chi | mean accuracy | range |\n|---|---|---|\n" + rows + f"\n\n**{chi_star_summary}**\n"
    )
    print(f"\n{chi_star_summary}")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
