"""Train model_03_mps_classifier's reference quantum model, then sweep its MPS surrogate's
bond dimension -- the second preregistered battery result (PREREGISTRATION.md), testing
the falsifiable prediction chi* ~= 2 (this circuit's own bond dimension V=1 implies a
Schmidt rank cap of 2^V=2 across any cut).

Writes targets/model_03_mps_classifier/sweep_results.md.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pennylane as qml
import torch

from dequant_engine.mps_surrogate import MPSSurrogate
from dequant_engine.sweep import sweep_bond_dimension
from targets.model_03_mps_classifier.reimplementation import (
    N_PARAMS_PER_UNITARY,
    N_UNITARIES,
    N_WIRES,
    mps_classifier_circuit,
)


def make_binary_dataset(n_samples: int, n_features: int, seed: int = 0):
    """A simple, separable-by-a-linear-combination binary task: label = sign of the sum
    of the first half of features minus the second half -- exercises real-valued feature
    encoding (unlike model_01's pure-bitstring parity task) without needing an external
    dataset download.
    """
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 1, size=(n_samples, n_features))
    half = n_features // 2
    y = (x[:, :half].sum(axis=1) > x[:, half:].sum(axis=1)).astype(float) * 2 - 1
    return x, y


def accuracy(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    return float((torch.sign(predictions) == torch.sign(targets)).float().mean())


def train_reference_model(x: torch.Tensor, y: torch.Tensor, *, lr: float, epochs: int) -> dict:
    dev = qml.device("default.qubit", wires=N_WIRES)
    qnode = qml.QNode(mps_classifier_circuit, dev, interface="torch", diff_method="backprop")

    weights = torch.empty((N_UNITARIES, N_PARAMS_PER_UNITARY))
    torch.nn.init.uniform_(weights, 0, 2 * torch.pi)
    weights.requires_grad_(True)
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


SEEDS = (0, 1)  # 2 seeds, not 3 -- see NOTES_WEEK3.md/LIMITATIONS.md compute constraint;
# still enough to check whether chi*=2 is a coincidence of one random init


def main() -> None:
    x_np, y_np = make_binary_dataset(n_samples=12, n_features=N_WIRES, seed=0)
    x = torch.tensor(x_np)
    y = torch.tensor(y_np)

    lr, epochs = 0.2, 50
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

    chi_values = [1, 2, 4]
    per_seed_results = []
    for seed in SEEDS:
        torch.manual_seed(seed)
        weights_init = torch.empty((N_UNITARIES, N_PARAMS_PER_UNITARY))
        torch.nn.init.uniform_(weights_init, 0, 2 * torch.pi)

        result = sweep_bond_dimension(
            mps_classifier_circuit,
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
        Path(__file__).resolve().parent.parent / "targets/model_03_mps_classifier/sweep_results.md"
    )
    rows = "\n".join(
        f"| {chi} | {sum(vals) / len(vals):.3f} | {min(vals):.3f}–{max(vals):.3f} |"
        for chi, vals in accuracy_by_chi.items()
    )
    chi_star_summary = (
        f"chi* per seed: {[r.chi_star for r in per_seed_results]} "
        f"({len(chi_stars)}/{len(SEEDS)} seeds dequantized within the range tried)"
    )
    prediction_note = (
        "**Prediction check (PREREGISTRATION.md model 3): chi\\* ≈ 2 expected "
        "(circuit's own bond dimension V=1 => Schmidt rank cap 2^V=2).**"
    )
    out_path.write_text(
        "# model_03_mps_classifier — sweep results\n\n"
        f"Reference (default.qubit) accuracy: **{ref['accuracy']:.3f}** "
        f"(final training loss {ref['loss_history'][-1]:.4f}, lr={lr}, epochs={epochs})\n\n"
        f"{prediction_note}\n\n"
        f"chi sweep, {len(SEEDS)} seeds ({list(SEEDS)}), from-scratch training per chi:\n\n"
        "| chi | mean accuracy | range |\n|---|---|---|\n" + rows + f"\n\n**{chi_star_summary}**\n"
    )
    print(f"\n{chi_star_summary}")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
