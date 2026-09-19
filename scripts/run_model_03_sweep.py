"""Train model_03_mps_classifier's reference quantum model, then sweep its MPS surrogate's
bond dimension -- the second preregistered battery result (PREREGISTRATION.md), testing
the falsifiable prediction chi* ~= 2 (this circuit's own bond dimension V=1 implies a
Schmidt rank cap of 2^V=2 across any cut).

Uses the REAL Iris dataset (versicolor vs virginica), replacing the earlier invented
synthetic task -- see NOTES_WEEK5.md for why that was a real gap, not a stylistic choice.
Reports both train and held-out test accuracy (the earlier version only checked training
accuracy, a weaker standard than this session's own stated reproduction protocol called
for).

Writes targets/model_03_mps_classifier/sweep_results.md.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pennylane as qml
import torch

from dequant_engine.mps_surrogate import MPSSurrogate
from dequant_engine.sweep import sweep_bond_dimension
from targets.data.iris_loader import load_iris_binary, train_test_split
from targets.model_03_mps_classifier.reimplementation import (
    N_PARAMS_PER_UNITARY,
    N_UNITARIES,
    N_WIRES,
    mps_classifier_circuit,
)

# Compute-budget note (see LIMITATIONS.md): the full 100-sample versicolor/virginica set
# would multiply per-epoch cost ~8x over the earlier synthetic run's 12 samples, at
# roughly 2.7s/epoch that's well over an hour for the full seed/chi protocol. Subsampled
# to a smaller, still-real, still class-balanced train/test split -- a compute-driven
# choice, recorded here rather than silently using fewer samples without saying so.
N_TRAIN, N_TEST = 16, 10


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

    return {"weights": weights.detach(), "loss_history": history, "qnode": qnode}


SEEDS = (0, 1)


def main() -> None:
    x_all, y_all = load_iris_binary(class_a=1, class_b=2)  # versicolor vs virginica
    x_train_np, y_train_np, x_test_np, y_test_np = train_test_split(
        x_all, y_all, test_frac=0.4, seed=0
    )
    # Subsample for compute (see note above), keeping the split's own class balance.
    x_train_np, y_train_np = x_train_np[:N_TRAIN], y_train_np[:N_TRAIN]
    x_test_np, y_test_np = x_test_np[:N_TEST], y_test_np[:N_TEST]
    x_train, y_train = torch.tensor(x_train_np), torch.tensor(y_train_np)
    x_test, y_test = torch.tensor(x_test_np), torch.tensor(y_test_np)
    print(
        f"train n={len(x_train)} (class balance {(y_train_np == 1).sum()}/{len(y_train_np)}), "
        f"test n={len(x_test)} (class balance {(y_test_np == 1).sum()}/{len(y_test_np)})"
    )

    lr, epochs = 0.2, 50
    ref = train_reference_model(x_train, y_train, lr=lr, epochs=epochs)
    ref_qnode = ref["qnode"]
    ref_train_acc = accuracy(
        torch.stack([ref_qnode(ref["weights"], xi) for xi in x_train]), y_train
    )
    ref_test_acc = accuracy(torch.stack([ref_qnode(ref["weights"], xi) for xi in x_test]), y_test)
    print(
        f"Reference (default.qubit) train accuracy: {ref_train_acc:.3f}, test: {ref_test_acc:.3f}"
    )
    print(f"Final training loss: {ref['loss_history'][-1]:.4f}")

    def make_loss_fn(surrogate: MPSSurrogate):
        def loss_fn(weights):
            preds = torch.stack([surrogate(weights, xi) for xi in x_train])
            return torch.mean((preds - y_train) ** 2)

        return loss_fn

    def metric_fn(surrogate: MPSSurrogate, trained_weights: torch.Tensor) -> tuple[float, float]:
        train_preds = torch.stack([surrogate(trained_weights, xi) for xi in x_train])
        test_preds = torch.stack([surrogate(trained_weights, xi) for xi in x_test])
        return accuracy(train_preds, y_train), accuracy(test_preds, y_test)

    # sweep_bond_dimension's metric_fn contract returns a single float; wrap to satisfy
    # that while still capturing both numbers for the report.
    train_test_by_chi: dict[int, list[tuple[float, float]]] = {}

    def metric_fn_train_only(surrogate: MPSSurrogate, trained_weights: torch.Tensor) -> float:
        tr, te = metric_fn(surrogate, trained_weights)
        train_test_by_chi.setdefault(surrogate.chi, []).append((tr, te))
        return tr

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
            metric_fn_train_only,
            reference_metric=ref_train_acc,
            chi_values=chi_values,
            tolerance=0.01,
            lr=lr,
            epochs=epochs,
            example_args=(weights_init, x_train_np[0]),
        )
        per_seed_results.append(result)
        print(f"seed={seed}: {result.summary()}")

    chi_stars = [r.chi_star for r in per_seed_results if r.chi_star is not None]

    out_path = (
        Path(__file__).resolve().parent.parent / "targets/model_03_mps_classifier/sweep_results.md"
    )
    rows = "\n".join(
        f"| {chi} | {sum(tr for tr, _ in vals) / len(vals):.3f} | "
        f"{sum(te for _, te in vals) / len(vals):.3f} |"
        for chi, vals in sorted(train_test_by_chi.items())
    )
    chi_star_summary = (
        f"chi* per seed (by train accuracy, tolerance 0.01): {[r.chi_star for r in per_seed_results]} "
        f"({len(chi_stars)}/{len(SEEDS)} seeds dequantized within the range tried)"
    )
    out_path.write_text(
        "# model_03_mps_classifier — sweep results (real Iris data)\n\n"
        f"Dataset: Iris, versicolor vs virginica, {N_TRAIN} train / {N_TEST} test samples "
        "(subsampled from the full 100 for compute — see scripts/run_model_03_sweep.py).\n\n"
        f"Reference (default.qubit) accuracy: **train {ref_train_acc:.3f}, "
        f"test {ref_test_acc:.3f}** (final training loss {ref['loss_history'][-1]:.4f}, "
        f"lr={lr}, epochs={epochs})\n\n"
        "**Prediction check (PREREGISTRATION.md model 3): chi\\* ≈ 2 expected "
        "(circuit's own bond dimension V=1 => Schmidt rank cap 2^V=2).**\n\n"
        f"chi sweep, {len(SEEDS)} seeds ({list(SEEDS)}), from-scratch training per chi:\n\n"
        "| chi | mean train acc | mean test acc |\n|---|---|---|\n"
        + rows
        + f"\n\n**{chi_star_summary}**\n"
    )
    print(f"\n{chi_star_summary}")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
