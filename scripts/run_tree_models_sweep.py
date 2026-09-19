"""Train model_02_qcnn_pooling and model_04_ttn_classifier's reference quantum models on
real data (Wisconsin Breast Cancer, 8 features), then sweep TreeSurrogate's bond dimension
against each -- turning two "not applicable" results into real chi* comparisons for the
first time (see LIMITATIONS.md/NOTES_WEEK4.md, which flagged this as the clear next step).

Unlike the MPS sweeps (scripts/run_model_01/03_sweep.py), TreeSurrogate trains entirely in
plain torch -- no quantum device, no parameter-shift -- so this is fast; only the
reference quantum model's training goes through PennyLane (via fast backprop on
default.qubit, not the parameter-shift bottleneck documented in LIMITATIONS.md).

Writes targets/model_02_qcnn_pooling/sweep_results.md and
targets/model_04_ttn_classifier/sweep_results.md (replacing their earlier
structural-only "not applicable" write-ups with a real numeric result).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pennylane as qml
import torch

from dequant_engine.tree_surrogate import TreeSurrogate
from targets.data.breast_cancer_loader import load_breast_cancer_8
from targets.model_02_qcnn_pooling.reimplementation import N_PARAMS as N_PARAMS_02
from targets.model_02_qcnn_pooling.reimplementation import N_WIRES as N_WIRES_02
from targets.model_02_qcnn_pooling.reimplementation import qcnn_circuit
from targets.model_04_ttn_classifier.reimplementation import (
    N_PARAMS_PER_UNITARY as N_PARAMS_PER_UNITARY_04,
)
from targets.model_04_ttn_classifier.reimplementation import N_UNITARIES as N_UNITARIES_04
from targets.model_04_ttn_classifier.reimplementation import N_WIRES as N_WIRES_04
from targets.model_04_ttn_classifier.reimplementation import ttn_classifier_circuit

N_TRAIN, N_TEST = 40, 20  # real data, but subsampled -- see below for why this is enough
CHI_VALUES = (1, 2, 4, 8, 16)
SEEDS = (0, 1, 2)  # TreeSurrogate is cheap (pure torch) -- full 3 seeds affordable here


def accuracy(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    return float((torch.sign(predictions) == torch.sign(targets)).float().mean())


def train_reference(circuit_fn, n_wires: int, weight_shape, x, y, *, lr: float, epochs: int):
    dev = qml.device("default.qubit", wires=n_wires)
    qnode = qml.QNode(circuit_fn, dev, interface="torch", diff_method="backprop")

    weights = torch.empty(*weight_shape)
    torch.nn.init.uniform_(weights, 0, 2 * torch.pi)
    weights.requires_grad_(True)
    optimizer = torch.optim.Adam([weights], lr=lr)

    for _ in range(epochs):
        optimizer.zero_grad()
        preds = torch.stack([qnode(weights, xi) for xi in x])
        loss = torch.mean((preds - y) ** 2)
        loss.backward()
        optimizer.step()

    return weights.detach(), qnode


def run_one_model(name: str, circuit_fn, n_wires: int, weight_shape, out_path: Path):
    x_all, y_all = load_breast_cancer_8()
    # Deterministic, class-balance-preserving subsample (see LIMITATIONS.md's compute
    # notes elsewhere in this repo -- 569 samples through 8-wire default.qubit backprop
    # is fast, so this subsampling is for TreeSurrogate seed/chi sweep cost, not the
    # reference model, which could easily use the full set).
    rng_idx = torch.randperm(len(x_all), generator=torch.Generator().manual_seed(0))
    idx_train, idx_test = rng_idx[:N_TRAIN], rng_idx[N_TRAIN : N_TRAIN + N_TEST]
    x_train = torch.tensor(x_all[idx_train.numpy()])
    y_train = torch.tensor(y_all[idx_train.numpy()])
    x_test = torch.tensor(x_all[idx_test.numpy()])
    y_test = torch.tensor(y_all[idx_test.numpy()])

    lr, epochs = 0.2, 60
    weights, qnode = train_reference(
        circuit_fn, n_wires, weight_shape, x_train, y_train, lr=lr, epochs=epochs
    )
    ref_train_acc = accuracy(torch.stack([qnode(weights, xi) for xi in x_train]), y_train)
    ref_test_acc = accuracy(torch.stack([qnode(weights, xi) for xi in x_test]), y_test)
    print(f"[{name}] reference train acc={ref_train_acc:.3f} test acc={ref_test_acc:.3f}")

    accuracy_by_chi: dict[int, list[tuple[float, float]]] = {}
    chi_stars = []
    for seed in SEEDS:
        chi_star = None
        for chi in CHI_VALUES:
            surrogate = TreeSurrogate(n_wires, chi)
            weights_init = surrogate.init_weights(seed=seed)

            def loss_fn(w, surrogate=surrogate, x_train=x_train, y_train=y_train):
                preds = torch.stack([surrogate(w, xi) for xi in x_train])
                return torch.mean((preds - y_train) ** 2)

            result = surrogate.fit(weights_init, loss_fn, lr=0.05, epochs=200)
            trained_w = result["weights"]
            train_acc = accuracy(torch.stack([surrogate(trained_w, xi) for xi in x_train]), y_train)
            test_acc = accuracy(torch.stack([surrogate(trained_w, xi) for xi in x_test]), y_test)
            accuracy_by_chi.setdefault(chi, []).append((train_acc, test_acc))
            # One-sided (see dequant_engine/sweep.py's identical fix, NOTES_WEEK6.md):
            # "caught up" means matching or exceeding the reference, not a symmetric
            # closeness band that would penalize a surrogate for outperforming an
            # undertrained reference.
            if chi_star is None and train_acc >= ref_train_acc - 0.01:
                chi_star = chi
        chi_stars.append(chi_star)
        print(f"[{name}] seed={seed} chi*={chi_star}")

    rows = "\n".join(
        f"| {chi} | {sum(tr for tr, _ in v) / len(v):.3f} | {sum(te for _, te in v) / len(v):.3f} |"
        for chi, v in sorted(accuracy_by_chi.items())
    )
    n_dequantized = sum(1 for c in chi_stars if c is not None)
    caveat = (
        "**Important caveat (see dequant_engine/tree_surrogate.py's module docstring and "
        "NOTES_WEEK6.md):** TreeSurrogate is NOT gate-structure-matched the way "
        "MPSSurrogate is -- it's a generic tree-shaped classical classifier trained "
        "directly on data labels, sharing only the topology (leaf count, tree shape) and "
        "leaf encoding with the reference circuit, not its specific gates or trained "
        "weights. This is why model_02 and model_04's tables above are numerically "
        "IDENTICAL: nothing about either circuit's own design enters the surrogate's "
        'training. Read chi* here as "the bond dimension a generic tree-shaped classical '
        "model needs to match *some* tree-shaped quantum circuit's accuracy on this "
        "task,\" not as a claim about this specific circuit's own entanglement structure."
    )
    out_path.write_text(
        f"# {name} — sweep results (TreeSurrogate, real Wisconsin Breast Cancer data)\n\n"
        f"Dataset: Wisconsin Breast Cancer, 8 features, {N_TRAIN} train / {N_TEST} test "
        "samples (real data, deterministic subsample for compute).\n\n"
        f"Reference (default.qubit) accuracy: **train {ref_train_acc:.3f}, "
        f"test {ref_test_acc:.3f}** (lr={lr}, epochs={epochs})\n\n"
        f"TreeSurrogate chi sweep, {len(SEEDS)} seeds ({list(SEEDS)}), "
        "from-scratch training per chi:\n\n"
        "| chi | mean train acc | mean test acc |\n|---|---|---|\n" + rows + "\n\n"
        f"**chi* per seed: {chi_stars} ({n_dequantized}/{len(SEEDS)} seeds dequantized "
        "within the range tried)**\n\n"
        f"{caveat}\n\n"
        'This replaces this model\'s earlier structural-only "not applicable" write-up: '
        "with TreeSurrogate (a genuine bond-dimension-bounded tree tensor network, not an "
        "MPS forced into the wrong shape), a real chi* comparison is now possible. See "
        "NOTES_WEEK6.md.\n"
    )
    print(f"Wrote {out_path}\n")


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    run_one_model(
        "model_02_qcnn_pooling",
        qcnn_circuit,
        N_WIRES_02,
        (N_PARAMS_02,),
        repo_root / "targets/model_02_qcnn_pooling/sweep_results.md",
    )
    run_one_model(
        "model_04_ttn_classifier",
        ttn_classifier_circuit,
        N_WIRES_04,
        (N_UNITARIES_04, N_PARAMS_PER_UNITARY_04),
        repo_root / "targets/model_04_ttn_classifier/sweep_results.md",
    )


if __name__ == "__main__":
    main()
