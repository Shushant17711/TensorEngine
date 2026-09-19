"""Bond-dimension sweep + chi* extraction (spec section 4.2).

``default.tensor`` gives you one bond dimension at a time; nothing sweeps it and reports a
threshold. That's this module's entire job, and per the Week 1 scope decision
(NOTES_WEEK1.md) it's one of the parts of the spec that's a genuine, non-duplicated
contribution.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import torch

from dequant_engine.mps_surrogate import MPSSurrogate


@dataclass
class SweepResult:
    """Per-model sweep outcome: accuracy/metric at each chi, plus the extracted chi*."""

    chi_values: list[int]
    metric_at_chi: dict[int, float]
    reference_metric: float
    tolerance: float
    chi_star: int | None  # None means "not dequantized" within the sweep range tried
    loss_histories: dict[int, list[float]] = field(default_factory=dict)

    @property
    def dequantized(self) -> bool:
        return self.chi_star is not None

    def summary(self) -> str:
        rows = "\n".join(
            f"  chi={chi:>3}: metric={self.metric_at_chi[chi]:.4f}" for chi in self.chi_values
        )
        verdict = (
            f"DEQUANTIZED at chi*={self.chi_star}"
            if self.dequantized
            else "NOT DEQUANTIZED within the sweep range tried"
        )
        return (
            f"reference metric: {self.reference_metric:.4f} (tolerance {self.tolerance})\n"
            f"{rows}\n{verdict}"
        )


def sweep_bond_dimension(
    qfunc: Callable,
    n_wires: int,
    weights_init: torch.Tensor,
    make_loss_fn: Callable[[MPSSurrogate], Callable[[torch.Tensor], torch.Tensor]],
    metric_fn: Callable[[MPSSurrogate, torch.Tensor], float],
    reference_metric: float,
    *,
    chi_values: list[int] = (1, 2, 4, 8, 16, 32, 64),
    tolerance: float = 0.01,
    lr: float = 0.1,
    epochs: int = 50,
    example_args: tuple = (),
    surrogate_kwargs: dict | None = None,
) -> SweepResult:
    """Train an ``MPSSurrogate`` of ``qfunc`` at each chi in ``chi_values``, evaluate
    ``metric_fn`` after training, and report the smallest chi within ``tolerance`` of
    ``reference_metric`` (spec section 4.2: "the smallest bond dimension within some
    tolerance ... of the quantum model's accuracy").

    ``make_loss_fn(surrogate) -> loss_fn`` and ``metric_fn(surrogate, trained_weights) ->
    float`` are caller-supplied so this function stays agnostic to what "accuracy" means
    for a given target model (classification accuracy, MSE, fidelity, ...).
    """
    metric_at_chi: dict[int, float] = {}
    loss_histories: dict[int, list[float]] = {}
    chi_star = None
    surrogate_kwargs = surrogate_kwargs or {}

    for chi in chi_values:
        surrogate = MPSSurrogate(qfunc, n_wires, chi, *example_args, **surrogate_kwargs)
        loss_fn = make_loss_fn(surrogate)
        fit_result = surrogate.fit(weights_init, loss_fn, lr=lr, epochs=epochs)

        metric = metric_fn(surrogate, fit_result["weights"])
        metric_at_chi[chi] = metric
        loss_histories[chi] = fit_result["loss_history"]

        # One-sided: "caught up" means matching OR EXCEEDING the reference, within a
        # small allowed shortfall below it -- not a symmetric closeness band. Found via
        # a real case (NOTES_WEEK6.md): an undertrained reference model can score lower
        # than a well-optimized surrogate, and a surrogate that's *better* than the
        # reference has obviously already dequantized it, not failed to match it.
        if chi_star is None and metric >= reference_metric - tolerance:
            chi_star = chi

    return SweepResult(
        chi_values=list(chi_values),
        metric_at_chi=metric_at_chi,
        reference_metric=reference_metric,
        tolerance=tolerance,
        chi_star=chi_star,
        loss_histories=loss_histories,
    )
