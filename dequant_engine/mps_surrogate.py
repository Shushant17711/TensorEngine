"""Torch-trainable MPS surrogate for a PennyLane QNode (spec section 4.1/4.2).

Per the Week 1 finding (NOTES_WEEK1.md): for circuits ``parser.parse_qnode`` judges
1D-local, ``default.tensor`` at a bounded ``max_bond_dim`` already *is* the matched
classical surrogate -- we just re-run the exact same quantum function on that device with
the torch interface and parameter-shift gradients, so training is byte-for-byte the same
protocol (same loss, optimizer, schedule) as whatever trained the original quantum model.

For circuits the parser flags as non-local, this class raises ``NotImplementedError`` with
a message pointing at ``parser.FAILURE_MODES`` -- building an explicit quimb MPS with SWAP
networks for arbitrary entanglement graphs is out of scope for this session (see the repo's
plan / timeline: that's later-week work once real target models surface the actual need).
"""

from __future__ import annotations

from collections.abc import Callable

import pennylane as qml
import torch

from dequant_engine.parser import CircuitTopology, parse_qnode


class MPSSurrogate:
    """A bond-dimension-bounded classical stand-in for a quantum QNode's function.

    Parameters
    ----------
    qfunc:
        The bare quantum function (the callable originally passed to ``qml.QNode``, *not*
        the QNode itself) -- e.g. ``def circuit(weights): ...; return qml.expval(...)``.
        Re-executed unchanged against the surrogate device.
    n_wires:
        Number of wires the circuit uses.
    chi:
        Bond dimension cap (``max_bond_dim`` on ``default.tensor``).
    example_args / example_kwargs:
        Used once, at construction, purely to trace the circuit's structure via
        ``parser.parse_qnode`` (values don't matter, only which operations get queued).
    """

    def __init__(
        self,
        qfunc: Callable,
        n_wires: int,
        chi: int,
        *example_args,
        cutoff: float | None = None,
        diff_method: str = "parameter-shift",
        **example_kwargs,
    ) -> None:
        self.qfunc = qfunc
        self.n_wires = n_wires
        self.chi = chi

        probe_dev = qml.device("default.qubit", wires=n_wires)
        probe_qnode = qml.QNode(qfunc, probe_dev)
        self.topology: CircuitTopology = parse_qnode(probe_qnode, *example_args, **example_kwargs)

        if not self.topology.is_1d_local:
            raise NotImplementedError(
                "MPSSurrogate currently only supports circuits parser.parse_qnode judges "
                "1D-local (default.tensor's native ordering handles those directly, per "
                "NOTES_WEEK1.md). This circuit has entangling gates spanning up to "
                f"{self.topology.max_entangling_span} wires under the natural ordering. "
                "A general non-local surrogate (explicit quimb MPS + SWAP network under "
                "the proposed reorder) is not yet implemented -- see "
                "dequant_engine.parser.FAILURE_MODES and the proposed ordering at "
                f"{self.topology.wire_order}."
            )

        device_kwargs = {"method": "mps", "max_bond_dim": chi}
        if cutoff is not None:
            device_kwargs["cutoff"] = cutoff
        self.device = qml.device("default.tensor", wires=n_wires, **device_kwargs)
        self.qnode = qml.QNode(qfunc, self.device, interface="torch", diff_method=diff_method)

    def __call__(self, *args, **kwargs):
        return self.qnode(*args, **kwargs)

    def fit(
        self,
        weights_init: torch.Tensor,
        loss_fn: Callable[[torch.Tensor], torch.Tensor],
        *,
        lr: float = 0.1,
        epochs: int = 50,
        optimizer_cls=torch.optim.Adam,
    ) -> dict:
        """Train the surrogate's weights under a caller-specified loss/optimizer/epoch
        budget -- deliberately a thin pass-through, not a hidden default protocol, so the
        caller can mirror exactly whatever trained the reference quantum model (spec
        section 4.1: "train under the SAME loss, optimizer, learning rate, and epoch
        budget as the original quantum model").

        ``loss_fn(weights) -> scalar tensor`` is fully caller-defined -- it typically
        calls ``self.qnode(weights, ...)`` internally (over one or more data points) and
        compares against target labels/values, but this class doesn't prescribe a data
        or batching convention, only the optimization loop around whatever loss the
        caller builds.
        """
        weights = weights_init.clone().detach().requires_grad_(True)
        optimizer = optimizer_cls([weights], lr=lr)

        history = []
        for _ in range(epochs):
            optimizer.zero_grad()
            loss = loss_fn(weights)
            loss.backward()
            optimizer.step()
            history.append(float(loss.detach()))

        return {"weights": weights.detach(), "loss_history": history}
