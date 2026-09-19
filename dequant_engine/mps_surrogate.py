"""Torch-trainable MPS surrogate for a PennyLane QNode (spec section 4.1/4.2).

Per NOTES_WEEK1.md + NOTES_WEEK2.md: for circuits ``parser.parse_qnode`` judges
path-decomposable (a linear wire order exists that makes every entangling gate act on
adjacent wires), ``default.tensor`` at a bounded ``max_bond_dim`` already *is* the matched
classical surrogate. The relabeling ``parser`` proposes is applied with ``qml.map_wires``
before construction -- this is not cosmetic: ``default.tensor``'s bond truncation is keyed
to numeric wire-label order regardless of what order is passed to the device's ``wires=``
argument (confirmed empirically, see parser.py's module docstring), so skipping this step
for a circuit that needs reordering would silently report an inflated, unfair chi*.
Training is byte-for-byte the same protocol (same loss, optimizer, schedule) as whatever
trained the original quantum model -- only the wire *labels* differ, never the gate
sequence or parameter structure.

For circuits the parser flags as genuinely non-path-decomposable (a wire entangled with 3+
distinct partners, or an entangling cycle), this class raises ``NotImplementedError`` with
a message pointing at ``parser.FAILURE_MODES`` -- building an explicit mid-circuit SWAP
network is out of scope for this session (see the repo's plan / timeline: that's later-week
work once a real target model surfaces the actual need).
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
            if self.topology.ring_wraparound_edges:
                raise NotImplementedError(
                    "MPSSurrogate currently only supports path-decomposable circuits. "
                    f"This circuit has {len(self.topology.ring_wraparound_edges)} ring "
                    f"component(s) (wraparound edges {self.topology.ring_wraparound_edges}) "
                    "-- every edge but one per ring can be made adjacent by reordering, but "
                    "the wraparound edge remains genuinely long-range. A common real-world "
                    "case (e.g. qml.BasicEntanglerLayers' default entangler) -- see "
                    "dequant_engine.parser.FAILURE_MODES item 1a. A general non-local "
                    "surrogate (explicit mid-circuit SWAP network) is not yet implemented."
                )
            raise NotImplementedError(
                "MPSSurrogate currently only supports circuits parser.parse_qnode judges "
                "path-decomposable (some wire relabeling makes every entangling gate act "
                "on adjacent wires). This circuit's entanglement graph is not a disjoint "
                "union of simple paths (a wire touched by 3+ distinct entangling partners) "
                f"-- best-effort achievable span under the greedy ordering is "
                f"{self.topology.achievable_span}. A general non-local surrogate (explicit "
                "mid-circuit SWAP network) is not yet implemented -- see "
                "dequant_engine.parser.FAILURE_MODES."
            )

        # Apply the proposed relabeling generically -- see module docstring for why this
        # is required for correctness of the reported chi*, not just an optimization.
        wire_map = {logical: position for position, logical in enumerate(self.topology.wire_order)}
        mapped_qfunc = qml.map_wires(qfunc, wire_map) if self.topology.needs_reordering else qfunc

        device_kwargs = {"method": "mps", "max_bond_dim": chi}
        if cutoff is not None:
            device_kwargs["cutoff"] = cutoff
        self.device = qml.device("default.tensor", wires=n_wires, **device_kwargs)
        self.qnode = qml.QNode(
            mapped_qfunc, self.device, interface="torch", diff_method=diff_method
        )

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
