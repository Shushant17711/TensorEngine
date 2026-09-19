"""A small-scale QCNN (Cong, Choi, Lukin 2019 style: convolution + pooling reducing
8 wires -> 1) as a target model for the audit battery -- and, before that, as a stress
test of the matching layer (dequant_engine.parser).

Status: **not yet dequantization-tested.** ``parse_qnode`` on this circuit confirms its
entanglement graph is a genuine tree (8 nodes, 7 edges, connected, acyclic) -- each pooling
round's surviving wire accumulates one entangling partner per round it takes part in. This
is not a bug to fix; it's the well-known fact that QCNNs are naturally tree tensor networks
(TTNs), not matrix product states (see ``dequant_engine.parser`` module docstring point 4,
and NOTES_WEEK3.md). ``MPSSurrogate`` correctly refuses to build a surrogate for it with a
message pointing this out, rather than silently forcing a poor-fit path encoding.

This makes this target model an **"not applicable" outcome** in the audit battery's own
terms (spec section 1: "not applicable (surrogate family doesn't cover this circuit
class)"), not a failed reproduction and not evidence of anything about H1 -- and it belongs
in the preregistered battery specifically *because* it's expected to land there, per the
spec's instruction to include models "you expect to resist dequantization" (section 4.3)
so the battery isn't all one outcome. Building an MPS surrogate for it *would* be the wrong
thing to do; the honest path is a future TreeSurrogate (out of scope this session) or
excluding it from the MPS-only battery with this result recorded as the reason.
"""

from __future__ import annotations

import pennylane as qml

N_WIRES = 8


def conv_pool_round(active_wires: list[int], weights) -> list[int]:
    """One convolution+pooling round: apply a parameterized two-qubit entangler on each
    adjacent pair of currently-active wires, then keep only the second wire of each pair
    (the simplest possible pooling rule -- a controlled rotation is often used in the
    literature to make pooling itself contribute a trainable parameter; here we use CRZ so
    the pooling step is parameterized, matching that convention, and semantically discard
    the control wire from the active set without needing mid-circuit measurement).
    """
    pairs = list(zip(active_wires[0::2], active_wires[1::2], strict=False))
    for i, (control, target) in enumerate(pairs):
        qml.CRZ(weights[i], wires=[control, target])
    return [target for _, target in pairs]


def qcnn_circuit(weights):
    """``weights`` is a flat 1D array; sliced per round below.

    Round 1: 8 wires -> 4 active (4 CRZ gates)
    Round 2: 4 wires -> 2 active (2 CRZ gates)
    Round 3: 2 wires -> 1 active (1 CRZ gate)
    Total trainable params: 8 (initial RY) + 4 + 2 + 1 = 15
    """
    active = list(range(N_WIRES))
    idx = 0
    for w in active:
        qml.RY(weights[idx], wires=w)
        idx += 1

    for _ in range(3):  # log2(8) = 3 pooling rounds
        n_pairs = len(active) // 2
        active = conv_pool_round(active, weights[idx : idx + n_pairs])
        idx += n_pairs

    return qml.expval(qml.PauliZ(active[0]))


N_PARAMS = N_WIRES + N_WIRES // 2 + N_WIRES // 4 + N_WIRES // 8  # 8 + 4 + 2 + 1 = 15
