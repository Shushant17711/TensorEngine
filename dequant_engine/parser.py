"""QNode -> gate sequence -> MPS topology (spec section 4.1, "the matching layer").

Per the Week 1 finding (see NOTES_WEEK1.md), ``default.tensor`` already contracts an MPS
for us given a device and a qubit *ordering*; what it does not give us is the ordering
itself for circuits that aren't already laid out 1D-local. This module's job is:

1. Pull the operation sequence out of an arbitrary QNode (gates, wires, param structure).
2. Classify how "MPS-friendly" that sequence already is: entangling gates only between
   wires the circuit already treats as neighbors ("local") vs. entangling gates between
   distant wires ("non-local" / all-to-all).
3. For the non-local case, propose a canonical qubit ordering that minimizes how far
   entangled pairs are pushed apart -- the "ordering heuristic" the spec explicitly asks to
   be documented, failure cases included (see ``FAILURE_MODES`` below).

What this module does *not* do: build the tensors itself. That's ``default.tensor``'s job
(see NOTES_WEEK1.md) for circuits this module judges local-enough; ``mps_surrogate.py``
decides which path to take based on the ``CircuitTopology`` this module produces.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pennylane as qml

FAILURE_MODES = """
Known failure cases of the qubit-reordering heuristic used here (documented per spec
section 4.1's explicit instruction to record ordering-heuristic failure cases):

1. Dense all-to-all entanglement (e.g. a fully-connected entangling layer on N wires) has
   no linear ordering that keeps every entangling pair adjacent -- the heuristic minimizes
   *total* span, not worst-case span, so some pairs will still be pushed far apart and
   truncation error at fixed chi will be systematically underestimated for those pairs.
2. Circuits whose entanglement pattern changes across layers (e.g. layer 1 entangles
   (0,1),(2,3); layer 2 entangles (1,2),(3,0)) are optimized for a *single* static ordering
   here -- there is no support yet for a per-layer reordering (which would require SWAP
   networks / bond permutations that this module does not construct).
3. The heuristic is a greedy nearest-neighbor-chain construction (see
   ``_greedy_chain_ordering``), not an exact minimum linear arrangement -- it can be
   suboptimal versus an exact solver (which is NP-hard in general) for large wire counts.
"""


@dataclass
class CircuitTopology:
    """Structural summary of a QNode's operation sequence, used to decide how to build
    its MPS surrogate and, later (Week 7, H2), as regression features for chi* prediction.
    """

    n_wires: int
    wire_order: list[int]
    gate_names: list[str]
    entangling_pairs: list[tuple[int, int]]
    n_params: int
    is_1d_local: bool
    max_entangling_span: int
    ordering_note: str = field(default="")

    def summary(self) -> str:
        locality = (
            "1D-local (no reordering needed)"
            if self.is_1d_local
            else (
                f"non-local (max span {self.max_entangling_span}); proposed order {self.wire_order}"
            )
        )
        return (
            f"CircuitTopology: {self.n_wires} wires, {len(self.gate_names)} gates, "
            f"{self.n_params} trainable params, {len(self.entangling_pairs)} entangling "
            f"pairs, {locality}"
        )


def _extract_entangling_pairs(tape) -> list[tuple[int, int]]:
    """Two-qubit (and higher, decomposed pairwise) gates define the entanglement graph."""
    pairs = []
    for op in tape.operations:
        wires = op.wires.tolist()
        if len(wires) == 2:
            pairs.append((min(wires), max(wires)))
        elif len(wires) > 2:
            # Multi-qubit gates (e.g. Toffoli, MultiRZ): treat every pairwise combination
            # as a potential entangling edge -- conservative (may overcount locality
            # requirements) but never silently drops a real dependency.
            for i, a in enumerate(wires):
                for b in wires[i + 1 :]:
                    pairs.append((min(a, b), max(a, b)))
    return pairs


def _greedy_chain_ordering(n_wires: int, pairs: list[tuple[int, int]]) -> list[int]:
    """Greedy nearest-neighbor-chain heuristic: repeatedly attach the unplaced wire with
    the most entangling edges to already-placed wires, at whichever end of the chain
    minimizes the new edge's span. Not optimal (linear arrangement is NP-hard in general)
    -- see FAILURE_MODES item 3.
    """
    if n_wires <= 1:
        return list(range(n_wires))

    edge_weight: dict[tuple[int, int], int] = {}
    for a, b in pairs:
        edge_weight[(a, b)] = edge_weight.get((a, b), 0) + 1

    neighbors: dict[int, dict[int, int]] = {w: {} for w in range(n_wires)}
    for (a, b), weight in edge_weight.items():
        neighbors[a][b] = neighbors[a].get(b, 0) + weight
        neighbors[b][a] = neighbors[b].get(a, 0) + weight

    remaining = set(range(n_wires))
    # Seed with the most-connected wire.
    start = max(remaining, key=lambda w: sum(neighbors[w].values()), default=0)
    chain = [start]
    remaining.remove(start)

    while remaining:
        best_wire, best_score, best_end = None, -1.0, "right"
        for end_name, end_wire in (("left", chain[0]), ("right", chain[-1])):
            for candidate in remaining:
                score = neighbors[candidate].get(end_wire, 0)
                if score > best_score:
                    best_wire, best_score, best_end = candidate, score, end_name
        if best_wire is None or best_score <= 0:
            # No remaining wire shares an edge with either chain end: attach an
            # arbitrary leftover wire to keep the ordering total (isolated wire).
            best_wire = next(iter(remaining))
            best_end = "right"
        if best_end == "left":
            chain.insert(0, best_wire)
        else:
            chain.append(best_wire)
        remaining.remove(best_wire)

    return chain


def parse_qnode(qnode: qml.QNode, *example_args, **example_kwargs) -> CircuitTopology:
    """Extract a ``CircuitTopology`` from a QNode by tracing it once with example inputs.

    ``example_args``/``example_kwargs`` are only used to build the tape (they determine
    control flow for circuits with data-dependent structure); parameter *values* don't
    matter, only which operations get queued.
    """
    tape = qml.workflow.construct_tape(qnode)(*example_args, **example_kwargs)

    n_wires = len(tape.wires)
    gate_names = [op.name for op in tape.operations]
    n_params = sum(op.num_params for op in tape.operations)
    entangling_pairs = _extract_entangling_pairs(tape)

    spans = [abs(b - a) for a, b in entangling_pairs]
    max_span = max(spans, default=0)
    is_1d_local = max_span <= 1

    if is_1d_local:
        wire_order = list(range(n_wires))
        note = "already 1D-local under the natural wire order"
    else:
        wire_order = _greedy_chain_ordering(n_wires, entangling_pairs)
        note = "reordered via greedy nearest-neighbor-chain heuristic; see FAILURE_MODES"

    return CircuitTopology(
        n_wires=n_wires,
        wire_order=wire_order,
        gate_names=gate_names,
        entangling_pairs=entangling_pairs,
        n_params=n_params,
        is_1d_local=is_1d_local,
        max_entangling_span=max_span,
        ordering_note=note,
    )
