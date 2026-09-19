"""QNode -> gate sequence -> MPS topology (spec section 4.1, "the matching layer").

Week 2 finding (supersedes the tentative Week 1 note that this layer might be nearly
redundant with ``default.tensor`` -- see NOTES_WEEK2.md): ``default.tensor`` does **not**
auto-optimize qubit ordering. Its physical MPS chain order is fixed by numeric wire-label
order (wire 0 next to wire 1 next to wire 2, ...), regardless of what order you pass as the
device's ``wires=`` argument. Confirmed empirically: an 8-qubit circuit entangling pairs
(0,7),(1,6),(2,5),(3,4) needs bond dimension >= 8 for exact results under the natural
label order, but is *exactly* reproduced at bond dimension 2 once the wires are relabeled
(via ``qml.map_wires``) so each entangled pair is adjacent in label order. So this module's
ordering heuristic is not a nice-to-have for the rare all-to-all case -- it directly
determines whether a reported chi* is a fair measurement of the circuit's actual
entanglement structure, or an artifact of an unlucky natural wire order.

This module:

1. Pulls the operation sequence out of an arbitrary QNode (gates, wires, param structure).
2. Builds the entanglement graph (which wire pairs are ever jointly acted on by a
   multi-qubit gate).
3. Checks whether that graph is *path-decomposable* -- i.e. every wire has degree <= 2 and
   there are no cycles, which is exactly the condition under which SOME linear wire
   ordering makes every entangling gate act on physically adjacent wires (a chain, or a
   disjoint union of chains covering separate wire groups). When true, a relabeling alone
   (no mid-circuit SWAP gates) is sufficient -- this is what ``mps_surrogate.py`` acts on.
4. When the graph is *not* path-decomposable (a wire touched by >=3 entangling gates to
   distinct partners, or a cycle), no relabeling can make every gate local -- an explicit
   SWAP network embedded in the circuit itself would be needed, which this module does not
   construct (see ``FAILURE_MODES``).

What this module does *not* do: build the tensors itself. That's ``default.tensor``'s job
for any circuit this module judges path-decomposable; ``mps_surrogate.py`` applies the
relabeling this module proposes and lets the device do the actual MPS contraction.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pennylane as qml

FAILURE_MODES = """
Known failure cases of the qubit-ordering layer used here (documented per spec section
4.1's explicit instruction to record ordering-heuristic failure cases):

1. A wire touched by entangling gates with 3+ distinct partner wires (degree >= 3 in the
   entanglement graph) has NO linear ordering that keeps every entangling pair adjacent --
   an exact structural fact (a graph with max degree <= 2 is precisely a disjoint union of
   paths and simple cycles; a degree-3+ node rules that out entirely), not a heuristic
   failure. These circuits genuinely need a SWAP network mid-circuit to become MPS-local,
   which this module does not construct -- MPSSurrogate raises NotImplementedError for
   them rather than silently reporting an inflated chi*.
1a. A *ring* component (every wire in it has degree exactly 2, forming one simple cycle --
    e.g. ``qml.BasicEntanglerLayers``'s default entangler, a very common real ansatz
    pattern) is a special, less-severe case: cutting any ONE edge turns it into a path, so
    every edge but that one can be made adjacent by reordering. This module detects rings
    separately (``CircuitTopology.ring_wraparound_edges``) and still refuses to build a
    surrogate for them (a single unavoidable long-range edge remains, needing either extra
    bond dimension across the whole chain or a real SWAP), but reports this distinctly from
    the harder degree->=3 case since it is much closer to representable, and points at
    exactly which single edge is the problem.
2. Circuits whose entanglement pattern changes across layers in a way that would need a
   *different* wire order per layer (e.g. layer 1 entangles (0,1),(2,3); layer 2 entangles
   (1,2),(3,0)) are only checked against ONE static global ordering here -- the combined
   entanglement graph across all layers is what gets tested, so a circuit that is
   layer-by-layer path-decomposable but not path-decomposable in aggregate is (correctly,
   if conservatively) rejected rather than handled with per-layer relabeling.
3. Where the graph IS path-decomposable, the specific chain found is produced by a greedy
   nearest-neighbor-chain walk (see ``_greedy_chain_ordering``) rather than an exhaustive
   search -- for a true disjoint-union-of-paths graph this always recovers a fully valid
   ordering (there's no ambiguity to get wrong: each node has at most 2 neighbors), but it
   has not been checked against every conceivable disconnected-component arrangement.
"""


@dataclass
class CircuitTopology:
    """Structural summary of a QNode's operation sequence, used to decide how to build
    its MPS surrogate and, later (Week 7, H2), as regression features for chi* prediction.
    """

    n_wires: int
    wire_order: list[int]  # chain[position] = logical wire placed there
    gate_names: list[str]
    entangling_pairs: list[tuple[int, int]]
    n_params: int
    is_1d_local: bool  # True iff SOME relabeling makes every entangling pair adjacent
    needs_reordering: bool  # True iff wire_order differs from the natural 0..n-1 order
    max_entangling_span: int  # span under the circuit's own (natural) wire order
    achievable_span: int  # span under wire_order -- <=1 iff is_1d_local
    ring_wraparound_edges: list[tuple[int, int]] = field(default_factory=list)
    ordering_note: str = field(default="")

    def summary(self) -> str:
        if self.is_1d_local and not self.needs_reordering:
            locality = "already 1D-local under the natural wire order"
        elif self.is_1d_local:
            locality = f"path-decomposable after reordering to {self.wire_order}"
        elif self.ring_wraparound_edges:
            locality = (
                f"{len(self.ring_wraparound_edges)} ring component(s) found "
                f"(wraparound edges {self.ring_wraparound_edges}) -- one unavoidable "
                "long-range edge per ring, see FAILURE_MODES item 1a"
            )
        else:
            locality = (
                f"NOT path-decomposable (best-effort span {self.achievable_span}) "
                "-- needs a SWAP network, see FAILURE_MODES"
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


def _classify_topology(pairs: list[tuple[int, int]]) -> tuple[bool, list[tuple[int, int]]]:
    """Exact classification of the entanglement graph.

    Returns ``(degree_ok, ring_wraparound_edges)``:

    - ``degree_ok=False`` means some wire has 3+ distinct entangling partners -- no
      relabeling can help (FAILURE_MODES item 1); the ring analysis doesn't apply.
    - ``degree_ok=True`` and ``ring_wraparound_edges`` empty: the graph is a disjoint
      union of simple paths -- fully path-decomposable, every edge can be made adjacent.
    - ``degree_ok=True`` and ``ring_wraparound_edges`` non-empty: every node has degree
      <= 2 but one or more connected components is a simple cycle rather than a path (a
      "ring" -- e.g. ``qml.BasicEntanglerLayers``'s default entangler). Each such
      component contributes exactly one edge here: the one that, if removed, would turn
      that cycle into a path (found as the edge that would close a union-find cycle).
      FAILURE_MODES item 1a.
    """
    unique_edges = set(pairs)
    degree: dict[int, int] = {}
    for a, b in unique_edges:
        degree[a] = degree.get(a, 0) + 1
        degree[b] = degree.get(b, 0) + 1
    if any(d > 2 for d in degree.values()):
        return False, []

    parent: dict[int, int] = {}

    def find(x: int) -> int:
        root = x
        while parent.get(root, root) != root:
            root = parent[root]
        while parent.get(x, x) != root:
            parent[x], x = root, parent.get(x, x)
        return root

    ring_edges: list[tuple[int, int]] = []
    for a, b in unique_edges:
        parent.setdefault(a, a)
        parent.setdefault(b, b)
        ra, rb = find(a), find(b)
        if ra == rb:
            ring_edges.append((a, b))  # this edge closes a cycle -- the wraparound edge
        else:
            parent[ra] = rb

    return True, ring_edges


def _greedy_chain_ordering(n_wires: int, pairs: list[tuple[int, int]]) -> list[int]:
    """Greedy nearest-neighbor-chain heuristic: repeatedly attach the unplaced wire with
    the most entangling edges to already-placed wires, at whichever end of the chain
    minimizes the new edge's span. When the entanglement graph is path-decomposable (see
    ``_is_path_decomposable``), each node has at most 2 neighbors, so this walk has no
    real ambiguity and recovers a fully valid ordering. For non-path-decomposable graphs
    this is a best-effort heuristic only (see FAILURE_MODES item 3) -- its result is
    informational, not something ``mps_surrogate.py`` acts on in that case.
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
            # arbitrary leftover wire to keep the ordering total (isolated wire, or a
            # separate path component -- span across this junction doesn't matter since
            # there's no entangling edge there).
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
    # Real ansatze are usually built from templates (e.g. qml.BasicEntanglerLayers,
    # qml.StronglyEntanglingLayers) that queue as a single opaque operation rather than
    # their constituent gates -- decompose down to <=2-qubit gates so the entanglement
    # graph reflects what's actually applied, not the template's name.
    (tape,), _ = qml.transforms.decompose(tape, stopping_condition=lambda op: len(op.wires) <= 2)

    n_wires = len(tape.wires)
    gate_names = [op.name for op in tape.operations]
    n_params = sum(op.num_params for op in tape.operations)
    entangling_pairs = _extract_entangling_pairs(tape)

    natural_spans = [abs(b - a) for a, b in entangling_pairs]
    max_span = max(natural_spans, default=0)

    degree_ok, ring_edges = _classify_topology(entangling_pairs)
    path_decomposable = degree_ok and not ring_edges
    chain = _greedy_chain_ordering(n_wires, entangling_pairs)
    position = {wire: idx for idx, wire in enumerate(chain)}
    achievable_span = max((abs(position[a] - position[b]) for a, b in entangling_pairs), default=0)

    needs_reordering = chain != list(range(n_wires))
    if path_decomposable:
        note = (
            "already 1D-local under the natural wire order"
            if not needs_reordering
            else "path-decomposable; reordered via greedy nearest-neighbor-chain walk"
        )
    elif degree_ok:
        note = (
            f"{len(ring_edges)} ring component(s) -- see FAILURE_MODES item 1a "
            "(one unavoidable long-range edge per ring)"
        )
    else:
        note = "NOT path-decomposable (degree >= 3 wire) -- needs a SWAP network, see FAILURE_MODES"

    return CircuitTopology(
        n_wires=n_wires,
        wire_order=chain,
        gate_names=gate_names,
        entangling_pairs=entangling_pairs,
        n_params=n_params,
        is_1d_local=path_decomposable,
        needs_reordering=needs_reordering,
        max_entangling_span=max_span,
        achievable_span=achievable_span,
        ring_wraparound_edges=ring_edges,
        ordering_note=note,
    )
