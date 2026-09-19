"""Torch-trainable tree-tensor-network (TTN) surrogate for tree-shaped circuits (spec
section 4.1, extended per LIMITATIONS.md's flagged next step).

``MPSSurrogate`` refuses circuits whose entanglement graph is a tree rather than a path
(``CircuitTopology.is_tree``) -- QCNN-style convolution+pooling architectures
(``targets/model_02_qcnn_pooling``, ``targets/model_04_ttn_classifier``) are the clearest
real examples, per NOTES_WEEK3.md. Forcing a tree into an MPS encoding would need a bond
dimension inflated for no structural reason (the same "unfair chi*" problem NOTES_WEEK2.md
diagnosed for bad wire orderings); the honest fix is a *different* classical ansatz family
matched to the tree shape, not a bigger MPS.

This module builds that surrogate directly in torch (no `quimb`/`default.tensor` --
checked in NOTES_WEEK4.md's "TN method" note that `default.tensor` has no
bond-dimension-bounded tree-contraction mode to build on). It is a genuine, if classical
and un-normalized, tree tensor network: at each internal node, two children's
bond-dimension-chi state vectors are combined via their outer product and mapped back down
to dimension chi by a trainable linear map, then renormalized -- a multilinear
(tensor-network-faithful) contraction, not a generic neural net wearing a tree shape (no
elementwise nonlinearities inside the tree).

**Scope, stated explicitly (see LIMITATIONS.md):** only supports the "standard complete
balanced binary tree over N=2^k leaves, paired in leaf order" convention --
i.e. leaves (0,1),(2,3),...,(N-2,N-1) merge first, then their survivors pair up in order,
and so on to one root. This matches exactly how ``targets/model_02_qcnn_pooling`` and
``targets/model_04_ttn_classifier`` are built. General tree-topology inference from an
arbitrary ``CircuitTopology`` (reconstructing an unknown hierarchy from a flat edge list)
is NOT implemented -- out of scope this session, flagged as a further extension.
"""

from __future__ import annotations

from collections.abc import Callable

import torch

LEAF_DIM = 2  # each leaf starts as a 2-level (qubit-like) feature embedding


def _leaf_embed(x: torch.Tensor) -> torch.Tensor:
    """cos/sin embedding matching the paper's own single-qubit feature map (Huggins et
    al. 2019 Eq. 1 / the RY(pi*x) angle-encoding used throughout this repo's QNode
    circuits) -- kept consistent with the quantum reference model's own encoding so the
    surrogate is approximating the same function class, not a different one.
    """
    angle = torch.pi / 2 * x
    return torch.stack([torch.cos(angle), torch.sin(angle)])


class TreeSurrogate:
    """A bond-dimension-``chi``-bounded classical tree tensor network, trained to match a
    tree-shaped QNode's output. See module docstring for the exact topology this supports.

    Parameters
    ----------
    n_leaves:
        Number of leaves (input features / wires) -- must be a power of 2.
    chi:
        Bond dimension cap for every inter-level "message" passed up the tree.
    """

    def __init__(self, n_leaves: int, chi: int) -> None:
        if n_leaves & (n_leaves - 1) != 0 or n_leaves < 2:
            raise ValueError(f"n_leaves must be a power of 2 >= 2, got {n_leaves}")
        self.n_leaves = n_leaves
        self.chi = chi
        self.n_levels = n_leaves.bit_length() - 1  # log2(n_leaves)

        # Dimension entering each level: LEAF_DIM**2 at level 0 (two leaf embeddings
        # combined), chi**2 at every level after (two chi-dim children combined) --
        # output dimension is capped at chi (or the true input dim if smaller, so chi
        # only "bites" once the natural dimension would otherwise exceed it).
        self.level_dims: list[tuple[int, int, int]] = []  # (n_nodes, dim_in, dim_out)
        n_nodes = n_leaves // 2
        dim_in = LEAF_DIM * LEAF_DIM
        for _level in range(self.n_levels):
            dim_out = min(chi, dim_in)
            self.level_dims.append((n_nodes, dim_in, dim_out))
            n_nodes //= 2
            dim_in = dim_out * dim_out

    def init_weights(self, seed: int | None = None) -> list[torch.Tensor]:
        """One trainable (dim_out, dim_in) matrix per internal node, per level -- returned
        as a flat list of tensors (one per node, ordered level by level, left to right)
        so the caller can pass it straight to an optimizer, matching ``MPSSurrogate``'s
        convention of a single flat weights structure.
        """
        gen_kwargs = {"generator": torch.Generator().manual_seed(seed)} if seed is not None else {}
        weights = []
        for n_nodes, dim_in, dim_out in self.level_dims:
            for _ in range(n_nodes):
                w = torch.empty(dim_out, dim_in)
                torch.nn.init.xavier_uniform_(w, **gen_kwargs)
                weights.append(w)
        # final scalar readout from the root's dim_out-dimensional state
        root_dim = self.level_dims[-1][2]
        readout = torch.empty(root_dim)
        torch.nn.init.uniform_(readout, -1, 1, **gen_kwargs)
        weights.append(readout)
        return weights

    def __call__(self, weights: list[torch.Tensor], x: torch.Tensor) -> torch.Tensor:
        x = x.to(weights[0].dtype)
        nodes = [_leaf_embed(x[i]) for i in range(self.n_leaves)]
        w_idx = 0
        for n_nodes, _dim_in, _dim_out in self.level_dims:
            next_nodes = []
            for i in range(n_nodes):
                left, right = nodes[2 * i], nodes[2 * i + 1]
                combined = torch.outer(left, right).reshape(-1)  # (dim_in,)
                merged = weights[w_idx] @ combined  # (dim_out,)
                merged = merged / (torch.linalg.norm(merged) + 1e-12)  # keep bounded
                next_nodes.append(merged)
                w_idx += 1
            nodes = next_nodes
        root = nodes[0]
        readout = weights[w_idx]
        return torch.tanh(torch.dot(readout, root))  # bounded like a Pauli-Z expectation

    def fit(
        self,
        weights_init: list[torch.Tensor],
        loss_fn: Callable[[list[torch.Tensor]], torch.Tensor],
        *,
        lr: float = 0.1,
        epochs: int = 50,
        optimizer_cls=torch.optim.Adam,
    ) -> dict:
        """Same contract as ``MPSSurrogate.fit`` -- caller-defined loss, this class only
        owns the optimization loop, so the training protocol (loss/optimizer/epochs) can
        be kept identical to whatever trained the reference quantum model.
        """
        weights = [w.clone().detach().requires_grad_(True) for w in weights_init]
        optimizer = optimizer_cls(weights, lr=lr)

        history = []
        for _ in range(epochs):
            optimizer.zero_grad()
            loss = loss_fn(weights)
            loss.backward()
            optimizer.step()
            history.append(float(loss.detach()))

        return {"weights": [w.detach() for w in weights], "loss_history": history}
