# model_02_qcnn_pooling — sweep results (TreeSurrogate, real Wisconsin Breast Cancer data)

Dataset: Wisconsin Breast Cancer, 8 features, 40 train / 20 test samples (real data, deterministic subsample for compute).

Reference (default.qubit) accuracy: **train 0.800, test 0.900** (lr=0.2, epochs=60)

TreeSurrogate chi sweep, 3 seeds ([0, 1, 2]), from-scratch training per chi:

| chi | mean train acc | mean test acc |
|---|---|---|
| 1 | 0.642 | 0.767 |
| 2 | 0.983 | 0.850 |
| 4 | 1.000 | 0.850 |
| 8 | 1.000 | 0.817 |
| 16 | 1.000 | 0.833 |

**chi* per seed: [2, 2, 2] (3/3 seeds dequantized within the range tried)**

**Important caveat (see dequant_engine/tree_surrogate.py's module docstring and NOTES_WEEK6.md):** TreeSurrogate is NOT gate-structure-matched the way MPSSurrogate is -- it's a generic tree-shaped classical classifier trained directly on data labels, sharing only the topology (leaf count, tree shape) and leaf encoding with the reference circuit, not its specific gates or trained weights. This is why model_02 and model_04's tables above are numerically IDENTICAL: nothing about either circuit's own design enters the surrogate's training. Read chi* here as "the bond dimension a generic tree-shaped classical model needs to match *some* tree-shaped quantum circuit's accuracy on this task," not as a claim about this specific circuit's own entanglement structure.

This replaces this model's earlier structural-only "not applicable" write-up: with TreeSurrogate (a genuine bond-dimension-bounded tree tensor network, not an MPS forced into the wrong shape), a real chi* comparison is now possible. See NOTES_WEEK6.md.
