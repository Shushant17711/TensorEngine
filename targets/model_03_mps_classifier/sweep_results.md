# model_03_mps_classifier — sweep results (real Iris data)

Dataset: Iris, versicolor vs virginica, 16 train / 10 test samples (subsampled from the full 100 for compute — see scripts/run_model_03_sweep.py).

Reference (default.qubit) accuracy: **train 1.000, test 0.900** (final training loss 0.1930, lr=0.2, epochs=50)

**Prediction check (PREREGISTRATION.md model 3): chi\* ≈ 2 expected (circuit's own bond dimension V=1 => Schmidt rank cap 2^V=2).**

chi sweep, 2 seeds ([0, 1]), from-scratch training per chi:

| chi | mean train acc | mean test acc |
|---|---|---|
| 1 | 1.000 | 1.000 |
| 2 | 1.000 | 1.000 |
| 4 | 1.000 | 1.000 |

**chi* per seed (by train accuracy, tolerance 0.01): [1, 1] (2/2 seeds dequantized within the range tried)**
