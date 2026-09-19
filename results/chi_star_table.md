# χ* summary table

Aggregates `targets/*/sweep_results.md` across the battery. Updated as models are run —
each row links to its model's own detailed result file rather than duplicating numbers
that could drift out of sync.

| Model | Source | Topology | Outcome | χ\* |
|---|---|---|---|---|
| [model_01_vqc_chain](../targets/model_01_vqc_chain/sweep_results.md) | Farhi & Neven / Mitarai et al. style (pilot model, not preregistered) | Linear chain | Dequantized (MPSSurrogate) | **2** (3/3 seeds) |
| [model_02_qcnn_pooling](../targets/model_02_qcnn_pooling/sweep_results.md) | Cong-Choi-Lukin style (pilot model, not preregistered) | Tree | Dequantized (TreeSurrogate — see caveat below) | **2** (3/3 seeds) |
| [model_03_mps_classifier](../targets/model_03_mps_classifier/sweep_results.md) | Huggins et al. 2019, arXiv:1803.11537 | Linear chain | Dequantized (MPSSurrogate) | **1** (2/2 seeds — see note below) |
| [model_04_ttn_classifier](../targets/model_04_ttn_classifier/sweep_results.md) | Huggins et al. 2019, arXiv:1803.11537 | Tree | Dequantized (TreeSurrogate — see caveat below) | **2** (3/3 seeds) |
| [model_05_circuit_centric](../targets/model_05_circuit_centric/sweep_results.md) | Schuld et al. 2020, arXiv:1804.00633 | Genuinely tangled | Not applicable | — |
| [model_06_zz_feature_map](../targets/model_06_zz_feature_map/sweep_results.md) | Havlicek et al. 2019, arXiv:1804.11326 | Genuinely tangled (all-to-all) | Not applicable | — |

**Caveat on model_02/04's χ\*=2 (see NOTES_WEEK6.md and `dequant_engine/tree_surrogate.py`):**
`TreeSurrogate`, unlike `MPSSurrogate`, is not gate-structure-matched — it's a generic
tree-shaped classical classifier trained on data labels, sharing only the topology with
the reference circuit. Both models' full sweep tables are numerically **identical** as a
direct consequence. Read this χ\*=2 as "a generic tree-shaped classical model needs bond
dimension 2 to match a tree-shaped quantum circuit's accuracy here," not as a claim about
either specific circuit's own entanglement — a real, weaker claim than model_01/03's
`MPSSurrogate` results.

**Note on model_03's χ\*=1:** predicted χ\*≈2 (the circuit's own bond dimension); not
confirmed, on two separate attempts. First attempt (NOTES_WEEK4.md) used an *invented*
synthetic task and was correctly flagged as not meaningful — an entanglement-insensitive
made-up task tells you nothing about the circuit. Re-run (NOTES_WEEK5.md) against the
paper's own actual benchmark (real Iris data, versicolor vs virginica, the class pair
Schuld et al. 2020 — model_05's paper — is confirmed to use) still found χ\*=1. This time
it's a real result: Iris classification at this scale genuinely doesn't require this
circuit's entanglement to match its own reported accuracy — consistent with a known,
published critique that small classical benchmarks like Iris are often too easy to
demonstrate real quantum structure is doing anything, which is this project's own central
thesis (README §1). **χ\* is task-dependent, not purely a property of the circuit** — the
methodological lesson survived even after the underlying task got fixed.

## Reading this table

- "Not applicable" is not a gap in this table — it's the correct, informative outcome for
  any circuit whose entanglement graph no surrogate family currently covers (only
  genuinely-tangled graphs now, per LIMITATIONS.md — path and tree are both covered as of
  Week 6).
- Models 5/6 (genuinely tangled — a cycle present and a wire with 3+ partners) remain the
  only ones without any surrogate family available. A general non-local surrogate
  (explicit mid-circuit SWAP network) is the clear next-priority engineering work.
