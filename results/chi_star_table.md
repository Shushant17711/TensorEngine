# χ* summary table

Aggregates `targets/*/sweep_results.md` across the battery. Updated as models are run —
each row links to its model's own detailed result file rather than duplicating numbers
that could drift out of sync.

| Model | Source | Topology | Outcome | χ\* |
|---|---|---|---|---|
| [model_01_vqc_chain](../targets/model_01_vqc_chain/sweep_results.md) | Farhi & Neven / Mitarai et al. style (pilot model, not preregistered) | Linear chain | Dequantized | **2** (3/3 seeds) |
| [model_02_qcnn_pooling](../targets/model_02_qcnn_pooling/sweep_results.md) | Cong-Choi-Lukin style (pilot model, not preregistered) | Tree | Not applicable | — |
| [model_03_mps_classifier](../targets/model_03_mps_classifier/sweep_results.md) | Huggins et al. 2019, arXiv:1803.11537 | Linear chain | Dequantized | **1** (2/2 seeds — see note below) |
| [model_04_ttn_classifier](../targets/model_04_ttn_classifier/sweep_results.md) | Huggins et al. 2019, arXiv:1803.11537 | Tree | Not applicable | — |
| [model_05_circuit_centric](../targets/model_05_circuit_centric/sweep_results.md) | Schuld et al. 2020, arXiv:1804.00633 | Genuinely tangled | Not applicable | — |
| [model_06_zz_feature_map](../targets/model_06_zz_feature_map/sweep_results.md) | Havlicek et al. 2019, arXiv:1804.11326 | Genuinely tangled (all-to-all) | Not applicable | — |

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
  any circuit whose entanglement graph the matching layer can't turn into a path (tree or
  genuinely tangled — see LIMITATIONS.md for what each of those means structurally).
- Only path-decomposable circuits (models 01, 03) currently produce a numeric χ\*. This is
  itself a finding worth keeping (see NOTES_WEEK4.md): of the 6 models examined so far, 4
  fall outside this tool's current MPS-only coverage — extending coverage (a `TreeSurrogate`
  for the tree cases, a SWAP-network-based surrogate for the genuinely tangled ones) is the
  clear next-priority engineering work, not more battery models of the same path-shaped
  kind.
