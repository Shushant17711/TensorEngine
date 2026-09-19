#!/usr/bin/env bash
# Reproduce the current state of the project end to end.
#
# Grows as the project does (spec §7 lists this as the eventual one-command reproduction
# entry point for the full audit battery + sweep + figures). Covers everything that
# actually exists so far: environment setup, the Week 1 capability spike, all structural
# regression tests, and every model that produces a numeric chi* sweep (1, 2, 3, 4 -- see
# results/chi_star_table.md). Models 5/6 remain "not applicable" (genuinely tangled, no
# surrogate family covers them yet).
#
# Total runtime: ~20-25 minutes, dominated by the MPS sweep scripts (parameter-shift
# training through default.tensor is the bottleneck -- see LIMITATIONS.md). The tree-model
# sweep is fast (~2 min) since TreeSurrogate trains in plain torch, no quantum device.
set -euo pipefail

cd "$(dirname "$0")/.."

echo "== Setting up environment (uv) =="
uv sync --group dev

echo "== Week 1 spike: default.tensor capability probe =="
uv run python scripts/week1_spike.py

echo "== All structural/regression tests (parser topology classification for every model) =="
uv run pytest tests/ -v

echo "== model_01_vqc_chain: chi sweep (~13 min: 3 seeds x 3 chi, parameter-shift training) =="
uv run python scripts/run_model_01_sweep.py

echo "== model_03_mps_classifier: chi sweep (~8-10 min: 2 seeds x 3 chi, parameter-shift training) =="
uv run python scripts/run_model_03_sweep.py

echo "== model_02/04 (tree circuits): TreeSurrogate chi sweep (~2 min, pure torch) =="
uv run python scripts/run_tree_models_sweep.py

echo "== Lint =="
uv run ruff check .
uv run ruff format --check .
