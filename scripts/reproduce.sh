#!/usr/bin/env bash
# Reproduce the current state of the project end to end.
#
# Grows as the project does (spec §7 lists this as the eventual one-command reproduction
# entry point for the full audit battery + sweep + figures). Covers everything that
# actually exists so far: environment setup, the Week 1 capability spike, all structural
# regression tests (including every model's topology classification -- models 2/4/5/6 stop
# there since they're confirmed "not applicable", see results/chi_star_table.md), and the
# two models that actually produce a numeric chi* sweep (models 1 and 3).
#
# Total runtime: ~20-25 minutes, dominated by the two sweep scripts (parameter-shift
# training through default.tensor is the bottleneck -- see LIMITATIONS.md).
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

echo "== Lint =="
uv run ruff check .
uv run ruff format --check .
