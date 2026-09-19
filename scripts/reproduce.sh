#!/usr/bin/env bash
# Reproduce the current state of the project end to end.
#
# Grows as the project does (spec §7 lists this as the eventual one-command reproduction
# entry point for the full audit battery + sweep + figures). For now it only covers what
# actually exists: environment setup, the Week 1 capability spike, and the E1 validation
# test suite.
set -euo pipefail

cd "$(dirname "$0")/.."

echo "== Setting up environment (uv) =="
uv sync --group dev

echo "== Week 1 spike: default.tensor capability probe =="
uv run python scripts/week1_spike.py

echo "== E1 validation: chi*=2 recovery on toy circuit =="
uv run pytest tests/ -v

echo "== Lint =="
uv run ruff check .
uv run ruff format --check .
