#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PAPER_ROOT="$REPO_ROOT/paper"

echo "Generating paper plots..."
uv run python "$PAPER_ROOT/scripts/make_dummy_bridge_runtime_plot.py" \
  --output_path "$PAPER_ROOT/plots/bridge_vs_runtime_dummy.pdf"

echo "Compiling Typst..."
mkdir -p "$PAPER_ROOT/build"
typst compile \
  --root "$PAPER_ROOT" \
  "$PAPER_ROOT/typst/main.typ" \
  "$PAPER_ROOT/build/paper.pdf"
