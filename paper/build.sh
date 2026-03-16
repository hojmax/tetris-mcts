#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PAPER_ROOT="$REPO_ROOT/paper"

echo "Generating paper plots..."
uv run python "$PAPER_ROOT/scripts/make_avg_attack_vs_runtime_plot.py" \
  --output_path "$PAPER_ROOT/plots/avg_attack_vs_runtime.pdf"

echo "Compiling Typst..."
typst compile \
  --root "$PAPER_ROOT" \
  "$PAPER_ROOT/typst/main.typ" \
  "$PAPER_ROOT/paper.pdf"
