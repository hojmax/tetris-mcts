#!/usr/bin/env bash
# Sweep Q-value normalization strategies: min-max baseline + tanh(Q/scale) variants.
# Runs 50 games per configuration and outputs JSON + PNG plot.
#
# Usage:
#   ./benchmarks/sweep_q_normalization.sh <run_dir>
#   ./benchmarks/sweep_q_normalization.sh training_runs/v32
#   ./benchmarks/sweep_q_normalization.sh training_runs/v32 100  # 100 games each
set -euo pipefail

RUN_DIR="${1:?Usage: $0 <run_dir> [num_games]}"
NUM_GAMES="${2:-50}"

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${SCRIPT_DIR}/.venv/bin/python"

echo "=== Q-value normalization sweep ==="
echo "Run directory: ${RUN_DIR}"
echo "Games per config: ${NUM_GAMES}"
echo ""

"${PYTHON}" "${SCRIPT_DIR}/scripts/ablations/sweep_mcts_config.py" \
    --run_dir "${RUN_DIR}" \
    --sweep_param q_scale \
    --sweep_values '[2, 4, 8, 16, 32]' \
    --include_minmax true \
    --num_games "${NUM_GAMES}"
