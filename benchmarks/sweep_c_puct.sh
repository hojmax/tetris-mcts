#!/usr/bin/env bash
# Sweep c_puct with the generic MCTS config sweep script.
# Runs 50 games per configuration and outputs JSON + PNG plot.
#
# Usage:
#   ./benchmarks/sweep_c_puct.sh <run_dir>
#   ./benchmarks/sweep_c_puct.sh training_runs/v32
#   ./benchmarks/sweep_c_puct.sh training_runs/v32 100  # 100 games each
set -euo pipefail

RUN_DIR="${1:?Usage: $0 <run_dir> [num_games]}"
NUM_GAMES="${2:-50}"

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${SCRIPT_DIR}/.venv/bin/python"

echo "=== MCTS c_puct sweep ==="
echo "Run directory: ${RUN_DIR}"
echo "Games per config: ${NUM_GAMES}"
echo ""

"${PYTHON}" "${SCRIPT_DIR}/scripts/ablations/sweep_mcts_config.py" \
    --run_dir "${RUN_DIR}" \
    --sweep_param c_puct \
    --sweep_values '[0.75, 1.0, 1.5, 2.0, 2.5]' \
    --num_games "${NUM_GAMES}"
