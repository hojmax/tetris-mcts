#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
RUNS="${RUNS:-3}"
NUM_GAMES="${NUM_GAMES:-20}"
SEED_START="${SEED_START:-42}"
MCTS_SEED="${MCTS_SEED:-123}"
SIMS_DUMMY="${SIMS_DUMMY:-2000}"
SIMS_NN="${SIMS_NN:-1000}"
WITH_NN="${WITH_NN:-0}"
MODEL_PATH="${MODEL_PATH:-training_runs/vXX/checkpoints/latest.onnx}"

PROFILE_DUMMY_PATH="${PROFILE_DUMMY_PATH:-benchmarks/profile_cpu_dummy.jsonl}"
PROFILE_NN_PATH="${PROFILE_NN_PATH:-benchmarks/profile_with_nn.jsonl}"

HOST_TAG="$(hostname | tr '[:space:]' '_' | tr '/:' '__')"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
META_PATH="${META_PATH:-benchmarks/cpu_compare_meta_${HOST_TAG}_${STAMP}.txt}"

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Python not found or not executable at: $PYTHON_BIN"
    echo "Set PYTHON_BIN, for example: PYTHON_BIN=.venv/bin/python"
    exit 1
fi

usage() {
    cat <<'EOF'
Run deterministic CPU game-generation benchmarks for Mac vs AWS comparison.

Usage:
  benchmarks/run_cpu_compare.sh [--with-nn] [--model-path PATH]

Options:
  --with-nn           Also run ONNX-network benchmark in addition to dummy mode.
  --model-path PATH   ONNX model path for --with-nn (default: training_runs/vXX/checkpoints/latest.onnx).
  -h, --help          Show this help.

Environment overrides:
  PYTHON_BIN=.venv/bin/python
  RUNS=3
  NUM_GAMES=20
  SEED_START=42
  MCTS_SEED=123
  SIMS_DUMMY=2000
  SIMS_NN=1000
  PROFILE_DUMMY_PATH=benchmarks/profile_cpu_dummy.jsonl
  PROFILE_NN_PATH=benchmarks/profile_with_nn.jsonl
  META_PATH=benchmarks/cpu_compare_meta_<host>_<timestamp>.txt
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --with-nn)
            WITH_NN=1
            shift
            ;;
        --model-path)
            MODEL_PATH="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1"
            usage
            exit 1
            ;;
    esac
done

collect_machine_info() {
    {
        echo "=== META ==="
        date -u
        hostname
        uname -a
        git rev-parse HEAD
        "$PYTHON_BIN" -V
        rustc -V
        cargo -V
        echo

        if [[ "$(uname -s)" == "Darwin" ]]; then
            echo "=== MAC HARDWARE ==="
            sysctl -n machdep.cpu.brand_string
            echo -n "physicalcpu logicalcpu memsize_bytes: "
            sysctl -n hw.physicalcpu hw.logicalcpu hw.memsize
            echo -n "perflevel0 perflevel1: "
            sysctl -n hw.perflevel0.physicalcpu hw.perflevel1.physicalcpu 2>/dev/null || true
            pmset -g batt || true
        else
            echo "=== LINUX HARDWARE ==="
            lscpu
            echo "nproc: $(nproc)"
            free -h || true
            echo -n "cpu_scaling_governor: "
            cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || true
        fi
    } | tee "$META_PATH"
}

summarize_jsonl() {
    local jsonl_path="$1"
    local label="$2"

    "$PYTHON_BIN" - "$jsonl_path" "$label" <<'PY'
import json
import statistics
import sys
from pathlib import Path

jsonl_path = Path(sys.argv[1])
label = sys.argv[2]

rows = [json.loads(line) for line in jsonl_path.read_text().splitlines() if line.strip()]
if not rows:
    raise RuntimeError(f"No rows found in {jsonl_path}")

mps = [row["timing"]["moves_per_second"] for row in rows]
gps = [row["timing"]["games_per_second"] for row in rows]
mpm = [row["timing"]["avg_time_per_move_ms"] for row in rows]

print(f"\n=== SUMMARY ({label}) ===")
print(f"file: {jsonl_path}")
print(f"runs: {len(rows)}")
print(f"moves_per_second: {[round(x, 2) for x in mps]} median={statistics.median(mps):.2f}")
print(f"games_per_second: {[round(x, 3) for x in gps]} median={statistics.median(gps):.3f}")
print(f"avg_time_per_move_ms: {[round(x, 2) for x in mpm]} median={statistics.median(mpm):.2f}")
PY
}

run_dummy_benchmark() {
    rm -f "$PROFILE_DUMMY_PATH"
    for i in $(seq 1 "$RUNS"); do
        echo "Running dummy benchmark ${i}/${RUNS} ..."
        make profile \
            SIMS="$SIMS_DUMMY" \
            PROFILE_ARGS="--use_dummy_network --num_games ${NUM_GAMES} --seed_start ${SEED_START} --mcts_seed ${MCTS_SEED} --output ${PROFILE_DUMMY_PATH}"
    done
    summarize_jsonl "$PROFILE_DUMMY_PATH" "CPU-only dummy network"
}

run_nn_benchmark() {
    if [[ "$WITH_NN" != "1" ]]; then
        return
    fi

    if [[ ! -f "$MODEL_PATH" ]]; then
        echo "Model file not found: $MODEL_PATH"
        echo "Set MODEL_PATH=/path/to/latest.onnx or pass --model-path PATH"
        exit 1
    fi

    rm -f "$PROFILE_NN_PATH"
    for i in $(seq 1 "$RUNS"); do
        echo "Running NN benchmark ${i}/${RUNS} ..."
        make profile \
            MODEL_PROFILE="$MODEL_PATH" \
            SIMS="$SIMS_NN" \
            PROFILE_ARGS="--num_games ${NUM_GAMES} --seed_start ${SEED_START} --mcts_seed ${MCTS_SEED} --output ${PROFILE_NN_PATH}"
    done
    summarize_jsonl "$PROFILE_NN_PATH" "ONNX network"
}

echo "Building release extension..."
make build

echo "Collecting machine info..."
collect_machine_info

echo "Running deterministic CPU benchmark (dummy network)..."
run_dummy_benchmark

run_nn_benchmark

echo
echo "Saved outputs:"
echo "  meta:   $META_PATH"
echo "  dummy:  $PROFILE_DUMMY_PATH"
if [[ "$WITH_NN" == "1" ]]; then
    echo "  withnn: $PROFILE_NN_PATH"
fi
