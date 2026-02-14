# Performance Benchmarks

This directory contains performance profiling tools and results for tracking regression and finding bottlenecks.

## Files

- `models/` - Optional benchmark model directory (may be empty in local clones)
- `profile_results.jsonl` - Timing results from `make profile` (tracked history of benchmark runs)

## Quick Start

### Timing Benchmarks

Track performance over time with repeatable benchmarks:

```bash
# Run 10 games with 100 simulations (default)
make profile

# Faster profiling with fewer simulations
make profile SIMS=50

# More accurate with more simulations
make profile SIMS=200

# Bootstrap/dummy-network profiling (no ONNX inference), e.g. 4000 simulations
make profile SIMS=4000 PROFILE_ARGS="--use_dummy_network"

# If Makefile's default MODEL_PROFILE path is missing, set one explicitly
make profile MODEL_PROFILE=training_runs/v32/checkpoints/latest.onnx
```

Results are saved to `profile_results.jsonl` with:
- Total time, per-game time, per-move time
- Throughput (moves/sec, games/sec)
- Game statistics (moves, attack, etc.)
- Configuration used

Each run appends a new line, so you can track performance over time:

```bash
# View latest result
tail -1 benchmarks/profile_results.jsonl | python -m json.tool

# Compare all runs
cat benchmarks/profile_results.jsonl | jq '.timing.moves_per_second'
```

### Interactive Profiling (Recommended)

Use [samply](https://github.com/mstange/samply) for interactive flamegraph visualization:

```bash
# Install samply (one-time)
cargo install samply

# Profile and open interactive viewer
make profile-samply SIMS=50
```

This automatically:
1. Records all function calls (Rust + Python)
2. Opens interactive flamegraph in your browser
3. Shows time spent in each function
4. Lets you zoom/search/filter

**No code changes needed** - samply captures everything automatically!

### macOS Native Profiling

Use Instruments for Apple-optimized profiling:

```bash
# Record profile
instruments -t "Time Profiler" python tetris_mcts/scripts/profile_games.py --num_games 3

# View in Instruments.app
open profile.trace
```

## Interpreting Results

### Timing Benchmarks

Key metrics:
- `avg_time_per_move_ms` - Most important for training speed
- `moves_per_second` - Throughput metric
- Compare across runs to detect regressions

### Flamegraphs (samply)

Width = time spent in function. Look for:
- Wide bars = expensive functions (optimization targets)
- Tall stacks = deep call chains
- MCTS search typically dominates (~98% of time)

Typical hotspots:
- `search_internal` - MCTS tree traversal
- `predict_masked` - Neural network inference
- `get_possible_placements` - Move generation

## Benchmark Model

If `benchmarks/models/parallel.onnx` exists in your checkout, you can use it as a fixed model.
If not, pass an explicit existing checkpoint path with `MODEL_PROFILE=...` and keep it consistent
across comparison runs.

To update the benchmark model:
```bash
cp training_runs/vN/checkpoints/parallel.onnx benchmarks/models/
cp training_runs/vN/checkpoints/parallel.onnx.data benchmarks/models/
git add benchmarks/models/
```
