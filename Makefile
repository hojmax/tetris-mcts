.PHONY: run build build-dev clean rebuild test check play viz train replay profile profile-samply sweep-lr-model eval-nn-value-weight compare-offline-network-scaling

# Source cargo environment if available
SHELL := /bin/bash
CARGO_ENV := source $$HOME/.cargo/env 2>/dev/null || true
PYTHON := .venv/bin/python
PYTHON_ABS := $(abspath $(PYTHON))

# Find all Rust source files (including subdirectories)
RUST_SRC := $(shell find tetris_core/src -name '*.rs')
RELEASE_MARKER := .build_marker_release
DEV_MARKER := .build_marker_dev

# Build marker file to track if build is up to date (release mode)
$(RELEASE_MARKER): $(RUST_SRC) tetris_core/Cargo.toml tetris_core/pyproject.toml
	$(CARGO_ENV) && $(PYTHON) -m maturin develop --release --manifest-path tetris_core/Cargo.toml
	@rm -f $(DEV_MARKER)
	@touch $(RELEASE_MARKER)

# Explicit build target
build: $(RELEASE_MARKER)

# Fast debug build (much faster, for development only)
build-dev:
	$(CARGO_ENV) && $(PYTHON) -m maturin develop --manifest-path tetris_core/Cargo.toml
	@rm -f $(RELEASE_MARKER)
	@touch $(DEV_MARKER)

# Run the game (builds first if needed)
play: $(RELEASE_MARKER)
	$(PYTHON) tetris_mcts/scripts/tetris_game.py

# Run the MCTS visualizer (builds first if needed)
# Usage: make viz RUN_DIR=training_runs/v15
# Usage (dummy/no-network): make viz RUN_DIR=training_runs/v15 DUMMY_NETWORK=1
RUN_DIR ?= training_runs/v15
DUMMY_NETWORK ?= 0
viz: $(RELEASE_MARKER)
	$(PYTHON) tetris_mcts/scripts/inspection/mcts_visualizer.py $(if $(RUN_DIR),--run_dir $(RUN_DIR),) $(if $(filter 1 true TRUE yes YES,$(DUMMY_NETWORK)),--use_dummy_network true,)

# Force rebuild (clean first to avoid caching issues)
rebuild:
	cd tetris_core && $(CARGO_ENV) && cargo clean
	$(CARGO_ENV) && $(PYTHON) -m maturin develop --release --manifest-path tetris_core/Cargo.toml
	@rm -f $(DEV_MARKER)
	@touch $(RELEASE_MARKER)

# Run tests
test:
	$(MAKE) build-dev
	cd tetris_core && $(CARGO_ENV) && PYO3_PYTHON=$(PYTHON_ABS) cargo test
	$(PYTHON) -m pytest

# Clean build artifacts
clean:
	rm -rf tetris_core/target $(RELEASE_MARKER) $(DEV_MARKER)

check:
	$(PYTHON) -m ruff check --fix
	$(PYTHON) -m ruff format
	$(CARGO_ENV) && cargo fix --manifest-path tetris_core/Cargo.toml --lib -p tetris_core --allow-dirty
	$(CARGO_ENV) && cargo fmt --manifest-path tetris_core/Cargo.toml
	$(PYTHON) -m pyright

# Train a model (builds first if needed)
# Usage: make train ARGS="--iterations 10 --games-per-iter 50"
train: $(RELEASE_MARKER)
	@if [ -z "$$TMUX" ]; then \
		echo "Warning: tmux is not active. Training may stop if this terminal closes." >&2; \
	fi
	$(PYTHON) tetris_mcts/train.py $(ARGS)

# Run W&B sweep over learning rate and model size (builds first if needed)
# Usage: make sweep-lr-model ARGS="--count 20"
sweep-lr-model: $(RELEASE_MARKER)
	$(PYTHON) tetris_mcts/scripts/abalations/wandb_sweep_lr_model_size.py $(ARGS)

# Evaluate one model across nn_value_weight values (no training)
# Usage: make eval-nn-value-weight ARGS="--run_dir training_runs/v17 --num_games 50"
eval-nn-value-weight: $(RELEASE_MARKER)
	$(PYTHON) tetris_mcts/scripts/abalations/evaluate_nn_value_weight_sweep.py $(ARGS)

# Compare offline network scaling variants (default, 2x board trunk, 2x post-fusion)
# Usage: make compare-offline-network-scaling ARGS="--data_path training_runs/v32/training_data.npz"
compare-offline-network-scaling: $(RELEASE_MARKER)
	$(PYTHON) tetris_mcts/scripts/abalations/compare_offline_network_scaling.py $(ARGS)

# View replay file
# Usage: make replay FILE=replays.jsonl
FILE ?= replays.jsonl
replay: $(RELEASE_MARKER)
	$(PYTHON) tetris_mcts/scripts/inspection/replay_viewer.py $(FILE)

# Profile game generation performance (builds first if needed)
# Usage: make profile MODEL=benchmarks/models/parallel.onnx SIMS=100 OUTPUT=benchmarks/profile.jsonl
# Usage (dummy/no-network): make profile SIMS=4000 PROFILE_ARGS="--use_dummy_network"
MODEL_PROFILE ?= training_runs/v6/checkpoints/latest.onnx
SIMS ?= 1000
OUTPUT_PROFILE ?= benchmarks/profile_results.jsonl
PROFILE_ARGS ?=
profile: $(RELEASE_MARKER)
	$(PYTHON) tetris_mcts/scripts/inspection/profile_games.py --model_path $(MODEL_PROFILE) --simulations $(SIMS) --output $(OUTPUT_PROFILE) $(PROFILE_ARGS)

# Profile with samply (interactive flamegraph viewer)
# Usage: make profile-samply SIMS=50
# Requires: cargo install samply
profile-samply: $(RELEASE_MARKER)
	samply record $(PYTHON) tetris_mcts/scripts/inspection/profile_games.py --model_path $(MODEL_PROFILE) --simulations $(SIMS) --num_games 3
