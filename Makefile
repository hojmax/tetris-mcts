.PHONY: run build build-dev clean rebuild test check play viz train evaluate replay profile profile-samply

# Source cargo environment if available
SHELL := /bin/bash
CARGO_ENV := source $$HOME/.cargo/env 2>/dev/null || true
PYTHON := .venv/bin/python

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
viz: $(RELEASE_MARKER)
	$(PYTHON) tetris_mcts/scripts/mcts_visualizer.py

# Force rebuild (clean first to avoid caching issues)
rebuild:
	cd tetris_core && $(CARGO_ENV) && cargo clean
	$(CARGO_ENV) && $(PYTHON) -m maturin develop --release --manifest-path tetris_core/Cargo.toml
	@rm -f $(DEV_MARKER)
	@touch $(RELEASE_MARKER)

# Run tests
test:
	cd tetris_core && $(CARGO_ENV) && cargo test

# Clean build artifacts
clean:
	rm -rf tetris_core/target $(RELEASE_MARKER) $(DEV_MARKER)

check:
	uv run ruff check --fix
	uv run ruff format
	$(CARGO_ENV) && cargo fix --manifest-path tetris_core/Cargo.toml --lib -p tetris_core --allow-dirty
	$(CARGO_ENV) && cargo fmt --manifest-path tetris_core/Cargo.toml
	uv run pyright

# Train a model (builds first if needed)
# Usage: make train ARGS="--iterations 10 --games-per-iter 50"
train: $(RELEASE_MARKER)
	$(PYTHON) tetris_mcts/train.py $(ARGS)

# Evaluate a model and save replays (builds first if needed)
# Usage: make evaluate MODEL=checkpoints/latest.onnx OUTPUT=replays.jsonl
MODEL ?= checkpoints/latest.onnx
OUTPUT ?= replays.jsonl
evaluate: $(RELEASE_MARKER)
	$(PYTHON) tetris_mcts/scripts/evaluate.py --model-path $(MODEL) --output-path $(OUTPUT)

# View replay file
# Usage: make replay FILE=replays.jsonl
FILE ?= replays.jsonl
replay: $(RELEASE_MARKER)
	$(PYTHON) tetris_mcts/scripts/replay_viewer.py $(FILE)

# Profile game generation performance (builds first if needed)
# Usage: make profile MODEL=benchmarks/models/parallel.onnx SIMS=100 OUTPUT=benchmarks/profile.jsonl
MODEL_PROFILE ?= benchmarks/models/parallel_small.onnx
SIMS ?= 100
OUTPUT_PROFILE ?= benchmarks/profile_results.jsonl
profile: $(RELEASE_MARKER)
	$(PYTHON) tetris_mcts/scripts/profile_games.py --model_path $(MODEL_PROFILE) --simulations $(SIMS) --output $(OUTPUT_PROFILE)

# Profile with samply (interactive flamegraph viewer)
# Usage: make profile-samply SIMS=50
# Requires: cargo install samply
profile-samply: $(RELEASE_MARKER)
	samply record $(PYTHON) tetris_mcts/scripts/profile_games.py --model_path $(MODEL_PROFILE) --simulations $(SIMS) --num_games 3
