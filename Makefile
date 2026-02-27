.PHONY: run install ensure-rust build build-dev clean rebuild test check play viz train replay profile profile-samply sweep-lr-model eval-nn-value-weight compare-offline-network-scaling sweep-mcts-config

# Source cargo environment if available
SHELL := /bin/bash
CARGO_ENV := source $$HOME/.cargo/env 2>/dev/null || true
VENV_DIR := .venv
PYTHON := $(VENV_DIR)/bin/python
PYTHON_ABS := $(abspath $(PYTHON))
INSTALL_MARKER := $(VENV_DIR)/.install_marker

# Find all Rust source files (including subdirectories)
RUST_SRC := $(shell find tetris_core/src -name '*.rs')
RELEASE_MARKER := .build_marker_release
DEV_MARKER := .build_marker_dev

# Bootstrap project dependencies into local virtualenv with uv.
$(INSTALL_MARKER): pyproject.toml uv.lock
	@set -euo pipefail; \
	if command -v uv >/dev/null 2>&1; then \
		UV_BIN="$$(command -v uv)"; \
	elif [ -x "$$HOME/.local/bin/uv" ]; then \
		UV_BIN="$$HOME/.local/bin/uv"; \
	else \
		if ! command -v curl >/dev/null 2>&1; then \
			echo "Error: uv is missing and curl is required to install it." >&2; \
			exit 1; \
		fi; \
		echo "uv not found; installing to $$HOME/.local/bin..."; \
		curl -LsSf https://astral.sh/uv/install.sh | sh; \
		if [ -x "$$HOME/.local/bin/uv" ]; then \
			UV_BIN="$$HOME/.local/bin/uv"; \
		elif command -v uv >/dev/null 2>&1; then \
			UV_BIN="$$(command -v uv)"; \
		else \
			echo "Error: uv installation failed." >&2; \
			exit 1; \
		fi; \
	fi; \
	"$$UV_BIN" sync --frozen
	@touch $(INSTALL_MARKER)

ensure-rust:
	@set -euo pipefail; \
	if command -v rustc >/dev/null 2>&1 && command -v cargo >/dev/null 2>&1; then \
		exit 0; \
	fi; \
	if [ -f "$$HOME/.cargo/env" ]; then \
		source "$$HOME/.cargo/env"; \
	fi; \
	if command -v rustc >/dev/null 2>&1 && command -v cargo >/dev/null 2>&1; then \
		exit 0; \
	fi; \
	if ! command -v curl >/dev/null 2>&1; then \
		echo "Error: rustc/cargo are missing and curl is required to install rustup." >&2; \
		exit 1; \
	fi; \
	echo "Rust toolchain not found; installing via rustup..."; \
	curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain stable; \
	source "$$HOME/.cargo/env"; \
	rustc --version >/dev/null; \
	cargo --version >/dev/null

install: ensure-rust $(INSTALL_MARKER)

# Build marker file to track if build is up to date (release mode)
$(RELEASE_MARKER): ensure-rust $(INSTALL_MARKER) $(RUST_SRC) tetris_core/Cargo.toml tetris_core/pyproject.toml
	$(CARGO_ENV) && $(PYTHON) -m maturin develop --release --manifest-path tetris_core/Cargo.toml
	@rm -f $(DEV_MARKER)
	@touch $(RELEASE_MARKER)

# Explicit build target
build: $(RELEASE_MARKER)

# Fast debug build (much faster, for development only)
build-dev: ensure-rust $(INSTALL_MARKER)
	$(CARGO_ENV) && $(PYTHON) -m maturin develop --manifest-path tetris_core/Cargo.toml
	@rm -f $(RELEASE_MARKER)
	@touch $(DEV_MARKER)

# Run the game (builds first if needed)
play: $(RELEASE_MARKER)
	$(PYTHON) tetris_bot/scripts/tetris_game.py

# Run the MCTS visualizer (builds first if needed)
# Usage: make viz RUN_DIR=training_runs/v15
# Usage (dummy/no-network): make viz RUN_DIR=training_runs/v15 DUMMY_NETWORK=1
RUN_DIR ?= training_runs/v41
DUMMY_NETWORK ?= 0
viz: $(RELEASE_MARKER)
	$(PYTHON) tetris_bot/scripts/inspection/mcts_visualizer.py $(if $(RUN_DIR),--run_dir $(RUN_DIR),) $(if $(filter 1 true TRUE yes YES,$(DUMMY_NETWORK)),--use_dummy_network true,)

# Force rebuild (clean first to avoid caching issues)
rebuild: ensure-rust $(INSTALL_MARKER)
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
		echo "Error: tmux is not active. Training may stop if this terminal closes. Run inside tmux." >&2; \
		exit 1; \
	fi
	$(PYTHON) tetris_bot/scripts/train.py $(ARGS)

# Run W&B sweep over learning rate and model size (builds first if needed)
# Usage: make sweep-lr-model ARGS="--count 20"
sweep-lr-model: $(RELEASE_MARKER)
	$(PYTHON) tetris_bot/scripts/ablations/wandb_sweep_lr_model_size.py $(ARGS)

# Evaluate one model across nn_value_weight values (no training)
# Usage: make eval-nn-value-weight ARGS="--run_dir training_runs/v17 --num_games 50"
eval-nn-value-weight: $(RELEASE_MARKER)
	$(PYTHON) tetris_bot/scripts/ablations/evaluate_nn_value_weight_sweep.py $(ARGS)

# Sweep an MCTS config parameter (e.g. q_scale, nn_value_weight, c_puct) over multiple values
# Usage: make sweep-mcts-config ARGS="--run_dir training_runs/v32 --sweep_param q_scale --sweep_values '[2,4,8,16,32]'"
sweep-mcts-config: $(RELEASE_MARKER)
	$(PYTHON) tetris_bot/scripts/ablations/sweep_mcts_config.py $(ARGS)

# Compare offline network scaling variants (default, 2x board trunk, 2x post-fusion)
# Usage: make compare-offline-network-scaling ARGS="--data_path training_runs/v32/training_data.npz"
compare-offline-network-scaling: $(RELEASE_MARKER)
	$(PYTHON) tetris_bot/scripts/ablations/compare_offline_network_scaling.py $(ARGS)

# View replay file
# Usage: make replay FILE=replays.jsonl
FILE ?= replays.jsonl
replay: $(RELEASE_MARKER)
	$(PYTHON) tetris_bot/scripts/inspection/replay_viewer.py $(FILE)

# Profile game generation performance (builds first if needed)
# Usage: make profile MODEL=benchmarks/models/parallel.onnx SIMS=100 OUTPUT=benchmarks/profile.jsonl
# Usage (dummy/no-network): make profile SIMS=4000 PROFILE_ARGS="--use_dummy_network"
MODEL_PROFILE ?= training_runs/v6/checkpoints/latest.onnx
SIMS ?= 1000
OUTPUT_PROFILE ?= benchmarks/profile_results.jsonl
PROFILE_ARGS ?=
profile: $(RELEASE_MARKER)
	$(PYTHON) tetris_bot/scripts/inspection/profile_games.py --model_path $(MODEL_PROFILE) --simulations $(SIMS) --output $(OUTPUT_PROFILE) $(PROFILE_ARGS)

# Profile with samply (interactive flamegraph viewer)
# Usage: make profile-samply SIMS=50
# Requires: cargo install samply
profile-samply: $(RELEASE_MARKER)
	samply record $(PYTHON) tetris_bot/scripts/inspection/profile_games.py --model_path $(MODEL_PROFILE) --simulations $(SIMS) --num_games 3
