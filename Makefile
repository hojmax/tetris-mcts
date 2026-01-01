.PHONY: run build clean rebuild

# Source cargo environment if available
SHELL := /bin/bash
CARGO_ENV := source $$HOME/.cargo/env 2>/dev/null || true

# Run the game (builds first if needed)
run: .build_marker
	uv run python tetris_game.py

# Build marker file to track if build is up to date
.build_marker: tetris_core/src/lib.rs tetris_core/Cargo.toml tetris_core/pyproject.toml
	$(CARGO_ENV) && uv run maturin develop --release --manifest-path tetris_core/Cargo.toml
	@touch .build_marker

# Explicit build target
build: .build_marker

# Force rebuild and run
rebuild:
	$(CARGO_ENV) && uv run maturin develop --release --manifest-path tetris_core/Cargo.toml
	@touch .build_marker
	uv run python tetris_game.py

# Clean build artifacts
clean:
	rm -rf tetris_core/target .build_marker
