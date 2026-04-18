# tetris-mcts

An AlphaZero-style reinforcement learning system for Tetris. A neural network
guides a Monte Carlo Tree Search that plays the game, generates its own training
data through self-play, and improves by learning from it.

- **Rust** for the game engine, MCTS, and inference hot path.
- **Python / PyTorch** for the network, training loop, and analysis tooling.
- **PyO3 + ONNX** bridge the two: Python trains, Rust plays.

## Quick start

Requires Python 3.12 or 3.13 (not 3.14 — `onnxruntime` has no wheels yet).

```bash
make install     # uv sync + Rust toolchain + debug extension build
make build       # optimized release build of the Rust extension
make play        # interactive Tetris (human play)
make viz         # MCTS tree visualizer
make test        # Rust + Python test suites
make check       # ruff + pyright + rustfmt + clippy
```

`make install` bootstraps `uv`, a Rust toolchain via `rustup`, Python deps, and
a debug PyO3 extension. On Linux it also tries to install ORT build
prerequisites (`pkg-config`, OpenSSL headers, `patchelf`); pass
`AUTO_INSTALL_SYSTEM_DEPS=0` to skip.

## Training a model

```bash
# tune worker/backend/build settings for this machine (cached per-machine)
make optimize

# start training from scratch
make train

# or drive the script directly
uv run python tetris_bot/scripts/train.py --config config.yaml
```

## Documentation

- [`docs/NETWORK_ARCHITECTURE.md`](docs/NETWORK_ARCHITECTURE.md) — exact feature
  layout, tensor shapes, split-export contract.
- [`docs/TRAINING_LEARNINGS.md`](docs/TRAINING_LEARNINGS.md) — notes and
  takeaways from training runs.
- [`paper/paper.pdf`](paper/paper.pdf) - write-up, in progress.
- [`CLAUDE.md`](CLAUDE.md) — for AI assistants working in the repo.
