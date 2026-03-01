# tetris-mcts

AlphaZero-style Tetris with:
- Rust game engine + MCTS (`tetris_core`)
- Python training and scripts (`tetris_bot`)
- PyO3 bindings between Rust and Python

## Quick Start (Linux Container)

```bash
# from repo root
make install
```

`make install` bootstraps:
- `uv` (if missing)
- Rust toolchain via `rustup` (if missing)
- Python dependencies via `uv sync --frozen`
- Debug PyO3 extension via `maturin develop`

Then run scripts with:

```bash
uv run python tetris_bot/scripts/inspection/sweep_num_workers.py
```

`make optimize` behavior:
- If no `training_runs/v*/checkpoints/latest.onnx` split bundle exists yet, it auto-generates a baseline bundle at `benchmarks/models/optimize_bootstrap.onnx` and proceeds.

## Troubleshooting

### `rustc` not found during `uv run python -m maturin ...`

Symptom:
- `maturin failed ... rust compiler is not installed or not in PATH`

Fix:

```bash
source "$HOME/.cargo/env"
which rustc cargo
```

If those paths do not exist, rerun:

```bash
make install
```

### `ImportError: cannot import name 'MCTSConfig' from 'tetris_core'`

This usually means the extension module is not installed in the same environment running the script.

Fix (same shell, repo root):

```bash
source "$HOME/.cargo/env"
unset CONDA_PREFIX
unset VIRTUAL_ENV
uv run python -m maturin develop --manifest-path tetris_core/Cargo.toml
uv run python -c "from tetris_core import MCTSConfig; print(MCTSConfig)"
uv run python tetris_bot/scripts/inspection/sweep_num_workers.py
```

### `Both VIRTUAL_ENV and CONDA_PREFIX are set`

Symptom:
- `maturin failed ... Both VIRTUAL_ENV and CONDA_PREFIX are set. Please unset one of them`

Fix:

```bash
unset CONDA_PREFIX
unset VIRTUAL_ENV
```

Then rerun the `uv run python -m maturin develop ...` command.

### `Failed to set rpath ... did you install patchelf?`

This warning is usually non-fatal (build/install can still succeed), but you can silence it by installing `patchelf`:

```bash
uv run python -m pip install patchelf
```

### `pkg-config` / OpenSSL missing while building `nn-ort`

If ORT build prerequisites are unavailable, run optimize in tract-only mode:

```bash
make optimize OPT_PRIMARY_BACKEND=tract OPTIMIZE_ARGS="--backends tract --skip_build true"
```

### `WARNING: Running pip as the 'root' user ...`

This warning comes from `maturin` invoking `pip` internally and is expected in root containers. If installation completed, it is not a blocker.
