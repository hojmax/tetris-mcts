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

Python note:
- The repo currently supports Python `3.12` and `3.13`.
- Python `3.14` is not supported yet because `onnxruntime` does not publish compatible wheels there.
- `.python-version` pins the project default to `3.12`, and the project metadata rejects `3.14+`.

`make install` bootstraps:
- `uv` (if missing)
- Rust toolchain via `rustup` (if missing)
- Python dependencies via `uv sync --frozen`
- Debug PyO3 extension via `maturin build` + wheel install
- On Linux, best-effort install of ORT build deps (`pkg-config` + OpenSSL headers) when missing

Then run scripts with:

```bash
uv run python tetris_bot/scripts/inspection/sweep_num_workers.py
```

`make optimize` behavior:
- If no `training_runs/v*/checkpoints/latest.onnx` split bundle exists yet, it auto-generates a baseline bundle at `benchmarks/models/optimize_bootstrap.onnx` and proceeds.

## Resume Training

```bash
uv run python tetris_bot/scripts/train.py --config config.yaml --resume_dir training_runs/v46
uv run python tetris_bot/scripts/train.py --config config.yaml --resume_wandb entity/project/run_id
uv run python tetris_bot/scripts/inspection/download_wandb_training_data.py --reference entity/project/run_id --run_dir training_runs/v2 --overwrite true
```

`--resume_wandb` also accepts a direct artifact ref, for example:
`entity/project/tetris-model-<run_id>:final`.

## Troubleshooting

### `rustc` not found during the Rust extension build

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

### `ModuleNotFoundError: No module named 'tetris_core.tetris_core'`

This usually means the extension module is not installed in the same environment running the script.

Fix (same shell, repo root):

```bash
source "$HOME/.cargo/env"
unset CONDA_PREFIX
unset VIRTUAL_ENV
make build-dev
.venv/bin/python -c "from tetris_core.tetris_core import MCTSConfig; print(MCTSConfig)"
.venv/bin/python tetris_bot/scripts/inspection/sweep_num_workers.py
```

If this happens specifically during `make optimize` in a root/container shell, use an env-sanitized rebuild and keep Cargo on `PATH`:

```bash
source "$HOME/.cargo/env" 2>/dev/null || export PATH="$HOME/.cargo/bin:$PATH"
which rustc cargo
rustc --version
cargo --version

env -u CONDA_PREFIX -u VIRTUAL_ENV -u PYTHONPATH PATH="$HOME/.cargo/bin:$PATH" \
  PYO3_PYTHON="$PWD/.venv/bin/python" .venv/bin/python -m maturin build --release --out tetris_core/dist --features extension-module --manifest-path tetris_core/Cargo.toml

env -u CONDA_PREFIX -u VIRTUAL_ENV -u PYTHONPATH PATH="$HOME/.cargo/bin:$PATH" \
  VIRTUAL_ENV="$PWD/.venv" .venv/bin/python -m uv pip install --no-deps --reinstall tetris_core/dist/tetris_core-*.whl

env -u CONDA_PREFIX -u VIRTUAL_ENV -u PYTHONPATH PATH="$HOME/.cargo/bin:$PATH" \
  .venv/bin/python tetris_bot/scripts/inspection/optimize_machine.py \
  --num_games 20 --simulations 300 --num_repeats 1 \
  --worker_search adaptive --max_worker_evals_per_combo 6 \
  --backend_strategy staged --primary_backend tract --backends tract \
  --worker_candidates 2 32 64 128 256 512 --skip_build true
```

### `Both VIRTUAL_ENV and CONDA_PREFIX are set`

Symptom:
- `maturin failed ... Both VIRTUAL_ENV and CONDA_PREFIX are set. Please unset one of them`

Fix:

```bash
unset CONDA_PREFIX
unset VIRTUAL_ENV
```

Then rerun `make build-dev`.

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

`make train` also auto-falls back to `tract` if an optimized ORT build fails.
To skip system package auto-install during bootstrap: `make install AUTO_INSTALL_SYSTEM_DEPS=0`.
`make train` validates optimize cache machine fingerprint and refreshes cache if it was generated on a different machine.

### `WARNING: Running pip as the 'root' user ...`

This warning comes from `maturin` invoking `pip` internally and is expected in root containers. If installation completed, it is not a blocker.

### `onnxruntime` can't be installed on Python 3.14

Symptom:
- `uv sync --frozen` fails with an error saying `onnxruntime` has no wheel for `cp314`

Cause:
- The repo depends on `onnxruntime`, which currently supports Python `3.12` and `3.13`, but not `3.14`.

Fix:

```bash
uv python install 3.12
uv venv --python 3.12 .venv
uv sync --frozen
```
