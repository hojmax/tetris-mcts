# Memory

Short-term operational memory for this repo. Read this file at the start of every session before substantive work.

## Usage

- Add concise entries liberally when user feedback or hard-won realizations would help future work.
- Focus on final takeaways: corrected command usage, unintuitive behavior, debugging conclusions, hard-to-find facts, and workflow gotchas.
- When this file reaches 200 lines, compact stable notes into themed Markdown files in `memories/`, appending to an existing themed file when it matches.
- After compaction, keep this file as the active short-term log with only current notes and pointers to long-term memory files.

## Notes

- 2026-03-14: User requested a persistent repo memory system. Future sessions should read this file first and record learnings here liberally.
- 2026-03-14: `AGENTS.md` is a symlink to `CLAUDE.md` in this repo, so repo-level agent instruction edits land in `CLAUDE.md`.
- 2026-03-14: Useful test direction: keep a small replay-buffer NPZ fixture in git (around 10k states) and run practical consistency checks against it, not just synthetic/unit-level buffer checks.
- 2026-03-14: Replay-buffer semantic quicktest now lives in `tests/test_replay_buffer_semantics.py`; default fixture path is `tests/fixtures/replay_buffer_quicktest.npz`, with overrides via `TETRIS_REPLAY_BUFFER_QUICKTEST_PATH`, `TETRIS_REPLAY_BUFFER_MAX_PLACEMENTS`, and `TETRIS_REPLAY_BUFFER_PAIR_SAMPLE_LIMIT`.
- 2026-03-14: `tetris_core/src/inference/mod.rs` is a kitchen-sink module: it combines backend dispatch (`tract`/`ort`), split-model inference, board-embedding cache logic, feature encoding utilities reused by runtime/Python debug hooks, and a large inline test module. Much of the apparent duplication is backend-specific API duplication plus separate masked-vs-valid-action entry points for search-performance reasons.
- 2026-03-14: `tetris_core/src/runtime/evaluation.rs` defines `EvalResult.to_dict()`, but there are no in-repo call sites; current training/eval code reads `EvalResult` fields directly or recomputes `eval/*` metrics from `per_game_results`.
- 2026-03-14: The incumbent re-baselining avg-attack loop introduced in commit `3e705094` now lives as shared internal helper `runtime::evaluation::evaluate_avg_attack_on_fixed_seeds`; generator promotion code calls that helper so the stricter fail-closed, shutdown-aware semantics are shared instead of duplicated.
- 2026-03-14: On this macOS checkout, bare `cargo test` in `tetris_core/` can try linking against a missing Xcode `python3.9`; use `PYO3_PYTHON="$(cd .. && pwd)/.venv/bin/python" cargo test ...` from `tetris_core/`.
- 2026-03-14: Candidate promotion eval intentionally overrides `MCTSConfig.seed` to `Some(0)` and forces `visit_sampling_epsilon=0.0`; the exact seed value is not special, because search RNG is derived from `mcts_seed + env.seed + move_number`, so fixed env seeds still give deterministic but per-game/per-move variation.
- 2026-03-14: Removed the unused `output_path` replay-export branch from `evaluate_model` / `evaluate_model_without_nn`; the old JSONL `scripts/inspection/replay_viewer.py` path and `make replay` target were deleted with it.
- 2026-03-14: `tetris_core/src/replay/npz.rs` is repetitive mainly because it streams each `.npy` entry directly into/out of a zip archive with `npyz` + `zip`; higher-level crates like `ndarray-npy`/`ndarray-npz` reduce boilerplate but usually require materializing full arrays, which is a bad trade for large replay snapshots.
- 2026-03-14: `PYO3_PYTHON="$(cd .. && pwd)/.venv/bin/python" cargo test --manifest-path tetris_core/Cargo.toml runtime::evaluation --lib` passed in this checkout after the evaluation cleanup; use that form on macOS instead of bare `cargo test` to avoid linker issues against missing system `python3.9`.
- 2026-03-14: Fixed-seed candidate promotion eval games are benchmark-only and should not be added to replay; the default `model_promotion_eval_games` window is now 20.
- 2026-03-14: `placement_count` is intentionally not stored on `TetrisEnv`; the NN/search treat it as external episode-horizon context (`placements_so_far / max_placements`), not pure game mechanics state. It cannot be replaced by `pieces_spawned` because hold-with-empty-slot spawns a piece without incrementing placement count, and the same env may be searched starting from different placement counts/horizons.
- 2026-03-14: User preference: keep the Rust PyO3 surface minimal. Do not add convenience `get_stats`/`to_dict`/representation-style methods unless the Python side actually uses them; prefer exposing typed fields/methods already consumed in-repo.
- 2026-03-14: `GameGenerator` constructor validation should stay at the PyO3 boundary even though the binding is repo-internal; zero workers and non-finite `save_interval_seconds` still lead to real runtime breakage, so readability refactors should wrap or relocate those checks rather than deleting them.
- 2026-03-14: Inference cleanup refactor: `predict_with_valid_actions` now reuses the shared heads/logits path instead of maintaining a second backend-specific heads implementation, feature encoding shares internal `PieceAuxInputs`/`BoardStatInputs` writers, and inference tests were moved to `tetris_core/src/inference/tests.rs`.
- 2026-03-14: In the `refactor/inference-cleanup` worktree, `PYO3_PYTHON=/Users/axelhojmark/Desktop/tetris-mcts/.venv/bin/python cargo test --manifest-path tetris_core/Cargo.toml --lib inference::tests` passed after the inference refactor; the bare command still tried to link against a missing system `python3.9`.
- 2026-03-14: `make build`/`maturin build` writes generated wheel artifacts into `tetris_core/dist`, and the repo `.gitignore` did not ignore that directory by default; add `tetris_core/dist/` if you want to prevent local wheel files from showing up as untracked/tracked artifacts.
