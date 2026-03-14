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
- 2026-03-14: `tetris_core/src/inference/mod.rs` is a kitchen-sink module: it combines backend dispatch (`tract`/`ort`), split-model inference, board-embedding cache logic, feature encoding utilities reused by runtime/Python debug hooks, and a large inline test module. Much of the apparent duplication is backend-specific API duplication plus separate masked-vs-valid-action entry points for search-performance reasons.
- 2026-03-14: `tetris_core/src/runtime/evaluation.rs` defines `EvalResult.to_dict()`, but there are no in-repo call sites; current training/eval code reads `EvalResult` fields directly or recomputes `eval/*` metrics from `per_game_results`.
- 2026-03-14: On this macOS checkout, bare `cargo test` in `tetris_core/` can try linking against a missing Xcode `python3.9`; use `PYO3_PYTHON="$(cd .. && pwd)/.venv/bin/python" cargo test ...` from `tetris_core/`.
