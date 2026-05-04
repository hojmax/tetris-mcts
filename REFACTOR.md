# Refactor

## Points of improvement

- Remove old bloat
  - Tetris environment has a lot of old code and options that are no longer used.
  - Like the manual playing of the environment, drop all of that.
  - In the python code, we drop all of the scripts and settings and random nonsense that isnt used.
  - There are also soooo many checks and asserts, where we could just let it fail naturally. Especially on inputs. 
  - I want better thought out classes with clear responsibilites.
  - Better reuse of code, like the generator worker should be a thing.
  - Also the makefile is horrendous, I want clean simple commands that install and build and find optimal settings for the machine, thats it.
  - So we drop all the scripts essentially, and most of makefile is gone, and I don't want so much damn makefile boiler plate. Maybe we could move that logic into python scripts or something more suitable?
  - Also there are all sorts of strange functions like is_friendly_run_id. That is just used for testing?? But like obviously not needed? Like by construction it is a valid run id, so it should be valid.
  - Just like very very simple overall, we need to cut the bloat.
  - Also massive functions like apply_checkpoint_search_overrides. This should be like a class that handles settings or something. Things should be thought through explcitly and cleanly as classes with seperate responsibilites.
  - So screw all of ablations and inspection and scripts.
  - Also all the candidate gating logic with like storing params on the models should be dropped now in favor of the step based logic.
  - Also like cleaner way of parallelizing the game generation, so they all funnel in exactly like the main thread. I.e. I don't want patches and hacks to have the remote games be treated the same way as on machine games, they should all funnel in and be treated the same, in terms of logging and metrics and so on.
  - Get rid of all the candidate gating.
  - And like you really really dont need all that input validation like in NNValueWeightScheduleConfig. Like we are not taking in arbitrary inputs, we can just fail if something is set wrong the usual way! no need to verify all those things.
  - Also there are a lot of complicated metrics in the Rust that we could probably drop, like the rmse and variance along the trajectory and others. we should list out all of the different metrics, and then together decide which are actually useful and which are not.
  - Also things are duplicated like is build_aux_features used only in a script?
  - Also like the put everything on gpu idea is maybe not great, because then on smaller gpus, I am limited in terms of replay buffer size. So maybe we should just load in from cpu, is that much slower?
  - There should never be files that are like 1000+ lines long like the trainer.
  - Drop all backwards compatibility code. Like legacy actions and masks and so on. Like we should just have a fresh start, and no bloat from various formats etc.
  - Also think through if there would be a better format for saving and loading the replay buffers, that would make syncing faster for example or something.
  - Also tetris_bot is a not great name. Suggest alternatives.
  - Also the file organisation is not great in tetris_bot, with random top level files, and then just a big dump in tetris_bot/ml.
  - Drop all the gated schedueles.
  - I also want a runtime override for replay buffer size. if smaller, drop oldest. If larger, just larger.
  - Also the syncing of params to the generators via r2 should not be an afterthought, but done properly.
  - Cleaner model syncs with like a .yaml that is associated with each model with the relevant information, instead of putting it on the model itself. And like currently, it generates a ton of different files, wondering if we could zip them or if that would be cleaner?

---

## Big Plan

This section is the concrete plan. It maps every bullet in "Points of improvement" above to specific files, classes, deletions, and replacement structures. Read top-down: principles, target tree, then per-area breakdown with what gets deleted, what gets kept, and what gets rewritten.

### 0. Guiding principles (apply to every section below)

- One implementation path. No backwards-compat for internal formats. Replay buffer files written before this refactor are explicitly **not** loadable on the other side; old `training_runs/v*/` are read-only artifacts.
- No "legacy" anything in the new tree. If the word "legacy" survives the refactor, it's a bug.
- No defensive validation of *internal* values (config dataclasses, function args from our own code). Validate only at the system boundary: user-typed CLI args, files on disk, network bytes. Anywhere else, trust types and fail naturally.
- Files target ~300 lines, hard ceiling ~500. The trainer (currently 3073) and `r2_sync.py` (1182) are the worst offenders and must be split.
- One responsibility per class. The trainer today owns ~76 methods spanning self-play, R2, candidate gating, mirroring, gif rendering, checkpointing. Each of those becomes a separate component the trainer composes.
- Trust internal Rust APIs. Python-side wrappers should not re-validate values that Rust already constrains.
- Remote and local generators **funnel through the same path**. There is no "remote-only" or "trainer-only" code path; both are instances of the same `Generator` running with the same config snapshot.

### 1. Top-level rename and tree reorganization

`tetris_bot` is misleading — it's not a "bot," it's the AlphaZero training harness. The new package is **`tetris_mcts`**, paired with `tetris_core`. Note that this matches the repo basename (`tetris-mcts/`), which is conventional in Python (`scikit-learn` repo / `sklearn` package, `python-pillow` / `PIL`, etc.) — directories use `-`, packages use `_`.

Target tree (after refactor):

```text
tetris_core/                     # Rust crate — engine, search, runtime, replay, inference (largely unchanged)
tetris_viz/                      # NEW Rust crate — fast incremental MCTS tree visualizer (egui/eframe). See §15B.
tetris_mcts/
├── __init__.py
├── constants.py                 # board geometry, piece colors, action counts (unchanged)
├── action_space.py              # canonical-cell mappings only; legacy adapters DELETED
├── config/
│   ├── __init__.py              # re-exports TrainingConfig, RuntimeOverrides
│   ├── training.py              # NetworkConfig, OptimizerConfig, SelfPlayConfig, ReplayConfig, RunConfig, TrainingConfig
│   ├── runtime_overrides.py     # RuntimeOverrides + the resolved/applied dataclasses
│   └── snapshot.py              # SelfPlaySnapshot (the trainer-authoritative R2 snapshot)
├── runtime/                     # Python orchestration — generator loop, sync, lifecycle
│   ├── generator.py             # thin Python wrapper around Rust GameGenerator (single class used by both train and remote)
│   ├── completed_game_stream.py # unified queue: local + remote completed games funnel into one iterator
│   ├── runtime_overrides.py     # file-watcher + applier (replaces ~250 lines in trainer.py)
│   └── shutdown.py              # SIGINT-deferred shutdown helper
├── sync/                        # R2 sync, split out of the 1182-line r2_sync.py
│   ├── settings.py              # R2Settings, env-var sourcing
│   ├── client.py                # S3 client factory + low-level helpers
│   ├── keys.py                  # all `<prefix>/<run>/...` key construction
│   ├── replay_chunks.py         # ChunkUploader + ChunkDownloader
│   ├── game_stats.py            # GameStatsUploader + GameStatsDownloader
│   ├── model_bundle.py          # upload_model_bundle, download_model_bundle, ModelDownloader, ModelPointer
│   ├── self_play_snapshot.py    # publish + fetch SelfPlaySnapshot
│   └── discovery.py             # discover_active_runs (auto-pick run id)
├── ml/
│   ├── network.py               # TetrisNet + ResidualConvBlock + ResidualFusionBlock + AUX layout (kept; build_aux_features moved to inference helpers if even kept)
│   ├── loss.py                  # compute_loss + RunningLossBalancer (no validators on internal masks)
│   ├── optimizer.py             # OptimizerBundle + scheduler factory
│   ├── ema.py                   # ExponentialMovingAverage (small, kept)
│   ├── replay_mirror.py         # CircularReplayMirror + TrainingBatch (renamed from replay_buffer.py — there is no Python "replay buffer", only a device mirror of the Rust ring)
│   └── policy_mirroring.py      # mirror tensors helpers; legacy_*_to_flat helpers DELETED
├── train/
│   ├── trainer.py               # ~300 lines: composes the components below, runs the loop
│   ├── train_step.py            # forward/backward/log a single step (extracted from current `train_step`)
│   ├── checkpoint.py            # CheckpointManager: capture_snapshot, save, load, async writer (replaces weights.py + AsyncCheckpointSaver)
│   ├── onnx_export.py           # export_onnx, export_split_models, _export_cached_board_path_binary
│   ├── replay_pipeline.py       # ReplayPipeline: owns mirror, prefetch, sample → TrainingBatch (replaces ~600 lines of trainer mirror plumbing)
│   ├── completed_game_logger.py # CompletedGameLogger: drains generator + remote streams, logs per-game W&B, manages GIFs
│   ├── promotion.py             # PromotionScheduler: schedules `sync_model_directly`, applies new bundle to generator (no candidate gating!)
│   └── wandb_setup.py           # configure_wandb + define_metric calls (kept small)
├── entrypoints/
│   ├── train.py                 # ~80 lines: parse args, build config, build Trainer, trainer.train() (handles --offline_only)
│   ├── generator.py             # ~80 lines: parse args, fetch snapshot, build Generator, run until SIGINT
│   └── eval.py                  # ~60 lines: build deterministic Generator, drain N games, print aggregates
├── visualization/
│   ├── render.py                # render_board, render_replay (used by the trainer for W&B GIFs)
│   ├── overlay.py               # PredictedMoveOverlay + label placement helpers
│   └── gif.py                   # create_trajectory_gif
└── cli.py                       # Typer/simple_parsing CLI: `python -m tetris_mcts <subcommand>` (replaces most of the Makefile)

tools/                           # was tetris_bot/scripts; massively trimmed (see §11)
└── optimize_machine.py          # only inspection script that survives, used by `tetris_mcts optimize`

tests/                           # trimmed (see §10)
```

### 2. Build/install/CLI: kill the Makefile

Today: 380-line Makefile with 25+ phony targets, each repeating environment-bootstrap shell. Most targets exist for one-off ablations (`sweep-lr-model`, `eval-nn-value-weight`, `compare-offline-network-scaling`, `compare-offline-spatial-policy`, `compare-warm-start-trunk-sizes`, `loss-sensitivity`, `download-wandb-training-data`, `viz-policy-grid`, `profile-samply`, `rebuild`, `build-ort`, `play`).

After: a single `tetris_mcts` Typer CLI plus a minimal `Makefile` (or, better, a `justfile` / `pyproject.toml [project.scripts]` entry — no Makefile at all).

Surviving commands, all in Python:

| Command           | What it does                                              |
| ----------------- | --------------------------------------------------------- |
| `tetris_mcts install` | uv sync + `maturin build --release` + install wheel       |
| `tetris_mcts build` | `maturin build --release` + reinstall (debug via `--dev`) |
| `tetris_mcts check` | ruff + pyright + cargo fmt + cargo fix                    |
| `tetris_mcts test`  | pytest + `cargo test`                                     |
| `tetris_mcts optimize` | machine-fingerprint sweep, write env file              |
| `tetris_mcts train` | calls `optimize` if cache missing, then runs the trainer (accepts `--offline_only`, `--init_checkpoint`, `--resume_dir`) |
| `tetris_mcts generator` | remote-machine entrypoint                              |
| `tetris_mcts eval` | run the generator with deterministic config, drain N games, aggregate; replaces `evaluate_model` |
| `tetris_mcts viz` | shell out to the `tetris_viz` Rust binary (live or replay mode). See §15B. |

Deleted entirely: `play`, `viz`, `viz-policy-grid`, `sweep-lr-model`, `eval-nn-value-weight`, `sweep-mcts-config`, `compare-*`, `loss-sensitivity`, `warm-start`, `download-wandb-training-data`, `profile`, `profile-samply`, `rebuild`, `build-ort` (ORT becomes an installer flag, not a separate target). The system-deps bootstrap (`ensure-system-deps`) becomes a Python helper called by `install` only on Linux.

The tmux check survives (it's saved many a long run).

### 3. Drop the candidate gating system entirely

This is one of the largest deletions. Per the user: "Get rid of all the candidate gating."

Today candidate gating sprawls across:

- Rust: `runtime/game_generator/runtime.rs` (eval-worker arm of the worker loop), `shared.rs` (`CandidateModelRequest`, `ModelEvalEvent`, `pending_candidate`, `evaluating_candidate`, eval seeds, `incumbent_eval_avg_attack`), `py_api.rs` (`queue_candidate_model`, `drain_model_eval_events`, `candidate_gate_busy`, `model_promotion_eval_games`-related ctor args), and the worst-eval-tree persistence in `runtime.rs`.
- Python: `Trainer._candidate_gate_*` (8+ methods), `CandidateGateSchedule`, `_drain_model_eval_events`, `_initialize_candidate_gate_schedule`, `_update_candidate_gate_schedule_from_eval`, `_defer_candidate_gate_export`, `_evaluate_starting_incumbent_avg_attack`, the candidate-gate columns on every checkpoint, the `force_promote_next_candidate` runtime override, and the `model_promotion_eval_games` / `save_eval_trees` / `bootstrap_*` config fields that exist only for the gating path.

After: there is one path. The trainer exports a fresh ONNX bundle on a fixed step-based interval (or seconds-based; see §5) and pushes it to the generator via `sync_model_directly`. No evaluator worker, no fixed-seed eval, no auto-vs-force promotion logic, no streak-based backoff.

Concretely:

- **Rust deletions:** `pending_candidate`, `evaluating_candidate`, `model_eval_events`, `incumbent_eval_avg_attack`, `candidate_eval_seeds`, `candidate_gating_enabled`, `save_eval_trees`, `non_network_num_simulations`, `bootstrap_without_network` plumbing, the evaluator-worker branch of `worker_loop`, `ModelEvalEvent`, `CandidateModelRequest`, the worst-tree persistence path. `GameGenerator::new` shrinks from 14 args to ~6.
- **Config deletions:** `SelfPlayConfig.use_candidate_gating`, `model_promotion_eval_games`, `bootstrap_without_network`, `bootstrap_num_simulations`, `save_eval_trees`. `RunConfig.model_sync_failure_backoff_seconds`, `model_sync_max_interval_seconds`. `RuntimeSelfPlayOverrides.force_promote_next_candidate`.
- **Python deletions:** `CandidateGateSchedule`, all `_candidate_gate_*` methods in Trainer, `_drain_model_eval_events`, `_evaluate_starting_incumbent_avg_attack`, `tests/test_trainer_candidate_gate_schedule.py`, `tests/test_nn_value_weight_promotion.py` (rewritten), `tests/test_resume_checkpoint_state.py` (the gate-state assertions go away).
- **Checkpoint deletions:** `incumbent_eval_avg_attack`, `incumbent_uses_network`, `candidate_gate_*` keys, the giant `_runtime_override_checkpoint_state` integer hoisting that exists because resume state was being smuggled through string keys instead of a dataclass.

### 4. Replace per-model-stamped state with step-based truth

User: "Also all the candidate gating logic with like storing params on the models should be dropped now in favor of the step based logic."

Today the Rust `GameGenerator` carries 8 atomics for state that "belongs to the incumbent": `incumbent_nn_value_weight`, `incumbent_death_penalty`, `incumbent_overhang_penalty_weight`, `incumbent_uses_network`, `incumbent_model_step`, etc. Each promotion/sync atomically updates all of them so workers read a consistent set.

After: the only thing the generator stores about a model is `model_path` and `model_step`. Penalties and `nn_value_weight` are pure functions of the trainer's *current step* and the live config, computed every game by the worker from a single shared `LiveSearchSettings` snapshot (one `Arc<RwLock<LiveSearchSettings>>` instead of 8 atomics). Schedule values come from the trainer-published `SelfPlaySnapshot`.

Effect on shutdown/resume: the gigantic `restore_trainer_from_checkpoint` (~380 lines) collapses. It reads `step`, the optimizer/scheduler state, the runtime overrides struct (one nested `RuntimeOverrides` blob, not 14 flat keys), and that's it.

### 5. Step-based promotion replaces seconds-based promotion

User implication of "step based logic": tie model sync to optimizer step rather than wall clock so it is deterministic across machines, robust to compile/optimize stalls, and easy to reason about during resume.

- New config field `RunConfig.model_sync_step_interval: int` (default **1000**). Replaces `model_sync_interval_seconds`, `model_sync_failure_backoff_seconds`, `model_sync_max_interval_seconds`.
- A `PromotionScheduler` component owns the export+sync. After every train step it checks `step % interval == 0`; if so, exports ONNX (still asynchronous), pushes to local generator via `sync_model_directly`, and uploads to R2.
- On the generator side (local or remote), there is one polling loop in `ModelDownloader` that re-applies the latest pointer. No backoff, no streak logic.

### 6. Unified generator, unified completed-game stream

User: "cleaner way of parallelizing the game generation, so they all funnel in exactly like the main thread. I.e. I don't want patches and hacks to have the remote games be treated the same way as on machine games."

Today: `Trainer._drain_completed_games` (local) and `Trainer.push_remote_completed_games` (called by `GameStatsDownloader`) feed two parallel paths with different shapes (Rust `LastGameInfo` dict vs JSON-roundtripped dict), and the trainer special-cases each.

After: a single `CompletedGameStream` owns one queue. Both producers push `CompletedGame` (a Pydantic model with the canonical schema). `LastGameInfo.to_dict` in Rust and the JSON round-trip both target this one schema. Trainer logging code reads exactly one queue.

```python
class CompletedGame(BaseModel):
    game_number: int
    completed_at_s: float          # generator-local monotonic time, normalized at sink
    machine_id: str                # "trainer-local" for in-process, hostname for remote
    stats: GameStats               # named fields, not Dict[str, float]
    replay: GameReplay | None      # only present for in-process games
```

Same model serializes to JSON for R2 and is the in-memory type the trainer sees.

### 7. Replay buffer rework + runtime size override

User: "I also want a runtime override for replay buffer size. if smaller, drop oldest. If larger, just larger."

- Default `replay.buffer_size` raised to **10_000_000** (today: 3_250_000 in the committed config, 7_000_000 in the CLAUDE.md mention). With CPU-resident sampling as the new default (below), buffer size is no longer VRAM-bounded.
- Add `replay.buffer_size` to the `RuntimeOverrides` whitelist. The trainer-side override applies in two places: (a) Rust `SharedBuffer.max_size` is mutable via a new `set_max_size(usize)` method on `GameGenerator`. Shrinking pops front entries until size ≤ new max. Growing is a no-op until new examples land. (b) `CircularReplayMirror` resizes by reallocation (rare, gated).
- Drop `ReplayConfig.replay_mirror_delta_chunk_examples` and `replay_mirror_refresh_seconds` if the new pipeline can hold a single fixed cadence — investigate.
- `replay.mirror_replay_on_accelerator` default flips to **`false`**: the mirror lives in pinned CPU memory, batches are transferred non-blocking per step. Removes the VRAM ceiling on `buffer_size`. Exposed as a runtime override so it can be flipped to `true` on machines with VRAM headroom.
- Investigate replay format: today it's `training_data.npz` with seven separately-zlibbed arrays. R2 chunks are smaller NPZs of the same shape. Consider switching to **uncompressed NPZ + Zstd at the upload boundary** (faster CPU, similar ratio for sparse policy targets) or **Parquet/Arrow** if we want to scan slices without full materialization. For now: keep NPZ but remove ZIP64 special-casing once the format is uncompressed (large files become naturally chunked).

### 8. Schedules: kill `gated`, simplify `nn_value_weight`

User: "Drop all the gated schedueles."

- `PenaltyScheduleConfig.strategy` is reduced to `"constant_then_linear"` only — make it the only path, drop the `strategy` field.
- `NNValueWeightScheduleConfig.strategy = "per_promotion"` is deleted; `per_games_interval` is the only path. Drop `strategy` field. The class collapses to `{games_interval, initial, multiplier, max_delta, cap}`.
- All the validators on these dataclasses (the `model_validator(mode="after")` blocks) are deleted. We control the inputs; if `cap < initial` we want to crash naturally on the next ramp computation, not pre-emptively in pydantic.
- `compute_penalty_scale` and `compute_nn_value_weight` collapse to ~5-line functions each. The `PenaltyScheduleStrategy` and `NNValueWeightScheduleStrategy` `Literal` types are deleted.

### 9. Aux features and network: remove dead code

- `tetris_bot/ml/aux_features.py` has both `AuxFeatureLayout` (used) and is referenced by tests and metrics — keep.
- `network.build_aux_features` is used only in tests + one inspection script (`value_predictor.py`). Decision: move to a small `inference_helpers.py` (since it's strictly Python-side feature construction for tests), or **delete** if the inspection script also goes (see §11). If `value_predictor.py` is dropped, `build_aux_features` is dropped.
- The `ConvBackbone` and `HeadsModel` classes (split-export wrappers in `network.py`) only exist for ONNX split export; they belong in `train/onnx_export.py` next to `export_split_models`, not in `network.py`.

### 10. Action space: drop legacy adapters

- Delete `adapt_legacy_policy_targets`, `adapt_legacy_action_masks`, `_build_legacy_action_positions`, the legacy constants in `action_space.py`.
- Delete the Rust `adapt_legacy_policy_and_mask` in `tetris_core/src/replay/npz.rs` and the related test (`test_adapt_legacy_policy_and_mask_collapses_redundant_rotations`).
- Delete `legacy_action_masks_to_flat`, `legacy_policy_targets_to_flat`, and the `_build_legacy_to_flat_maps` helper in `policy_mirroring.py`.
- Delete `compare_offline_architectures.py` (its sole remaining user) and the corresponding test.
- After this, `action_space.py` is a flat list of canonical-cell + piece-mapping helpers — no "v1 vs v2" branching anywhere. Replay snapshots written before this refactor are unloadable; document this explicitly in the migration note.

### 11. Tests: massive trim

Today: 31 test files, ~7000 lines. Most cover code paths that this refactor deletes. Survival list:

| Test file                                | Keep? | Notes                                                          |
| ---------------------------------------- | ----- | -------------------------------------------------------------- |
| `test_action_space.py`                   | Trim  | Delete legacy-adapter cases; keep canonical-cell tests         |
| `test_artifacts.py`                      | Keep  | ONNX bundle layout — still relevant                            |
| `test_async_checkpoint_saver.py`         | Keep  | Becomes `test_checkpoint.py` after the rename                  |
| `test_compare_warm_start_trunk_sizes.py` | DROP  | Test for a deleted ablation                                    |
| `test_download_wandb_training_data.py`   | DROP  | Inspection script removed                                      |
| `test_game_metrics.py`                   | Keep  | Still useful                                                   |
| `test_loss_balancer.py`                  | Keep  |                                                                |
| `test_loss.py`                           | Keep  |                                                                |
| `test_network_config.py`                 | Trim  | Drop validator-coverage cases                                  |
| `test_network_parity.py`                 | Keep  | Critical: Python ↔ Rust feature parity                         |
| `test_nn_value_weight_promotion.py`      | DROP  | Promotion goes away                                             |
| `test_nn_value_weight_schedule.py`       | Trim  | Single-strategy now; cut `per_promotion` cases                 |
| `test_optimize_machine.py`               | Keep  |                                                                |
| `test_optimizer_state_steps.py`          | Keep  |                                                                |
| `test_penalty_schedule.py`               | Trim  | Drop `gated` cases                                              |
| `test_policy_mirroring.py`               | Trim  | Drop `legacy_*` paths                                          |
| `test_r2_sync.py`                        | Split | Mirror the split of `r2_sync.py`: one test file per module     |
| `test_render_seed_with_nn_overlay.py`    | DROP  | Inspection script removed                                      |
| `test_replay_buffer_semantics.py`        | Keep  | Critical: data integrity                                       |
| `test_replay_mirror_bootstrap.py`        | Keep  |                                                                |
| `test_resume_checkpoint_state.py`        | Trim  | Drop gate-state and per-model-stamp keys                        |
| `test_run_naming.py`                     | Trim  | Drop `is_friendly_run_id` tests (function deleted)              |
| `test_run_setup.py`                      | Keep  |                                                                |
| `test_runtime_overrides.py`              | Trim  | Drop `force_promote_next_candidate`; add buffer-size override   |
| `test_sequential_game_numbers.py`        | Keep  |                                                                |
| `test_trainer_candidate_gate_schedule.py`| DROP  | Gating removed                                                 |
| `test_trainer_direct_sync_gifs.py`       | Keep  | Direct-sync is the only path now                                |
| `test_trainer_shutdown.py`               | Keep  |                                                                |
| `test_wandb_resume.py`                   | Keep  |                                                                |
| `test_warm_start.py`                     | Trim  | Drop the offline-loop/early-stopping cases; keep one test for `--offline_only` mode |

### 12. Scripts dump: delete almost everything

`tetris_bot/scripts/ablations/` (16 files) → DELETE all. None of these are part of the supported workflow; they're frozen one-shot experiments.

`tetris_bot/scripts/inspection/` (22 files) → keep one:

- `optimize_machine.py` — wired into `tetris_mcts optimize`

Everything else (`mcts_visualizer.py`, `tetris_game.py`, `buffer_viewer.py`, `analyze_training_data.py`, `audit_mcts_tree.py`, `inspect_*.py`, `extract_*.py`, `render_*.py`, `policy_grid_visualizer.py`, `profile_games.py`, `download_wandb_training_data.py`, `value_predictor.py`, `viz_state_presets/`, `tree_playback_artifact.py`, `count_reachable_states.py`, `row_fill_zero_rates.py`, `sweep_num_workers.py`, `measure_bootstrap_tree_reuse.py`, `remap_checkpoint_aux_keys.py`) gets deleted. The Dash/plotly MCTS visualizer goes with this batch. Anything genuinely useful migrates into `tools/` as a single Typer subcommand.

`tetris_bot/scripts/utils/` (3 files) — `eval_utils`, `plot_utils`, `run_search_config` — most consumers are deleted ablations. Audit case-by-case; expect 0–1 files survive.

`warm_start.py` (1712 lines, top-level) → DELETE. The "offline training on a saved replay buffer" capability is folded into the trainer as a thin `--offline_only` flag (~30 lines): when set, the trainer skips `Generator.start()` and only samples from a preloaded `training_data.npz`. No early-stopping logic, no eval-after-each-chunk loop, no separate W&B project. If you want eval after offline training, run `tetris_mcts eval` against the resulting checkpoint. Weight-only transfer between architectures stays on `--init_checkpoint`.

`run_naming.py` (536 lines, mostly word lists) — keep but split: word lists into `data/run_words.json`, `generate_run_id` into a 20-line module. `is_friendly_run_id` is **deleted** — by construction the run id we generate is friendly; nothing in the runtime should be "checking" that.

### 13. Trainer rewrite — the centerpiece

Today: `Trainer` (3073 lines, 76 methods) owns 18 responsibilities. After:

```python
class Trainer:
    def __init__(self, config, device):
        self.model = TetrisNet(**config.network.model_dump()).to(device)
        self.optimizer = OptimizerBundle(self.model, config.optimizer)
        self.scheduler = build_scheduler(self.optimizer, config.optimizer)
        self.ema = ExponentialMovingAverage(self.model, config.optimizer.ema_decay)

        self.replay = ReplayPipeline(config.replay, device)
        self.checkpoint = CheckpointManager(config.run.checkpoint_dir)
        self.generator = Generator(config, replay=self.replay)
        self.completed_games = CompletedGameStream(self.generator)
        self.game_logger = CompletedGameLogger(self.completed_games)
        self.promotion = PromotionScheduler(
            generator=self.generator,
            export_dir=config.run.checkpoint_dir,
            interval_steps=config.run.model_sync_step_interval,
        )
        self.runtime_overrides = RuntimeOverrideWatcher(config)
        self.shutdown = GracefulShutdown()

    def train(self):
        self.generator.start()
        while not self.shutdown.requested:
            self.runtime_overrides.poll_and_apply(self)
            batch = self.replay.next_batch()
            metrics = train_step(self.model, batch, self.optimizer, self.scheduler, self.ema)
            self.game_logger.flush()
            if self.promotion.due(self.step):
                self.promotion.export_and_sync(self.model)
            if self.checkpoint.due(self.step):
                self.checkpoint.save(self.snapshot())
            self.step += 1
        self.shutdown.run_finalizers([self.generator.stop, self.checkpoint.flush, self.game_logger.flush_final])
```

Each component is its own ~150–300 line file. The trainer file collapses to ~250 lines.

### 14. Trainer-side cleanups (the smaller surgical ones)

- `apply_checkpoint_search_overrides`-style lift-and-shove: the giant `restore_trainer_from_checkpoint` collapses because (a) candidate gate state is gone, (b) per-model penalties are gone (computed from step), (c) runtime overrides are persisted as a single nested dict, not as 7 flat keys.
- `_build_direct_sync_recent_game_wandb_data` (44 lines) → `CompletedGameLogger.recent_game_gif()`, ~10 lines.
- `_create_wandb_gif_video`/`_cleanup_wandb_gif_files` → moved into `visualization/gif.py` with a `with TempGifFile() as path: ...` context.
- `_drain_remote_completed_games` and `push_remote_completed_games` → folded into `CompletedGameStream`.
- `_pin_batch_if_needed`, `_is_batch_on_training_device`, `_to_training_device`, `_training_batch_bytes`, `_tensor_field_pairs`, `_write_to_mirror`, `_sample_prefetched_batches`, `_load_replay_mirror`, `_refresh_replay_mirror`, `_sample_from_replay_mirror`, `_use_device_replay_mirror` → all collapse into `ReplayPipeline` (~3 methods: `bootstrap`, `next_batch`, `set_buffer_size`).

### 15. R2 sync: split out and stop being an afterthought

User: "the syncing of params to the generators via r2 should not be an afterthought, but done properly."

Today `r2_sync.py` is 1182 lines, mixes config / client / 5 background threads / pointer JSON / discovery / model bundle upload-download. The 4 uploader/downloader classes share ~80% of their threading + cursor + retry shape but each has its own copy.

Split (see target tree §1):

- `keys.py` (~50 lines): every R2 path string lives here. Stops `R2Settings` from owning all those `*_key` methods.
- `client.py` (~40 lines): just `make_s3_client(settings)`.
- A shared base `class PeriodicWorker` (in `runtime/`?) with `start()`, `stop()`, `_run()` skeleton; the 4 sync threads inherit and only define `_poll_once()`. ChunkUploader + ChunkDownloader + GameStatsUploader + GameStatsDownloader + ModelDownloader all become ~80–120 line classes instead of 200+ each.
- Self-play snapshot publishing: today the trainer publishes once on startup and re-publishes when `add_noise`/`visit_sampling_epsilon` change. After: the snapshot is published whenever **any** runtime override that affects MCTS changes, and it carries every field in `SELF_PLAY_SNAPSHOT_FIELDS`. The remote generator polls and applies the diff; no special-casing of two fields.
- Model publishing flow becomes "one archive + one yaml + one pointer" — see §15A for the full layout.

### 15A. Model bundle: one archive + sidecar yaml (no per-model state stamping)

Today every model export produces up to 7 files on disk and uploads up to 7 R2 objects per publish:

- `bundle.onnx` (full model)
- `bundle.conv.onnx` / `bundle.heads.onnx` / `bundle.fc.bin` (split for cached-board inference)
- `bundle.onnx.data` / `bundle.conv.onnx.data` / `bundle.heads.onnx.data` (optional ONNX external-weights sidecars)
- Plus `incumbent.json` at the run level — a "pointer" that smuggles per-model state (`nn_value_weight`, `death_penalty`, `overhang_penalty_weight`) into a JSON blob so workers know what search settings to use with this model.

That layout has three problems:

1. **Partial-bundle risk.** 7 separate uploads, ordered, with the pointer written last to limit the damage. A connection drop mid-bundle still leaves stale `.data` files lying around.
2. **The ONNX external-data rename hack.** When a bundle is copied/renamed locally (`candidate_step_X.onnx` → `incumbent.onnx`), the stale embedded filename inside the protobuf has to be rewritten in `_fix_onnx_external_data_references`. That hack only exists because the on-disk filenames change after export.
3. **Per-model state stamping.** §4 already kills the *runtime* dependency on these stamped values (everything is computed from the step), but the *shape* — "publish-time config encoded in the pointer JSON" — is wrong regardless. The user's ask: a sidecar yaml carrying the relevant information for that model, not stamping it on the model file or the pointer.

**After:** two files per publish, plus the pointer.

```
<prefix>/<run>/models/<step:020d>/bundle.tar.zst   # single archive of all member files
<prefix>/<run>/models/<step:020d>/bundle.yaml      # sidecar metadata
<prefix>/<run>/models/incumbent.yaml               # pointer: {step, archive_key, metadata_key}
```

`bundle.tar.zst` contents (zstd over tar — strong ratio at SSD-tier speeds; falls back to `.zip` if we don't want the libzstd dep):

```
bundle/
├── full.onnx
├── full.onnx.data        # only present when external-weights split exists
├── conv.onnx
├── conv.onnx.data
├── heads.onnx
├── heads.onnx.data
└── fc.bin
```

Member names inside the archive are fixed and never get renamed, so `_fix_onnx_external_data_references` is **deleted**.

`bundle.yaml` (the sidecar the user asked for) holds everything that today is either jammed into the pointer JSON or implicit in the surrounding run state:

```yaml
step: 12345
trained_at: 2026-05-04T08:14:23Z
git_sha: a1b2c3d
parameters: 1234567
network:                       # the architecture this checkpoint was trained with
  trunk_channels: 32
  num_conv_residual_blocks: 5
  ...
training:                      # informational
  total_steps_at_export: 12345
  optimizer_state: muon+adamw
config_digest: sha256:...      # hash of the TrainingConfig used; lets us detect drift
notes: ""
```

`incumbent.yaml` (pointer) is tiny:

```yaml
step: 12345
archive_key: tetris-mcts/<run>/models/00000000000000012345/bundle.tar.zst
metadata_key: tetris-mcts/<run>/models/00000000000000012345/bundle.yaml
```

This means **no MCTS schedule values live in the pointer or on the model.** Both `nn_value_weight` and the penalty values are derived from step + the trainer-published `SelfPlaySnapshot`. The pointer's job is strictly "which archive is current."

**Atomicity.** Publish order is `archive → bundle.yaml → incumbent.yaml`. The pointer write is the atomic swap; no partial-archive state is observable to readers.

**Local layout mirrors remote.**

```
training_runs/<run>/checkpoints/incumbent/
├── bundle.tar.zst
└── bundle.yaml
```

The Rust generator's `model_path` field becomes `archive_path`. On swap, it lazily extracts to `<run>/checkpoints/extracted/<step>/` and points inference at the extracted `full.onnx` / split files. We keep the last 1–2 extracted dirs and prune the rest. (Alternatively: extract straight to a tmpfs path; benchmark.)

**Format: `.tar.zst`.** The `zstandard` dep is already going in for the replay-chunk format (see §7), so this is free. ~30–40% smaller than zip-deflate on ONNX weights, 3–5× faster (de)compression.

**Code that gets deleted.**

- `tetris_bot/ml/artifacts.py:_fix_onnx_external_data_references` — gone (no rename inside archive).
- `tetris_bot/ml/artifacts.py:assert_rust_inference_artifacts`, `required_model_artifact_paths`, `optional_model_artifact_paths`, `copy_model_artifact_bundle` — collapse into a single `extract_bundle(archive: Path, dest: Path)` helper.
- `tetris_bot/ml/r2_sync.py:_bundle_member_paths` — gone.
- `tetris_bot/ml/r2_sync.py:upload_model_bundle`, `download_model_bundle` — each becomes ~15 lines (one upload, one download).
- `ModelPointer.{nn_value_weight, death_penalty, overhang_penalty_weight}` — fields removed; pointer carries only `step` + `archive_key` + `metadata_key`.

**Trainer-side change.** `Trainer._persist_incumbent_model_artifacts`, `_export_rust_inference_artifacts`, the candidate-export scratch directory dance, and the rename-and-fix-external-refs flow all collapse into a single `ModelBundlePublisher.publish(model, step, metadata)` call inside the new `PromotionScheduler` (see §13).

This subsection ties together three of the user's bullets: "syncing of params to the generators via r2 should not be an afterthought," "Cleaner model syncs with like a .yaml," and "currently, it generates a ton of different files, wondering if we could zip them."

### 15B. MCTS tree visualizer rewrite (fast, incremental, thousands of nodes)

The current `mcts_visualizer.py` (Dash + Plotly `graph_objects`) is on the delete list — but the *capability* needs a proper replacement, not a removal. The user's requirements:

- **Real-time / incremental.** Adding nodes should append to the existing render, not rebuild the whole figure. Today every change triggers a full Plotly re-emit over the WebSocket.
- **Step through fast.** Keyboard-driven scrubbing through simulations / moves / chance outcomes; no laggy click-and-wait.
- **Thousands of nodes at once.** Plotly chokes past ~1k. Need 10k+ rendered, with aggregation when zoomed out.

The current architecture can't get there incrementally; it needs to be replaced.

**Architecture: native Rust GUI binary using `egui` (with `eframe` for the windowing layer).**

Why egui:
- Immediate-mode GUI — frame-rate decoupled from MCTS expansion rate. The visualizer renders at 60 FPS regardless of how fast new nodes arrive.
- Native canvas (wgpu under the hood) handles 10k+ shapes without breaking a sweat.
- Keyboard handling is direct, no browser indirection — "step through fast" works the way keyboard shortcuts work in a real editor.
- Crate-local: lives next to `tetris_core` so it can use the same Rust types (`MCTSAgent`, `MCTSTreeExport`, `GameTreePlayback`, `SavedGameTreePlayback`) without round-tripping through Python.
- Single binary the CLI shells out to. No browser, no server, no port juggling.

**Layout in the workspace:**

```text
tetris_core/                     # Rust crate — engine, search, runtime, replay, inference
tetris_viz/                      # NEW Rust crate — tree visualizer + game state explorer
├── Cargo.toml
├── src/
│   ├── main.rs                  # eframe::run_native(...)
│   ├── app.rs                   # top-level App struct: tree state + UI state + input handling
│   ├── tree_state.rs            # incremental TreeState: NodeArena, parent pointers, stable IDs
│   ├── layout.rs                # incremental layout: Reingold-Tilford with frozen positions
│   ├── render/
│   │   ├── nodes.rs             # node glyph rendering, LOD aggregation, edge drawing
│   │   └── overlays.rs          # selected-node panel, value heatmap, policy bars
│   ├── controls.rs              # keyboard map, step/playback controls
│   └── source/
│       ├── live.rs              # connect to a running `MCTSAgent`, stream tree deltas
│       └── replay.rs            # load `SavedGameTreePlayback` JSON, scrub timeline
tetris_mcts/
└── cli.py                       # `tetris_mcts viz [...]` shells out to the tetris_viz binary
```

**Incremental tree model.** The visualizer keeps `TreeState` as a stable arena: nodes are appended; existing node IDs and `(x, y)` layout positions never move. New children of node N go into a slot reserved next to N's existing children — a Reingold-Tilford "freeze parents, place children" variant. Cost of adding K new nodes: O(K), not O(total tree).

**Tree delta stream.** Rust side exposes a small new API on `MCTSAgent`:

```rust
impl MCTSAgent {
    pub fn drain_tree_deltas(&self) -> Vec<NodeDelta>;
}

pub struct NodeDelta {
    pub node_id: u64,
    pub parent_id: Option<u64>,
    pub action: Option<u32>,            // None for chance edges
    pub visits: u32,
    pub value_sum: f32,
    pub prior: f32,
    pub kind: NodeKind,                 // Decision / Chance
}
```

Workers tagged for visualization push deltas into a bounded queue; the visualizer drains. No more "serialize the whole tree to JSON every time you want to inspect it" (which is what `MCTSTreeExport` does today).

**Level-of-detail.** When zoomed out:
- Subtrees with > N total nodes collapse to a single glyph showing `(node_count, max_visits, avg_value)`.
- Edges within the collapsed subtree are not drawn.
- Zoom-in re-expands smoothly (animation, not a hard pop) by walking the arena from the focused node outward until the visible-node budget is hit.

This is what unlocks "thousands of nodes": the renderer never tries to draw all 10k at once — it draws the ~500 that fit in the current viewport at the current zoom, plus aggregate glyphs for the rest.

**Step controls (keyboard map).**

| Key | Action |
| --- | --- |
| Space | play/pause MCTS expansion (live mode) |
| `→` / `←` | step one simulation forward/back |
| `Shift+→` / `Shift+←` | step one move forward/back |
| `Ctrl+→` / `Ctrl+←` | step one chance outcome |
| `f` | focus on best child |
| `r` | reset zoom to fit tree |
| `g` | go to root |
| `[` / `]` | shrink / grow visible-node budget |

**Two modes.**

1. **Live mode** (`tetris_mcts viz live --run_dir <run>`): connects to a fresh `MCTSAgent` loaded with the run's incumbent ONNX, runs simulations, streams deltas. The user can step through an in-progress search.
2. **Replay mode** (`tetris_mcts viz replay --playback <json>`): loads a `SavedGameTreePlayback` and lets the user scrub through the recorded simulation timeline. This replaces the `mcts_visualizer.py --saved_playback` path.

**Game state explorer.** A second pane shows the Tetris board for whichever node is selected — current piece, hold, queue, board state. This already exists in `mcts_visualizer.py`'s side panel and is uncontroversial; it just gets reimplemented in egui (cheaper).

**What gets deleted (final version).**

- `tetris_bot/scripts/inspection/mcts_visualizer.py` (Dash + Plotly rebuild-everything model)
- `tetris_bot/scripts/inspection/policy_grid_visualizer.py` — capability folded in as a viz panel
- `tetris_bot/scripts/inspection/extract_viz_state_preset.py`, `extract_saved_playback_step.py`, `viz_state_presets/` — replay mode reads `SavedGameTreePlayback` directly, no preset extraction needed
- The "save full-game playback" rendering hook in `runtime/game_generator/runtime.rs` stays (it's the source for replay mode), but `persist_worst_candidate_eval_tree` (already gone with §3) doesn't come back.

**What gets added.**

- New `tetris_viz/` Rust crate (~1500–2000 lines once feature-complete)
- New `MCTSAgent::drain_tree_deltas` + `NodeDelta` type in `tetris_core`
- One `tetris_mcts viz` Typer subcommand that shells out to the binary

**Build.** `tetris_mcts install` runs `cargo build --release -p tetris_viz` alongside the existing `maturin build`. Binary lands at `tetris_core/target/release/tetris_viz` (or a workspace target dir if we restructure to a Cargo workspace, which is the right move regardless).

**Implementation order.** This is its own substantial project — slot it as **step 13** in the migration order (§19) so it lands after the Python-side trainer/sync split is stable. Replay mode first (no live MCTS coupling), then live mode.

### 16. Rust runtime cleanups

- `LastGameInfo` (in `shared.rs`) carries 28 fields; many feed `to_dict()` for W&B. Per the user, "list out all of the different metrics, and then together decide which are actually useful."

  **Metrics inventory** (drop list to discuss):
  - `singles, doubles, triples, tetrises, tspin_*, perfect_clears, back_to_backs, max_combo, total_lines, holds, total_attack` — keep
  - `avg_overhang, episode_length, avg_valid_actions, max_valid_actions` — keep `episode_length` and `total_attack`; drop `avg_valid_actions`/`max_valid_actions` (debug-only)
  - `tree_avg_branching_factor, tree_avg_leaves, tree_avg_total_nodes, tree_avg_max_depth, tree_max_attack` — drop all 5; recompute on demand from saved playbacks if needed
  - `cache_hit_rate` — keep (cheap signal of placement-cache health, useful at a glance)
  - `cache_hits, cache_misses, cache_size` — drop (raw counts, not training health; recompute from profiler when needed)
  - `tree_reuse_hits, tree_reuse_misses, tree_reuse_rate, tree_reuse_carry_fraction` — drop
  - `traversal_total, traversal_expansions, traversal_terminal_ends, traversal_horizon_ends, traversal_*_fraction` — drop
  - `trajectory_predicted_total_attack_count/variance/std/rmse` — drop (this is the four-field block the user called out by name)

  After: `GameStats` carries clear types (singles, doubles, …, total_attack, total_lines, episode_length, avg_overhang). Any deeper signal lives in the saved playback, not in the per-game W&B blob.

- `runtime/game_generator/runtime.rs` shrinks roughly by half once the eval-worker arm is gone.
- `runtime/evaluation.rs` is **deleted entirely** (~370 lines): `evaluate_model`, `evaluate_model_without_nn`, `evaluate_seed`, `evaluate_parallel`, `GameEval`, `EvalResult`, `aggregate_game_evals`. Eval becomes "spin up `Generator` with deterministic config (`add_noise=false`, `visit_sampling_epsilon=0`, fixed `mcts_seed`, fixed seed list), drain N completed games via `CompletedGameStream`, aggregate Python-side." Same code path as self-play, no parallel implementation to maintain.
- `debug_predict_masked_from_tensors`, `debug_encode_state`, `debug_get_action_mask`, `debug_masked_softmax` (`lib.rs` lines 40–112) exist only for inspection scripts. With those gone, delete the four debug functions and shave 70+ lines.
- The `update_search_overrides(add_noise, visit_sampling_epsilon)` API on `GameGenerator` becomes `update_search_settings(LiveSearchSettings)` — one struct with all live-tunable fields, not a moving target of method args.
- Per the user, "checks and asserts on inputs": `validate_new_args` in `py_api.rs` (the constructor validator) — drop most of it. We construct `GameGenerator` from a typed Python config; if `num_workers <= 0` something is already very wrong. Keep checks that defend against *external* file-system state (e.g., training_data path is writable).

### 17. Typed-config and validator cleanup (bot-wide)

Per the user: "you really really dont need all that input validation like in NNValueWeightScheduleConfig."

- Delete every `model_validator(mode="after")` in `config.py`. We're not parsing user-typed YAML in production hot paths; the YAML is committed in the repo. Pydantic's type-coercion gives us 95% of the safety; the rest can crash on use.
- Delete `_validate_runtime_override_value` (44 lines in trainer).
- Delete the `if not 0.0 <= ema_decay < 1.0` and similar bounds checks in `Trainer.__init__`.
- Delete `loss.py:validate_action_masks_have_valid_rows` — Rust guarantees this; if the assertion fails we want a ValueError from CrossEntropyLoss, not a wrapper.

### 18. Files to delete outright (master list)

```text
# Python — top-level scripts
tetris_bot/scripts/warm_start.py                  (capability folded into `--offline_only` trainer flag)
tetris_bot/scripts/ablations/                     (entire dir)
tetris_bot/scripts/inspection/                    (all but optimize_machine.py + mcts_visualizer.py)
tetris_bot/scripts/utils/                         (likely all)

# Python — modules to fold/rename (delete originals)
tetris_bot/ml/r2_sync.py                          → split into tetris_mcts/sync/*.py
tetris_bot/ml/trainer.py                          → split into tetris_mcts/train/*.py
tetris_bot/ml/weights.py                          → split into tetris_mcts/train/{checkpoint,onnx_export}.py
tetris_bot/ml/replay_buffer.py                    → renamed to tetris_mcts/ml/replay_mirror.py
tetris_bot/ml/penalty_schedule.py                 → folded into config (single-strategy)
tetris_bot/ml/nn_value_weight_schedule.py         → folded into config (single-strategy)
tetris_bot/ml/wandb_resume.py                     → keep, move to tetris_mcts/train/

# Python — symbols to delete
tetris_bot/run_naming.py:is_friendly_run_id
tetris_bot/action_space.py: legacy adapters and tests
tetris_bot/ml/loss.py:validate_action_masks_have_valid_rows
tetris_bot/ml/network.py:build_aux_features        (if value_predictor.py is dropped)
tetris_bot/ml/policy_mirroring.py:legacy_*         (5 helpers)

# Rust
tetris_core/src/runtime/evaluation.rs              (entire file: evaluate_model, evaluate_model_without_nn, evaluate_seed, evaluate_parallel, GameEval, EvalResult, aggregate_game_evals)
tetris_core/src/lib.rs:debug_*                     (4 functions; eval pyfunctions also unregistered)
tetris_core/src/replay/npz.rs:adapt_legacy_policy_and_mask  + tests
tetris_core/src/runtime/game_generator/runtime.rs:persist_worst_candidate_eval_tree (and surrounding eval-worker arm)
tetris_core/src/runtime/game_generator/shared.rs:CandidateModelRequest, ModelEvalEvent, IncumbentState fields tied to gating

# Build / workflow
Makefile                                          → replaced by `tetris_mcts` Typer CLI

# Tests — see §10 table
```

### 19. Migration / order of operations

Suggested implementation order (each step ends in green tests + a working `make train`):

1. **Delete** ablation scripts, inspection scripts, `warm_start.py`, the `Makefile` ablation targets, and matching tests. (No code-path changes — just dead-code removal.)
2. **Delete** legacy adapters in Python and Rust + matching tests.
3. **Drop validators** from config classes; rip out `_validate_runtime_override_value` and similar.
4. **Collapse schedules** to single-strategy each; drop the `strategy` field.
5. **Delete candidate gating** from Rust and Python in one large commit. This is the biggest single change. Self-play continues, but the trainer pushes models on a step interval to a single generator (local). Tests for gating get deleted.
6. **Move per-model state to `LiveSearchSettings`** in Rust; collapse `restore_trainer_from_checkpoint`.
7. **Split `r2_sync.py`** into `tetris_mcts/sync/*`. R2 keys go through `sync/keys.py`.
8. **Split `trainer.py`** into `tetris_mcts/train/*`. The order inside this step: extract `ReplayPipeline` first (it's the most self-contained), then `CheckpointManager`, then `CompletedGameLogger`/`CompletedGameStream`, then `PromotionScheduler`, then `RuntimeOverrideWatcher`. Trainer file shrinks each round.
9. **Rename** `tetris_bot` → `tetris_mcts` and reorganize subfolders.
10. **Replace Makefile** with the `tetris_mcts` Typer CLI. Drop the Makefile.
11. **Add runtime override for `replay.buffer_size`** with shrink/grow semantics.
12. **Re-evaluate metrics list** with the user; delete the dropped Rust `LastGameInfo` fields.

After step 5, the working tree is already dramatically simpler. Steps 7–9 are mostly mechanical motion.

### 20. Resolved decisions

All decisions locked in:

1. **Package name: `tetris_mcts`.**
2. **Metric inventory:** keep the 16 game-outcome metrics + `cache_hit_rate`. Drop all tree/traversal/trajectory/cache-count fields. See §16 for the full list.
3. **`evaluate_model` deleted.** Eval is `tetris_mcts eval`: deterministic Generator config, drain N games, aggregate. All ~370 lines of `runtime/evaluation.rs` removed.
4. **Replay format: Zstd-NPZ.** `zstandard` dep used by both replay chunks and model archives.
5. **`model_sync_step_interval` default: 1000 steps.**
6. **`replay.mirror_replay_on_accelerator` default: `false`.** `replay.buffer_size` default: **10_000_000.** Override exposed for VRAM-flush machines.
7. **`play.py` and the Dash MCTS visualizer deleted.** Trainer's PIL gif renderer survives for W&B.
8. **Model archive format: `.tar.zst`.** Zstd is already in for replay; one dep, two payoffs.
9. **Warm-start replaced by `--offline_only` trainer flag.** ~30 lines, no early-stopping ceremony, no separate W&B project. Eval after offline training is `tetris_mcts eval`.
