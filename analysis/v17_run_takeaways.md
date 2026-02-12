# v17 run takeaways (live snapshot)

Source data inspected:
- `wandb/run-20260212_205042-yhijdb8q/files/output.log`
- `training_runs/v17/config.json`
- `training_runs/v17/checkpoints/`
- `training_runs/v17/checkpoints/model_candidates/`

Snapshot time: **2026-02-12 22:55:11** (log-local)

## Current status
- Run is still active.
- Latest training log point: **step 138,500**, **games_generated 1,841**, **buffer_size 117,189**.
- Effective throughput over the run window (~2.07h): **18.61 steps/s**, **0.247 games/s**.
- `training_data.npz` has **not** been written yet (config uses `games_per_save=2000`; only 1,841 games so far).
- Latest checkpoint is still `checkpoint_80000.pt` (next checkpoint at step 160,000).

## Most important findings

1. **No model promotions at all (0/44 decisions).**
- Evaluator decisions observed: **44**.
- Promotions: **0**.
- Incumbent has stayed at **step 0** and `incumbent_uses_network=False` in every decision.
- Candidate attack is far below incumbent baseline:
  - Candidate avg attack: min **0.0**, median **0.133**, max **0.8**.
  - Incumbent avg attack: roughly **6.89 -> 7.53** over time.
- Candidate/incumbent attack ratio:
  - Mean ratio ~**2.2%**.
  - Best ratio ~**11.0%**.

2. **Promotion gate is extremely strict in this phase.**
- Gate logic in Rust (`tetris_core/src/generator/game_generator.rs`): promote only if
  `candidate_avg_attack > incumbent_avg_attack` (strict greater-than).
- Candidate is compared on **30 games** to incumbent **lifetime average** (currently ~1.8k games), making promotion from bootstrap very hard.

3. **Candidate exports are outpacing evaluator throughput, so many candidates are dropped.**
- Candidate exports queued: **69** (every 2k steps).
- Evaluated: **44**.
- Not evaluated: **25** candidate steps.
- Mean lag from candidate step to evaluation decision: ~**4,211 trainer steps** (~**3.76 minutes** at current speed).
- This matches generator behavior: pending candidate is replaced by newer one when a new candidate arrives.

4. **LR schedule has already turned upward after step 100k.**
- `lr_schedule=cosine`, `lr_decay_steps=100000`, `lr_min_factor=0.5`.
- Learning rate reached its minimum **0.00025** at **step 100,000** (`2026-02-12 22:18:48`), then started increasing again.
- At step 138,500, LR is **0.0003308**.
- If monotonic decay was intended, current scheduler behavior is mismatched.

5. **Periodic evals show very weak gameplay currently.**
- Eval events (steps 30k, 60k, 90k, 120k) each ran `num_games=1` (due `eval_seeds=[0]`).
- All had `avg_attack=0.0`; lines were [0, 5, 0, 1].
- This is noisy (single-seed eval) and does not show meaningful improvement yet.

6. **Artifact cleanup bug: stale `.data` files are accumulating in `model_candidates/`.**
- `model_candidates/` size currently ~**107.6 MB**.
- `.data` files account for ~**106.0 MB**.
- **201 orphan `.data` files** have no parent `.onnx`/split file.
- Rust cleanup removes `.onnx`, `.conv.onnx`, `.heads.onnx`, `.fc.bin` but not corresponding `.data` sidecars.

## Secondary observations
- Loss is noisy and mostly flat-to-slightly-worse so far:
  - Min **1.93** (early), max **12.38** (spike), mean ~**4.7**.
  - Recent values are typically **4.3–5.0**.
- Steps/game over the run is high (~**75.4**), reflecting trainer running much faster than new game generation.

## Net takeaway
The run is healthy operationally (no crashes), but **learning is not crossing the promotion gate**. The strongest blockers appear to be:
- bootstrap incumbent baseline being much stronger than candidate policy quality,
- strict promotion criterion against incumbent lifetime average,
- evaluator throughput that drops many intermediate candidates,
- and potentially unintended LR re-increase past 100k.
