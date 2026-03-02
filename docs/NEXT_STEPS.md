# Next Steps

- [ ] Higher exploration
      Command:

```
env -u CONDA_PREFIX -u VIRTUAL_ENV -u PYTHONPATH PATH="$HOME/.cargo/bin:$PATH" \
    .venv/bin/python tetris_bot/scripts/inspection/optimize_machine.py \
    --num_games 600 --simulations 300 --num_repeats 1 \
    --worker_search adaptive --max_worker_evals_per_combo 6 \
    --backend_strategy staged --primary_backend tract --backends tract \
    --worker_candidates 64 128 256 512 --skip_build true
```

```
eyboardInterrupt
(main) root@C.32137294:/workspace/tetris-mcts$ uv run python tetris_bot/scripts/inspection/sweep_num_workers.py
2026-02-27 22:14:09 [info     ] Starting worker sweep          model_path=None num_games=40 num_repeats=1 simulations=2000 use_dummy_network=True worker_candidates=[24, 40, 48, 56, 64, 160]
2026-02-27 22:14:09 [info     ] Evaluating worker setting      num_workers=24 repeats=1
2026-02-27 22:14:46 [info     ] Completed benchmark run        elapsed_sec=36.896 games_per_sec=1.084 num_workers=24 repeat_idx=0
2026-02-27 22:14:46 [info     ] Evaluating worker setting      num_workers=40 repeats=1
2026-02-27 22:15:12 [info     ] Completed benchmark run        elapsed_sec=25.914 games_per_sec=1.544 num_workers=40 repeat_idx=0
2026-02-27 22:15:12 [info     ] Evaluating worker setting      num_workers=48 repeats=1
2026-02-27 22:15:37 [info     ] Completed benchmark run        elapsed_sec=24.286 games_per_sec=1.647 num_workers=48 repeat_idx=0
2026-02-27 22:15:37 [info     ] Evaluating worker setting      num_workers=56 repeats=1
2026-02-27 22:16:02 [info     ] Completed benchmark run        elapsed_sec=25.047 games_per_sec=1.597 num_workers=56 repeat_idx=0
2026-02-27 22:16:02 [info     ] Evaluating worker setting      num_workers=64 repeats=1
2026-02-27 22:16:27 [info     ] Completed benchmark run        elapsed_sec=24.988 games_per_sec=1.601 num_workers=64 repeat_idx=0
2026-02-27 22:16:27 [info     ] Evaluating worker setting      num_workers=160 repeats=1
2026-02-27 22:16:52 [info     ] Completed benchmark run        elapsed_sec=25.201 games_per_sec=1.587 num_workers=160 repeat_idx=0

=== Worker Sweep Summary ===
 Workers   Median G/s   Mean G/s    Min G/s    Max G/s   Median M/s   Median sec
--------------------------------------------------------------------------------------
      24        1.084      1.084      1.084      1.084         50.6       36.896
      40        1.544      1.544      1.544      1.544         72.1       25.914
      48        1.647      1.647      1.647      1.647         76.9       24.286
      56        1.597      1.597      1.597      1.597         74.6       25.047
      64        1.601      1.601      1.601      1.601         74.8       24.988
     160        1.587      1.587      1.587      1.587         74.1       25.201

Best setting: workers=48 median_games_per_sec=1.647 mean_games_per_sec=1.647
2026-02-27 22:16:52 [info     ] Saved worker sweep results     output=/workspace/tetris-mcts/benchmarks/worker_sweep.json
(main) root@C.32137294:/workspace/tetris-mcts$ uv run python tetris_bot/scripts/inspection/sweep_num_workers.py
2026-02-27 22:19:37 [info     ] Starting worker sweep          model_path=None num_games=64 num_repeats=1 simulations=2000 use_dummy_network=True worker_candidates=[64, 160]
2026-02-27 22:19:37 [info     ] Evaluating worker setting      num_workers=64 repeats=1
2026-02-27 22:20:19 [info     ] Completed benchmark run        elapsed_sec=41.713 games_per_sec=1.534 num_workers=64 repeat_idx=0
2026-02-27 22:20:19 [info     ] Evaluating worker setting      num_workers=160 repeats=1
2026-02-27 22:20:57 [info     ] Completed benchmark run        elapsed_sec=37.714 games_per_sec=1.697 num_workers=160 repeat_idx=0

=== Worker Sweep Summary ===
 Workers   Median G/s   Mean G/s    Min G/s    Max G/s   Median M/s   Median sec
--------------------------------------------------------------------------------------
      64        1.534      1.534      1.534      1.534         72.2       41.713
     160        1.697      1.697      1.697      1.697         79.8       37.714

Best setting: workers=160 median_games_per_sec=1.697 mean_games_per_sec=1.697
2026-02-27 22:20:57 [info     ] Saved worker sweep results     output=/workspace/tetris-mcts/benchmarks/worker_sweep.json
(main) root@C.32137294:/workspace/tetris-mcts$ uv run python tetris_bot/scripts/inspection/sweep_num_workers.py
2026-02-27 22:24:17 [info     ] Starting worker sweep          model_path=None num_games=128 num_repeats=1 simulations=2000 use_dummy_network=True worker_candidates=[128, 160]
2026-02-27 22:24:17 [info     ] Evaluating worker setting      num_workers=128 repeats=1
2026-02-27 22:25:35 [info     ] Completed benchmark run        elapsed_sec=77.875 games_per_sec=1.644 num_workers=128 repeat_idx=0
2026-02-27 22:25:35 [info     ] Evaluating worker setting      num_workers=160 repeats=1
```

- [ ] We are overfitting on the test evals.
- [ ] Go back to hyper params like v37 and then scale up and run on vast.ai
- [ ] What if you subtracted the root node value from all the child values before tanh? Then you immediately see if it is worse or better than expected.
- [ ] look at mcts tree across multiple moves, not just root

- [ ] I screwed up the loose penalty I think
- [ ] The bootstrapping is way worse now...
- [ ] Make network smaller and faster
- [ ] The tetris tspin stuff is still weird.
  - Investigate with make viz
- [ ] Fix the t-spin logic
- [ ] Why are the replays always worse than the average attack would suggest?
- [ ] Are we actually using the whole action space? I guess we should log a bit about whether some actiosn are always masked out.
- [ ] Speed up data generation
- [ ] Multi machine RL environment generation?
- [ ] Replay buffer should grow gradually

- [ ] What is so inherently noisy about the values? Why does it have such a high variance?
- [ ] What AWS instance would be well suited for this workload?
- [ ] Adding in alpha downweighting of value loss?

- [ ] Do offline learning experiments. Things to validate:
  - Learning rate and scheduler
  - Model size
  - Batch size
  - Policy / Value loss weighting
  - Weight decay
  - Optimizer
  - Architecture choices
  - Huber loss vs. MSE loss
- [ ] predict "n-step bootstrapped return" instead of "cummulative reward"

# Confusions

- [ ] Also twice the number of simulations???
- [ ] How come the model is not even better even when doing tree reuse
- [ ] Why does MCTS sometimes have games with like 1 in attack? And sometimes 37? What is the difference between these games? Look at replay buffer and inspect.

# Deep Review

Done: ✅
In progress: 🟨

## Rust Code

### tetris_core/src/scoring.rs ✅

### tetris_core/src/lib.rs ✅

### tetris_core/src/piece.rs ✅

### tetris_core/src/nn.rs ✅

### tetris_core/src/moves.rs ✅

### tetris_core/src/kicks.rs ✅

### tetris_core/src/constants.rs ✅

### tetris_core/src/env/board.rs ✅

### tetris_core/src/env/clearing.rs ✅

### tetris_core/src/mcts/utils.rs ✅

### tetris_core/src/mcts/search.rs ✅

### tetris_core/src/mcts/results.rs ✅

## Python Code

### tetris_bot/config.py ✅

### tetris_bot/run_setup.py ✅

### scripts/train.py ✅

### tetris_bot/ml/loss.py ✅

# Backlog

- [ ] Is huber loss even better than MSE loss?
- [ ] Reading through and validating all code
- [ ] Do we need a larger network?
- [ ] What alternative hardware could we run on?
- [ ] Would an instance with a GPU be better? What is the major bottleneck, CPU or GPU? Like maybe AWS instance with a ton of CPUs for deep tree search.
- [ ] Looking at training data
- [ ] Optimize int4 and int8 everywhere.
- [ ] More rust profiling
- [ ] Better splitting up of rust between the environment and the MCTS. Two different packages / folders.
- [ ] Play the tetris game to ensure the environment is working correctly
- [ ] Save a full rollout tree from during training and inspect it with `make viz` tool
- [ ] Stress test next possible pieces with unit tests, like that it can do wild twists and stuff and that it correctly decides those possible locations
- [ ] Benchmarking and improving speed of MCTS search
- [ ] Testing speed on different hardware then the Macbook Air
- [ ] Sweep over best value head weighting for ~final networks.
- [ ] Maybe this is just a hella slow learning algorithm, and we need to scale up compute.
- [ ] Visualizing MCTS search and verifying correctness
- [ ] Adding hand crafted heuristics to offload work off the neural network
  - can t-spin filter?
- [ ] Try to handcraft great Tetris bot.
- [ ] Another round of benchmarking and optimizing?
- [ ] Make a readme

## Test Environment State

Current Piece:

```
T
```

Queue:

```
I,O,L,S,Z
```

Board:

```
..........
..........
..........
..........
..........
..........
..........
..........
..........
..........
..........
..........
..........
..........
..........
..........
..........
....LLL...
LL...LLLLL
LLL.LLLLLL
```
