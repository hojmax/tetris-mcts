# Next Steps

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
