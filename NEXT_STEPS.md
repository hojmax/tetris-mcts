# Next Steps

- [ ] What is so inherently noisy about the values? Why does it have such a high variance?
- [ ] Bumpiness is not scaled correctly.
- [ ] Turn off the nn value weight scheduele.
- [ ] Deeper search depth potentially.
- [ ] Make sure all the features are scaled correctly.
- [ ] What AWS instance would be well suited for this workload?
- [ ] Benchmark conv net depth impact on speed, since caching so high. 96% caching.
- [ ] Adding in alpha downweighting of value loss?
- [ ] What the hell do we do about the extreme about of noise in the value head? Also the policy is waay too myopic right now.
  - Maybe the training loop could in the background run tests on what value head weight to use, and so it could slowly rise over time as the network gets more calibrated?
  - Train with Huber loss
  - Train with "n-step bootstrapped return"

- Make nn_value_weight adaptive, not fixed: ramp it up only when value quality is decent (you already log train/value_explained_variance)
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
- [ ] Batched leaf inference.

# Confusions

- [ ] Why does MCTS sometimes have games with like 1 in attack? And sometimes 37? What is the difference between these games? Look at replay buffer and inspect.

# Deep Review

Done: ✅
In progress: 🟨

## tetris_core/src/scoring.rs ✅

## tetris_core/src/piece.rs ✅

## tetris_core/src/nn.rs ✅

## tetris_core/src/moves.rs ✅

## tetris_core/src/kicks.rs ✅

## tetris_core/src/constants.rs ✅

## tetris_core/src/env/board.rs ✅

## tetris_core/src/env/clearing.rs ✅

# Backlog

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
