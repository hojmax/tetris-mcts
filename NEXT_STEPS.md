# Next Steps

- [ ] Deeper search depth potentially.
- [ ] What AWS instance would be well suited for this workload?
- [ ] Try running it on AWS instance and see generation speed compared to Mac.
- [ ] Do offline learning experiments. Things to validate:
  - Learning rate and scheduler
  - Model size
  - Batch size
  - Policy / Value loss weighting
  - Weight decay
  - Optimizer
  - Architecture choices
  - Huber loss vs. MSE loss

- [ ] Run with updated feature set, integrate all the new features.
- [ ] Another round of benchmarking and optimizing?
- [ ] Benchmark conv net depth impact on speed, since caching so high. 96% caching.
- [ ] Adding in alpha downweighting of value loss?
- [ ] predict "n-step bootstrapped return" instead of "cummulative reward"

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

- [ ] Visualizing MCTS search and verifying correctness
- [ ] Reading through and validating all code
- [ ] Proper network split caching. Caching board CNN head, and optimizing such that we only run last part of network.
- [ ] Do we need a larger network?
- [ ] What alternative hardware could we run on?
- [ ] Would an instance with a GPU be better? What is the major bottleneck, CPU or GPU? Like maybe AWS instance with a ton of CPUs for deep tree search.
- [ ] Looking at training data
- [ ] Caching board representation from network in inference.
- [ ] Optimize int4 and int8 everywhere.
- [ ] More rust profiling
- [ ] Better splitting up of rust between the environment and the MCTS. Two different packages / folders.
- [ ] Play the tetris game to ensure the environment is working correctly
- [ ] Save a full rollout tree from during training and inspect it with `make viz` tool
- [ ] Stress test next possible pieces with unit tests, like that it can do wild twists and stuff and that it correctly decides those possible locations
- [ ] Benchmarking and improving speed of MCTS search
- [ ] Testing speed on different hardware then the Macbook Air
- [ ] Sweep over best value head weighting.
- [ ] Maybe this is just a hella slow learning algorithm, and we need to scale up compute.
- [ ] Adding hand crafted heuristics to offload work off the neural network
  - can t-spin filter?
- [ ] Try to handcraft great Tetris bot.
- [ ] All the steps setttings depend on batch size which is kind of annoying. Maybe wall clock is nicer for all these settings?

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
