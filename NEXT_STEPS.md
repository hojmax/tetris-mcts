# Next Steps

- [ ] Why are the replays always worse than the average attack would suggest?
- [ ] Are we actually using the whole action space? I guess we should log a bit about whether some actiosn are always masked out.
- [ ] The tetris tspin stuff is still weird.
  - Investigate with make viz
- [ ] Batch chance nodes
- [ ] Too low learning rate?
- [ ] Slow batches still?
- [ ] Faster training loop
- [ ] Speed up data generation
- [ ] Multi machine RL environment generation?
- [ ] try with and without the penalties on offline seeds.
- [ ] Try training model offline and see if we hit the same ceiling that the model had during training. I.e. is it fully fit, and we are simply shifting the distribution and that is why value loss increases, or is there something suboptimal in the training during game generation
- [ ] Replay buffer should grow gradually
- [ ] Think about how much of the aux computation can be cached
- [x] Check out new gif rendering
- [x] Compare larger trunk effect on computation speed. Like we have a lot of caching,
- [x] Speed of game generation based on model sizes.
- [x] Double the board trunk since it is cached. (seems to be 2% slower for a 1% val loss improvement)
- [x] Batch the chance nodes?
- [x] Min / Max Normalization vs. tanh
- [x] Should we divide by the mean value in the tanh normalization?
- [x] Run min max vs. tanh normalization constants.
- [x] Right optimizer?
- [x] Bigger replay buffer
- [x] Divisor in tanh sweep
- [x] Instead of seperate evaluation (complex) just use the candidate runs? They will be on fixed seeds anyways. So we can take one of those and save the gif for.
- [x] Not all plots are x-axis synced like top1_accuracy
- [x] Fixed candidate seeds, no need for 50 random
- [x] Scale down death and pverhang penalty

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
