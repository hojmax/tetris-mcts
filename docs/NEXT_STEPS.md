# Next Steps

- [ ] Maybe I can simplify code now and drop all the bootstrapping logic, and just pretrain the network on the replay buffer from earlier experiment as warm start. But kind of wack to have unreproducable training run. Like does the method actually work without bootstrapping now? Could we get rid of the penalties and stuff? Once everything is fixed we should try, is kind of ugly. Why would this be necessary. Could of course just be compute multiplier.
- [ ] Look at game that ends really early, and understand exactly why that happened. I would never expect it to look like higher reward to screw up the game board.
- [ ] New Metric: Is past attack + future attack roughly stable? Maybe we log per game variance in this estimate.
- [ ] Why is the GPU going cold repeatedly like 50% of the time?
- [ ] Why don't the final moves look full optimal? Like just clean out the board in the last 5 moves? Surely that is highest attack possible? Instead it does random nonsense.
- [ ] Why don't we always have 50 piece episodes??????
- [ ] One thing we could do is to make the amount of resources allocated to evaluating candidates configurable, and then spend alot on it every time we have upgrade, and then less and less each time they fail to upgrade. So we evaluate frequently when we are seeing big changes, and rarely when we are seeing small changes.

- [ ] Are we actually using the whole action space? I guess we should log a bit about whether some actiosn are always masked out.
- [ ] Speed up data generation
- [ ] Multi machine RL environment generation?
- [ ] Replay buffer should grow gradually?

- [ ] Adding in alpha downweighting of value loss?
- [ ] Do offline learning experiments. Things to validate:
  - Learning rate and scheduler
  - Model size
  - Batch size
  - Policy / Value loss weighting
  - Weight decay
  - Optimizer
  - Architecture choices
- [ ] predict "n-step bootstrapped return" instead of "cummulative reward"
- [ ] Improving model architecture

# Confusions

- [ ] How come the model is not even that much better even when doing tree reuse
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

### tetris_core/src/inference/mod.rs ✅

### tetris_core/src/replay/mod.rs ✅

### tetris_core/src/replay/types.rs ✅

### tetris_core/src/replay/npz.rs ✅

### tetris_core/src/runtime/evaluation.rs ✅

### tetris_core/src/runtime/mod.rs ✅

### tetris_core/src/runtime/game_generator/py_api.rs ✅

## Python Code

### tetris_bot/config.py ✅

### tetris_bot/run_setup.py ✅

### scripts/train.py ✅

### tetris_bot/ml/loss.py ✅

# Backlog

- [ ] Reading through and validating all code
- [ ] Do we need a larger network?
- [ ] Looking at training data
- [ ] More rust profiling
- [ ] Better splitting up of rust between the environment and the MCTS. Two different packages / folders.
- [ ] Play the tetris game to ensure the environment is working correctly
- [ ] Save a full rollout tree from during training and inspect it with `make viz` tool
- [ ] Stress test next possible pieces with unit tests, like that it can do wild twists and stuff and that it correctly decides those possible locations
- [ ] Benchmarking and improving speed of MCTS search
- [ ] Testing speed on different hardware then the Macbook Air
- [ ] Maybe this is just a hella slow learning algorithm, and we need to scale up compute.
- [ ] Visualizing MCTS search and verifying correctness
- [ ] Adding hand crafted heuristics to offload work off the neural network
  - can t-spin filter?
- [ ] Try to handcraft great Tetris bot.
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
