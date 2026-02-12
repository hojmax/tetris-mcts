# Next Steps

- [ ] Try to handcraft great Tetris bot heuristics.
- [ ] Try running this with just the policy head, no value head. Must be better?
    - My leading hypothesis is that the noise in the value estimates are ruining the model's game playing ability. But the policy loss is actually going pretty good. So should be quite useful?
- [ ] What AWS instance would be well suited for this workload?
- [ ] Take an interestnig state like tetris_mcts/scripts/outputs/game_3194.gif at step 14 (oppurtinity for t spin), and see how the model evaluates that state? Is it not reached or something?
- [ ] Learning rate of around 0.0005. 0.005 is too high. Cycling too much on the learning rate.
- [ ] All the steps setttings depend on batch size which is kind of annoying.
- [ ] Maybe this is just a hella slow learning algorithm, and we need to scale up compute.

# Confusion

- Game length steadily increasing whilst attack is decreasing.
- Valid actions steadily increasing whilst attack is decreasing.
- Why did we see a big drop in attack mid training?
- I think the learning rate is too high, but the model was actually at its peak right at the highest LR.
- Hold rate is steadily increasing. Why is it so good to hold that much? I hold alot, but not every 0.375 moves. Is that not almost all the time, when you can only hold 50% of the time?

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

- [ ] We might want to scale cpuct to average value head precition magnitude?
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
