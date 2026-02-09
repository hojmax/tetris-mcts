# Next Steps

- [ ] Take an interestnig state like tetris_mcts/scripts/outputs/game_3194.gif at step 14 (oppurtinity for t spin), and see how the model evaluates that state? Is it not reached or something?
- [ ] Learning rate of around 0.0005. 0.005 is too high. Cycling too much on the learning rate.
- [ ] Model sometimes gets worse at playing, I think the LR spikes causes this? We might want to gate swapping it out with the evaluation step? Yeah like playing 20 games, only swapping out if total attack is higher than the previous model. Hmm wait but that would mean huge intervals between swaps... I guess that is fine later on, but in the beginning we want to swap out more often. You could do something like this. You swap out one of the works with the new model, the other ones just run the best, and keep putting into the replay buffer. THen the one worker, plays 20 games, and then we compare total attack to previous models. If it is higher, then we first add in all the new games to the replay buffer from that one worker, and then swap out the model for all workers. If it is lower, then we just keep the old model, and throw out the newly generated games for that one worker.
- [ ] Maybe larger buffer to avoid catastrophic forgetting?
- [ ] Larger batch sizes? Seems like low GPU utilization?
- [ ] I need tree stats. Branchning factor (avg. children), number of leafs, total nodes, max depth, max attack seen in the tree (like a move causing attack, not the model prediction).
- [ ] All the steps setttings depend on batch size which is kind of annoying.
- Value loss weight of 70.0 is maybe a tad too high. Changing to 30.0.

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
