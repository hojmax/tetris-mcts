# Next Steps

- [ ] Final phase of training with low learning rate, no noise and no stochasticity.
- [ ]  Auxiliary loss
    - [ ]  Predict next move logits
    - [ ]  Predict value sum over next 1,2,3,4,5 moves
    - [ ]  See offline loss
- [ ]  Check loss on input augmentation
- [ ] Add final finetuning phase with no random piece placements and lower LR.

- [ ] Change network to do spatial policy. 4 x 20 x 10
- [ ] Are we actually using the whole action space? I guess we should log a bit about whether some actions are always masked out.
  - Actually would be great to see for all 7 pieces, how often each type of action is chosen. so would be like a vector of 7 x 732 + one number for holds. Maybe actually 7 x 323 where the last entry is for holds. Also will be less than 732 for all of them, more like 6XX. Could be cool to visualize the heatmap here.
- [ ] Simplify code and drop all the bootstrapping and weighting logic
- [ ] In the piece mask placemenet bfs, for O piece no need to consider rotation.
- [ ] Maybe higher exploration factor? Yeah especially since sampling from visit frequencies anyways, so rare moves are still really really rare.
- [ ] Remove penalties
- [ ] Are there any auxiliary losses we could apply?
  - Predict logits for n+1 move.
  - Predict cummulative attack for next 1,2,3,4,5 moves.

# Experiments

- [ ] Redo the value / policy loss tradeoff experiment on better network
  - The noise-to-loss approximation trick assumes that value estimate is unbiased, which we know is true and can verify. I am not sure if the same is true for the probability distribution? I guess cross entropy is a proper scoring rule, such that I would expect probabilities to be unbiased (actually we could also check this, compare estimate to the train buffer).

- [ ] Would be great to do the marginal KL divergence between a final distribution after 20K search, and then save the intermediate distributions. After every single simulation, save the distribution. End up with 20K ones. Do KL divergence for all of them relative to the final one, and plot this. Then also plot the emperical direvative of this.
  - Do this for many different like 10 different ones, and show all the different curves in one plot. Maybe this varies quite a bit.
  - Then we also show an average such curve.

- [ ] Take trained network, and then gridsearch c_puct and temperature
  - What else could we search?

- [ ] Produce pareto frontier plot.
  - Start by generating for just single model. Like make script for getting the data.
  - We let the current run keep going until plateues, then we try to train some larger trunk models and see what the speed difference is.

- [ ] Showing auto-research results and summarizing findings

- [ ] Ablating all parts of architecture and the custom board features

# Confusions

- [ ] Why is episode length not 50/50? I guess random piece placements?? Still confused, mysterious answer.
- [ ] Why are some of the games still ending early?
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

### tetris_core/src/game/action_space.rs ✅

- Is Y_MAX_EXCLUSIVE not just board height from others consts? Same with X_MAX_EXCLUSIVE being board width?

### tetris_core/src/game/env/global_cache.rs ✅

- Why not raise instead of returning None in build_board_key?
- Why is rotation part of key in build_placement_lookup_key?

### tetris_core/src/game/env/lock_delay.rs ✅

### tetris_core/src/game/env/movement.rs ✅

### tetris_core/src/game/env/piece_management.rs ✅

- Lets combine the shared logic for spawn_piece_internal and spawn_piece_from_type

### tetris_core/src/game/env/placement.rs ✅

### tetris_core/src/game/env/pymethods.rs 🟨

- Kind of weird that step just does nothing if the action is not 1 through 7? Maybe should raise?

## Python Code

### tetris_bot/config.py ✅

- Why do we have both NetworkConfig and ModelKwargs? Can't we just do only NetworkConfig?

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
- [ ] Look up the gradient norm vs. learning rate schedueling advice again.
- [ ] Speed up data generation
- [ ] Multi machine RL environment generation?
- [ ] Replay buffer should grow gradually?
- [ ] Maybe just ask an agent team for like a full rewrite from scratch to make everything clean? altough all the reading I have been doing may have been for naught then.
- [ ] Reintroduce the board stats in the fusion part. Ablate this.
- [ ] Optimized EMA.
- [ ] Adding in holes map into input layer?
- [ ] Adding in overhang map into input layer?
- [ ] sometimes evaluating candidates takes way longer than other times?

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

# Good Instances For Workload

1. Training Workload:

   ```
   1x RTX 5090
   Verified
   89.22.197.55

   1 mon 1d
   1 mon 26d
   $0.009/hr
   Instance ID: 32315723
   Host: 56505
   Machine ID: 37025
   Vol:
   No Volumes
   Max CUDA: 13.0
   108.1 TFLOPS
   VRAM 0.5/31.8 GB
   1452.9 GB/s
   DLPerf
   202.8 DLPerf
   374.0 DLP/$/hr
   Network
   125 ports
   364.2 Mbps
   845.6 Mbps
   CPU
   AMD EPYC 9754 128-Core Processor
   64.0/512 CPU
   4 / 129.0 GB
   Disk
   Predator SSD GM7 4TB
   14028.8 MB/s
   22.0 / 32.0 GB
   Motherboard
   GENOA2D24G-2L
   PCIE 5.0/16x
   54.2 GB/s
   ```

2. Experimentation Workload:

   ```
   4x RTX 3090
   Verified
   175.155.64.175

   2m 2s
   3 mon
   $0.548/hr
   Instance ID: 34144369
   Host: 124072
   Machine ID: 37162
   Vol:
   No Volumes
   Max CUDA: 12.8
   141.1 TFLOPS
   VRAM 1.8/96.0 GB
   825.1 GB/s
   DLPerf
   162.4 DLPerf
   296.7 DLP/$/hr
   Network
   500 ports
   70.1 Mbps
   264.0 Mbps
   CPU
   Xeon® Gold 6330
   56.0/112 CPU
   1 / 257.8 GB
   Disk
   SAMSUNG MZQLB3T8HALS-00007
   2408.8 MB/s
   0.0 / 16.0 GB
   Motherboard
   JIUTIAN
   PCIE 4.0/16x
   17.6 GB/s
   ```
