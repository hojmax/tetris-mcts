# Next Steps

- [ ] Improve policy layout. (not 735).
- [ ] Figure out data augmentation.
- [ ] How large can I make replay buffer?
- [ ] Adding in coords into input layer
- [ ] Larger trunk impact in warm start?

- [ ] Data augmentation. horisontal flip.
- [ ] Optimized EMA. 
- [ ] Adding in holes map into input layer?
- [ ] Adding in overhang map into input layer?
- [ ] sometimes evaluating candidates takes way longer than other times?
- [ ] the value / policy loss tradeoff experiment

- Not quite sure how important value vs. policy loss is? How would I test this?
  - One thing would be take a well trained network, and then try adding noise to the respective policy and value, and with the noisy estimates calculate the held-out loss. Then you can make two graphs, one is held-out policy loss on x-axis, and then avg. attack on y-axis. The other plot is held-out value loss on x-axis vs. avg. attack on y-axis.
  - What would we conclude from this? I guess we would want some notion of marginal utility of change in loss in the two? I guess we will also have to extrapolate to some degree probably, since the model will probably be improving outside the range of what I trained the marginal utility curves on. So say we had some extrapolated curve, and value loss said for 0.1 change in loss, I get +0.05 in avg. attak, and policy was for 0.1 change in loss I get +0.01 in attack, Then I guess you would scale the loss like: policy_loss + normalized_value_loss \* 5, i.e. the value loss gets to dominate 5 times as much in the loss.
  - I would imagine we would fit a sigmoid here on both, as they would start of with noise not being useful, then sharp increase, then plateu at maximal performance. Then for the marginal improvement we can simply use the derivative of the sigmoid.
  - We want to baseline this training method against just training 1:1 like we do right now.
  - You could even do this during training, like run 20 games with a tiny bit of noise, get loss, and then apprixomate gradient from these 2 data points.
  - Equation:
    - pl x (d pl / d vl) + vl x (~pl / ~vl)
  - The noise-to-loss approximation trick assumes that value estimate is unbiased, which we know is true and can verify. I am not sure if the same is true for the probability distribution? I guess cross entropy is a proper scoring rule, such that I would expect probabilities to be unbiased (actually we could also check this, compare estimate to the train buffer).

- [ ] Maybe just ask an agent team for like a full rewrite from scratch to make everything clean? altough all the reading I have been doing may have been for naught then.

- [ ] Implment new architecture, and then run warm start
- [ ] Allow an agent to iterate on the architecture by itself, so just give it a replay buffer and a fresh repo, and the idea is for it to get as low of a loss as possible, whilst being amenable to caching.
- [ ] Produce pareto frontier plot.
  - [ ] Start by generating for just single model. Like make script for getting the data.
- [ ] Try out spatial policy.

- [ ] We let the current run keep going until plateues, then we try to train some larger trunk models and see what the speed difference is.
- [ ] I need to somehow decide the optimal trunk size. Like I could try a smaller and a larger one, trained locally on the replay buffer, and then run both and see what the concrete speed difference is.
- [ ] Maybe I can simplify code now and drop all the bootstrappinitlogic, and just pretrain the network on the replay buffer from earlier experiment as warm start. But kind of wack to have unreproducable training run. Like does the method actually work without bootstrapping now? Could we get rid of the penalties and stuff? Once everything is fixed we should try, is kind of ugly. Why would this be necessary. Could of course just be compute multiplier.
- [ ] Why is the GPU going cold repeatedly like 50% of the time?
- [ ] Why don't the final moves look full optimal? Like just clean out the board in the last 5 moves? Surely that is highest attack possible? Instead it does random nonsense.
- [ ] Why don't we always have 50 piece episodes??????
- [ ] One thing we could do is to make the amount of resources allocated to evaluating candidates configurable, and then spend alot on it every time we have upgrade, and then less and less each time they fail to upgrade. So we evaluate frequently when we are seeing big changes, and rarely when we are seeing small changes.

- [ ] Are we actually using the whole action space? I guess we should log a bit about whether some actions are always masked out.
  - [ ] Actually would be great to see for all 7 pieces, how often each type of action is chosen. so would be like a vector of 7 x 732 + one number for holds. Maybe actually 7 x 323 where the last entry is for holds. Also will be less than 732 for all of them, more like 6XX. Could be cool to visualize the heatmap here.
  - [ ] Maybe higher exploration factor? Yeah especially since sampling from visit frequencies anyways, so rare moves are still really really rare.

- Look up the gradient norm vs. learning rate schedueling advice again.

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

- [ ] Why is episode length not 50/50? I guess random piece placements?? Still confused, mysterious answer.
- [ ] Why was the lower loss warm started network not better than it was?
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
