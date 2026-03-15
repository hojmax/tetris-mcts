# Next Steps

- [ ] Maybe I can simplify code now and drop all the bootstrapping logic, and just pretrain the network on the replay buffer from earlier experiment as warm start. But kind of wack to have unreproducable training run. Like does the method actually work without bootstrapping now? Could we get rid of the penalties and stuff? Once everything is fixed we should try, is kind of ugly. Why would this be necessary. Could of course just be compute multiplier.
- [ ] Look at game that ends really early, and understand exactly why that happened. I would never expect it to look like higher reward to screw up the game board.
- [ ] New Metric: Is past attack + future attack roughly stable? Maybe we log per game variance in this estimate.
- [ ] I don't want random move sampling in the the last ~10 moves.
- [ ] Why is the GPU going cold repeatedly like 50% of the time?
- [ ] Why don't the final moves look full optimal? Like just clean out the board in the last 5 moves? Surely that is highest attack possible? Instead it does random nonsense.
- [ ] Why don't we always have 50 piece episodes??????
- [ ] One thing we could do is to make the amount of resources allocated to evaluating candidates configurable, and then spend alot on it every time we have upgrade, and then less and less each time they fail to upgrade. So we evaluate frequently when we are seeing big changes, and rarely when we are seeing small changes.
- [ ] Hmm the test set overfitting is kind of a problem. Maybe we should just never include those trajectories in the replay buffer? Yeah probably the right move. That way we have the low variance model estimate, whilst avoiding the issue of later models being better on the test set simply because they have been trained on it... Yeah drop the adding the samples to the test set.

- [ ] Less sharpening of temperature
- [ ] Try lower cpuct
- [ ] Higher exploration
- [ ] We are overfitting on the test evals.
- [ ] Go back to hyper params like v37 and then scale up and run on vast.ai
- [ ] What if you subtracted the root node value from all the child values before tanh? Then you immediately see if it is worse or better than expected.
- [ ] look at mcts tree across multiple moves, not just root

- [ ] Try MSE loss
- [ ] I screwed up the loose penalty I think
- [ ] Make network smaller and faster
- [ ] The tetris tspin stuff is still weird.
  - Investigate with make viz
- [ ] Fix the t-spin logic
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
  - Huber loss vs. MSE loss
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

Why is there so much duplicate code in tetris_core/src/inference/mod.rs?

## Prompts To Run

> In tetris_core/src/runtime/game_generator/runtime.rs do I have a worker just for saving files? Why? Like is_save_worker. Also wait am I writing to disk every 4 (GAME_COMMIT_BATCH_SIZE) games ending? That seems very slow.

> Is the slicing algorithm for loading up the gpu and efficient way of doing it? Any simpler way of ensuring the training data is on the gpu? Also why is my gpu not running out of memory, is like 2M training points not a crap ton of data. Or maybe not?

> When I do ctrl + c twice, I think this does not actually stop the wandb run properly and upload the current replay buffer and model, but just kills everything. I just want to stop the workers and wrap up, not just stop workers and die.

> In tetris_core/src/search/utils.rs why do we need:

    ```

        let mut overhang_fields: u32 = 0;
        let mut holes: u32 = 0;
        for x in 0..env.width {
            let mut seen_filled = false;
            for y in 0..env.height {
                let idx = y * env.width + x;
                let cell = board[idx];
                if cell != 0 {
                    seen_filled = true;
                    continue;
                }
                if seen_filled {
                    overhang_fields += 1;
                    if !reachable[idx] {
                        holes += 1;
                    }
                }
            }
        }
    ```

    Is holes not just 200 - blocks - visit_count. I.e. holes are what is left when you account for all the air you can visit and the placed blocks? But actually thinking of it, since we anyway need the loop for overhang, then kind of makes sense to just do the holes in there?

> Is the replay storage format the most efficient?

> I want a new metric that considers how much variance there is in the value estimate along the chosen trajectory. Like if the model was a perfect predictor, then the past cummulative attack + predicted future, should be fully constant throughout the trajectory. So I want to for each game, get the variance of this for the chosen actions, so variance over past cum + network prediction for each step of played game and return that. Or is variance like the best metric here, or what would you think would be useful to log here for this?

> Should we try to compress the policy targets for saving to disk? Like only saving the floats for the valid actions, rest is just zeros anyway right?

> Am I correctly currently taking the average over 100 steps when logging the step info to wanddb, like I do for games, or am I logging every single GPU step?

> I still cant see a full tree with reuse in the make viz. It just loads for ever please fix.

> Is there a cleaner way than what I am doing right now for syncing replay buffer to GPU?

> Why is tetris_core/src/replay/mod.rs its own file?

> Clean up dead code please around the repo

> Instead of encode_board_features should we just save that directly on the env?


## Python Code

### tetris_bot/config.py ✅

### tetris_bot/run_setup.py ✅

### scripts/train.py ✅

### tetris_bot/ml/loss.py ✅

# Backlog

- [ ] Is huber loss even better than MSE loss?
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
