# Next Steps

- [ ] New Metric: Is past attack + future attack roughly stable? Maybe we log per game variance in this estimate.
- [ ] I don't want random move sampling in the the last ~10 moves.
- [ ] Why is the GPU going cold repeatedly like 50% of the time?
- [ ] Why don't the final moves look full optimal? Like just clean out the board in the last 5 moves? Surely that is highest attack possible? Instead it does random nonsense.
- [ ] Why don't we always have 50 piece episodes??????

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

Why is there so much duplicate code in tetris_core/src/inference/mod.rs?

## Prompts To Run

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

> tetris_core/src/replay/npz.rs seems really repetitive. Is there not a library one could use for all this? Or some way of condensing it?

> I want a new metric that considers how much variance there is in the value estimate along the chosen trajectory. Like if the model was a perfect predictor, then the past cummulative attack + predicted future, should be fully constant throughout the trajectory. So I want to for each game, get the variance of this for the chosen actions, so variance over past cum + network prediction for each step of played game and return that. Or is variance like the best metric here, or what would you think would be useful to log here for this?

> Should we try to compress the policy targets for saving to disk? Like only saving the floats for the valid actions, rest is just zeros anyway right?

> Am I correctly currently taking the average over 100 steps when logging the step info to wanddb, like I do for games, or am I logging every single GPU step?

> I still cant see a full tree with reuse in the make viz. It just loads for ever please fix.

> Is there a cleaner way than what I am doing right now for syncing replay buffer to GPU?

> Why is tetris_core/src/replay/mod.rs its own file?

> Clean up dead code please around the repo

> How could I improve my model architecture?

> Why is there so much duplicate code in tetris_core/src/inference/mod.rs? And why are there so many different functions in mod.rs? Seems like a lot of code specialized for a single use case that could have been generalized?

> Why is placement count passed in here?:

    ```

    /// Encode a TetrisEnv state into neural network input tensors
    #[cfg(test)]
    fn encode_state(
        env: &TetrisEnv,
        placement_count: usize,
        max_placements: usize,
    ) -> TractResult<(Tensor, Tensor)> {
        let (board_tensor, aux_tensor) = encode_state_features(env, placement_count, max_placements)?;
        let board =
            tract_ndarray::Array4::from_shape_vec((1, 1, BOARD_HEIGHT, BOARD_WIDTH), board_tensor)?
                .into_tensor();
        let aux = tract_ndarray::Array2::from_shape_vec((1, AUX_FEATURES), aux_tensor)?.into_tensor();
        Ok((board, aux))
    }
    ```
    Is that not encoded in the env? Why is this not tracked on the env in general?

> Instead of encode_board_features should we just save that directly on the env?

> Why the hell is there fallback here?:

```

    let current_piece = env.get_current_piece().map(|p| p.piece_type).unwrap_or(0);
```

We don't want silent fallbacks when unexpected things happen!!! Please look through the whole codebase for similar cases, and fix them. Things failing loudly is fine and good! Also like why a max??:

```

pub fn denormalize_combo_feature(combo_feature: f32) -> u32 {
    (combo_feature.max(0.0) * COMBO_NORMALIZATION_MAX as f32).round() as u32
}
```

> I want to do a quicktest type of thing on an actual replay buffer. So the test is pointed to a replay buffer I track in git, with like 10K states, and we have a bunch of consistency checks there on the replay buffer which we check whether hold in practice. Suggest a long list of potential consistency checks we could do on the replay buffer.

> Is the minus the max logit just a way of making the calculation more stable? or what is it doing?:

```

/// Softmax with mask (invalid actions get 0 probability)
pub fn masked_softmax(logits: &[f32], mask: &[bool]) -> Vec<f32> {
    let max_logit = logits
        .iter()
        .zip(mask.iter())
        .filter(|(_, &m)| m)
        .map(|(&x, _)| x)
        .fold(f32::NEG_INFINITY, f32::max);

    let mut result = vec![0.0; logits.len()];
    let mut sum = 0.0;

    for (i, (&logit, &valid)) in logits.iter().zip(mask.iter()).enumerate() {
        if valid {
            let exp_val = (logit - max_logit).exp();
            result[i] = exp_val;
            sum += exp_val;
        }
    }

    if sum > 0.0 {
        for x in &mut result {
            *x /= sum;
        }
    }

    result
```

> Lets try to do a memory .md file. In the repo, make a memories folder. Then have a main MEMORY.md file. Every time you learn something or get feedback from me, write to that. Like something goes wrong, you figure out how to call that command correctly write it down. Something was uintutivie and you had to think alot to figure it out, write down the final realization. Something was hard to find, write it down. Anything else, write it down. Do this liberally. Now once the MEMORY.md hits 200 lines, you do compaction, and group together relevant stuff and write out into other files .md in that memorieis folder. I want every session of you that starts to always start by reading MEMORY.md. For the long term files, if there already exists a file that matches the theme of the chunck of short term stuff that you are compacting, add to that. If not, just make a new one with a descriptive name. Now add these rules to AGENTS.md and make the memories folder and file.

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
