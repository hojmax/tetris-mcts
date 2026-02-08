# Deep Review

## Issues

1:
```

  3. Medium: clear_lines_internal() rebuilds full board every lock, even when no
     lines clear

  - tetris_core/src/env/clearing.rs:96 through tetris_core/src/env/
    clearing.rs:129 always allocates/rebuilds row vectors.
  - No fast path for “no full rows”.
  - Impact: avoidable per-move overhead in hot game loop.

```
2:
```
  4. Medium: repeated O(n) placement scans on hot path (duplicate logic)

  - tetris_core/src/mcts/search.rs:185 does linear .find(...) by (x,y,rot)
    during expansion.
  - tetris_core/src/mcts/agent.rs:116 does another linear .find(...) for chosen
    action.
  - tetris_core/src/env/pymethods.rs:431 already has execute_action_index()
    path.
  - Impact: unnecessary repeated scans and duplicated conversion logic in MCTS-
    critical paths.

```
3:
```
  5. Medium: unnecessary board cloning in inference/training-example
     construction

  - tetris_core/src/nn.rs:79 uses env.get_board() (clones board) every
    prediction.
  - tetris_core/src/mcts/agent.rs:198 uses state.get_board() (clone) before
    flattening.
  - Impact: repeated allocations/copies in very hot loops.

```
and:
```
  6. Low: dead/unused field in core node struct

  - tetris_core/src/mcts/nodes.rs:66 DecisionNode.prior is set at tetris_core/
    src/mcts/nodes.rs:96 and only surfaced in export (tetris_core/src/mcts/
    export.rs:30), not used in selection/backprop.
  - Impact: extra state and cognitive load without behavioral effect.

```


## tetris_core/src/scoring.rs ✅

Are we throwing an error on "\_ => ClearType::None, // Invalid"? in determine_clear_type

I don't understand this distinction?:

```
        (1, true, true) => ClearType::TSpinMiniSingle,
        (1, true, false) => ClearType::TSpinSingle,
```

What is a mini t spin?

What is this for?:

```


impl Default for AttackResult {
    fn default() -> Self {
        Self::new()
    }
}
```

## tetris_core/src/piece.rs ✅

Is tetris_core/src/piece.rs efficient? The look ups and all that?

Why the heck do we have colors in .rs code?:

```

/// Colors for each tetromino (RGB) - matching Jstris style
pub const COLORS: [(u8, u8, u8); 7] = [
    (93, 173, 212), // I - Light blue/Cyan
    (219, 174, 63), // O - Golden yellow
    (178, 74, 156), // T - Magenta
    (114, 184, 65), // S - Green
    (204, 65, 65),  // Z - Red
    (59, 84, 165),  // J - Blue
    (227, 127, 59), // L - Orange
];
```

All visualization sort of stuff should go into the python code. Only environment logic, and mcts in rust.

Please update CLAUDE.md to make this clear:
Rust = Env logic, MCTS
Python = Training, Visualization

Like there should not be getting any colors from rust and all that. get_color_for_type is definitely a mistake and all other color related stuff in rust.

Why does this default to x position?:

```

#[pymethods]
impl Piece {
    #[new]
    pub fn new(piece_type: usize) -> Self {
        Piece {
            piece_type,
            x: 3,
            y: 0,
            rotation: 0,
        }
    }

    pub fn get_color(&self) -> (u8, u8, u8) {
        COLORS[self.piece_type]
    }

    pub fn get_cells(&self) -> Vec<(i32, i32)> {
        get_cells(self.piece_type, self.rotation, self.x, self.y).to_vec()
    }
}
```

Is this not dependent on whether it is an I or a O or any of the other pieces? Maybe we should move that logic into piece .rs, to not have to do overrides and just being able to build one.

## tetris_core/src/nn.rs ✅

Why can't encode_state return the correct formatting and typing like:

```

        let board =
            tract_ndarray::Array4::from_shape_vec((1, 1, BOARD_HEIGHT, BOARD_WIDTH), board_tensor)?
                .into_tensor();

        let aux =
            tract_ndarray::Array2::from_shape_vec((1, AUX_FEATURES), aux_tensor)?.into_tensor();

```

IT could just return tensors right instead of formatting back and forth?

Is this correct softmax?:

```

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

```

This is totally wrong:

```

/// Get action mask from environment
pub fn get_action_mask(env: &TetrisEnv) -> Vec<bool> {
    use crate::mcts::{get_action_space, NUM_ACTIONS};

    let action_space = get_action_space();
    let placements = env.get_possible_placements();

    let mut mask = vec![false; NUM_ACTIONS];

    for p in placements {
        if let Some(idx) = action_space.placement_to_index(p.piece.x, p.piece.y, p.piece.rotation) {
            mask[idx] = true;
        }
    }

    mask
}
```

We should not be passing around x, y and rotation. We should only be passing around a move index, and so you could just directly index based on that. This is NOT just for nn.rs, but in general in the code base. Like we should just have a function (we already have right?) for index to x,y,rotation, and then use that when we actually need the x,y,rotation, and maybe a function for x,y,rotation to index.

# tetris_core/src/moves.rs

Why is the action not an to_u8 in the first place?:

```

impl Action {
    pub fn to_u8(self) -> u8 {
        self as u8
    }
}
```

Why is this not an u8:

```

    /// The column where the piece lands (leftmost cell x coordinate)
    #[pyo3(get)]
    pub column: i32,
```

In general, why are we not using u8 way more all over the place? like most things in the program are binary, or board state, that can be encoded in 8 states (7 different types of pieces, and 1 empty state). I guess this is actually 3 bits, so we could also just do it with a 4 bit type. Not sure if this is a useful optimization, maybe tiny speedup / memory decrease usage? Maybe nice for huge mcts trees.

I don't full understand this, explain:

```

    /// The kick index used for the final rotation (0 = no kick, 1-4 = kick used)
    /// This is needed for proper T-spin detection (kick 4 = always full T-spin)
    #[pyo3(get)]
    pub last_kick_index: usize,
```

Maybe we can get rid of this:

```

    /// Pre-computed action index for fast lookup (0-733)
    #[pyo3(get)]
    pub action_index: usize,
}
```

When we just encode everything as indeces instead of x and y and rotation anyways. not sure.

Do we literally ever use this?:

```

#[pymethods]
impl Placement {
    fn __repr__(&self) -> String {
        format!(
            "Placement(col={}, rot={}, moves={:?})",
            self.column, self.rotation, self.moves
        )
    }
}
```

Or can we delete all the **repr** stuff in .rs.

# Next Steps

- [ ] Fix hold move bug in mcts. Like why are we tracking x and y and rotation, and just raw move indices? Why are we not passing a single int around corresponding to a specific move? We have the lookup for this. And hold should be a move.
- [ ] Take an interestnig state like tetris_mcts/scripts/outputs/game_3194.gif at step 14 (oppurtinity for t spin), and see how the model evaluates that state? Is it not reached or something?
- [ ] No need to show Value and attack in tetris_mcts/scripts/outputs/game_3194.gif? They are the same? Just skip the value one, and show the attack. They are reading from the same field right? Or where is the attack ocming from?
- [ ] When inspecting games, with the training viewer, I need to see network predictions. At least value predictions.
- [ ] What happened with the LR lol? Why did it go so low, even though we trained with 0.5 lr min?
- [ ] Do we have a mismatch between the action somewhere? Like training on one set of indeces and taking actions with another? This would shift all the agents agents / randomize it? Need to check for this.
- [ ] tetris_mcts/scripts/inspect_training_data.py not showing can hold?
- [ ] game/avg_moves, game/max_moves and not eval/...
- [ ] I don't think the eval/trajectory matches the eval/max_attack?? Is the replay correct?
- [ ] Verify that the inference and training netwroks are tacking the exavt same inputs in the exavt same ordering and the exavt same scaling and values and all that. Make unit tests for this as well. Also the masked softmax and all that. We need tests that compare the two netwrok files and check equal outputs. Might need to be a pytest? Add pytest for this that runs both and compares inputs and outputs that are written to files or something like that.

# Backlog

- [ ] Proper network split caching. Caching board CNN head, and optimizing such that we only run last part of network.
- [ ] Do we need a larger network?
- [ ] What alternative hardware could we run on?
- [ ] Would an instance with a GPU be better? What is the major bottleneck, CPU or GPU? Like maybe AWS instance with a ton of CPUs for deep tree search.
- [ ] Looking at training data
- [ ] Caching board representation from network in inference.
- [ ] Optimize int4 and int8 everywhere.
- [ ] More rust profiling
- [ ] Better splitting up of rust between the environment and the MCTS. Two different packages / folders.
- [ ] Visualizing MCTS search and verifying correctness
- [ ] Play the tetris game to ensure the environment is working correctly
- [ ] Save a full rollout tree from during training and inspect it with `make viz` tool
- [ ] Reading through and validating all code
- [ ] Stress test next possible pieces with unit tests, like that it can do wild twists and stuff and that it correctly decides those possible locations
- [ ] Benchmarking and improving speed of MCTS search
- [ ] Testing speed on different hardware then the Macbook Air
