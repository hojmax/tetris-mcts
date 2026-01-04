# Questions and Issues

## FIXED

### Argmax logic bug (CRITICAL)

The original code only selected an action if it had ALL visits:

```rust
result_policy[action_idx] = if child.visit_count() == total_visits { 1.0 } else { 0.0 };
```

**Fixed**: Now correctly finds the action with the most visits.

### Silent failures throughout

- `unwrap_or((0, 0, 0))` → Now uses `expect()` with clear error messages
- `Err(_) => break` → Now uses `expect()` for NN failures
- Various `_ => 0.0` and `_ => break` → Now `panic!("BUG: ...")` for unreachable code

### Constants file

Created `tetris_core/src/constants.rs` with:

- `BOARD_WIDTH`, `BOARD_HEIGHT`
- `NUM_PIECE_TYPES`, `QUEUE_SIZE`, `NUM_ROTATIONS`
- `DEFAULT_LOCK_DELAY_MS`, `DEFAULT_LOCK_MOVES`

### drop_last_n parameter removed

Since we're not doing discounting and just predict cumulative attack, no need to drop last N moves.

### Renamed get_all_placements → get_possible_placements

More accurate name - these are the possible placements, not all theoretical ones.

### Added states/attacks length assertion

```rust
debug_assert_eq!(states.len(), attacks.len(), "States and attacks should have same length");
```

### Chance nodes now properly track queue

**Previously**: `set_current_piece_type()` incorrectly changed the current piece.

**Fixed**: Now uses `push_queue_piece()` to add pieces to the END of the queue, which is the actual "chance" in Tetris - what piece appears next in the queue.

Changes:
- `expand_chance()` now calls `push_queue_piece(piece)` instead of `set_current_piece_type(piece)`
- The chance node represents which piece appears at the end of the visible queue

### Proper 7-bag tracking implemented

**Previously**: Always returned all 7 pieces as possible.

**Fixed**: Added proper 7-bag state tracking:
- `TetrisEnv.pieces_spawned` tracks total pieces drawn
- `get_possible_next_pieces()` computes which pieces could appear next based on 7-bag constraints
- `expand_action()` now uses `get_possible_next_pieces()` for accurate bag tracking

### T-spin kick detection now uses actual kick index

**Previously**: Used a heuristic based on counting rotations in the move sequence.

**Fixed**:
- `Placement` struct now has `last_kick_index` and `last_move_was_rotation` fields
- `try_rotate()` returns the actual kick index (0-4) used during rotation
- `find_all_placements()` tracks kick info through BFS and stores it in Placement
- `execute_placement()` uses the actual kick index for accurate T-spin detection

---

## MINOR (Keep or Remove)

### MCTSResult returns value

**Verdict**: Keep it. Useful for diagnostics.

### `__repr__` on MCTSResult

**Verdict**: Keep it. Useful for Python debugging.

### nn is Option

**Verdict**: Acceptable pattern for flexible initialization (load model later). Now fails loudly if missing during search.

### GameResult.num_moves

**Verdict**: Minor convenience. Could remove if wanted.

### Split mcts.rs

**Verdict**: Good idea but not blocking. File is ~970 lines. Suggested split:

- `mcts/mod.rs` - Main agent, search
- `mcts/config.rs` - MCTSConfig
- `mcts/action_space.rs` - ActionSpace
- `mcts/nodes.rs` - MCTSNode, DecisionNode, ChanceNode
- `mcts/training.rs` - TrainingExample, GameResult, play_game
- `mcts/utils.rs` - sample_dirichlet, sample_action

### Two placement functions

- `place_piece(x, y, rot)`: Simple, uses heuristic for T-spin
- `execute_placement(&placement)`: Uses move sequence for accurate T-spin

**Verdict**: Keep both. `place_piece` useful for simple testing, `execute_placement` for MCTS.

---

# Features we could add as input

- Number of holes in the board.
- Height of the tallest column.
- Number of empty columns.

# Optimizations

- Change piece representations to list of x,y instead of 4x4 matrix.
- I think this could be done way fastre:

```
/// Expand an action from a decision node (creates chance node)
    fn expand_action(&self, parent: &DecisionNode, action_idx: usize) -> MCTSNode {
        let mut new_state = parent.state.clone();

        // Get placement coordinates from action index
        let (x, y, rot) = self.action_space.index_to_placement(action_idx).unwrap_or((0, 0, 0));

        // Find the matching placement to get move sequence for T-spin detection
        let placements = new_state.get_all_placements();
        let attack = if let Some(placement) = placements.iter().find(|p| {
            p.piece.x == x && p.piece.y == y && p.piece.rotation == rot
        }) {
            // Use execute_placement for proper T-spin detection
```

If we know the x and y position and rotation, and just need the sequence of moves that made that position, then we don't need to generate all possible moves. Also I don't get why we even need rotations etc. to know what points to give. If we we a two line cleared and there is some block blcoking direct placement down of the t (so you could not just drop it in place), then there must have been a tslin. Or actually I guess this also depends on whether it was horisontal (like for a 3 line clear), but only cleared 2 lines because the third had a hole somewhere. I guess in this case, I am not quite sure what poitns this counts as? But anyhow, I don't quite understand what particular infomration is load bearing from the sequence of moves to place a piece, that we could not just gather from looking directly at the board?
Oh actuall

# Questions

Why does MCTSResult return a value?:

```
/// MCTS search result
#[pyclass]
#[derive(Clone)]
pub struct MCTSResult {
    /// Policy (visit counts normalized) over all 734 actions
    #[pyo3(get)]
    pub policy: Vec<f32>,
    /// Selected action index
    #[pyo3(get)]
    pub action: usize,
    /// Root value estimate
    #[pyo3(get)]
    pub value: f32,
    /// Number of simulations run
    #[pyo3(get)]
    pub num_simulations: u32,
}
```

Is this necessary?:

```
#[pymethods]
impl MCTSResult {
    fn __repr__(&self) -> String {
        format!(
            "MCTSResult(action={}, value={:.3}, simulations={})",
            self.action, self.value, self.num_simulations
        )
    }
}
```

Is the nn ever not supplied?:

```
/// MCTS Agent for Tetris
#[pyclass]
pub struct MCTSAgent {
    config: MCTSConfig,
    action_space: ActionSpace,
    /// Optional neural network for leaf evaluation (pure Rust mode)
    nn: Option<crate::nn::TetrisNN>,
}
```

Do we need to expose num moves here?:

```

/// Result from playing a full game
#[pyclass]
#[derive(Clone)]
pub struct GameResult {
    /// Training examples from this game
    #[pyo3(get)]
    pub examples: Vec<TrainingExample>,
    /// Total attack scored
    #[pyo3(get)]
    pub total_attack: u32,
    /// Number of moves played
    #[pyo3(get)]
    pub num_moves: u32,
}

```

I don't think we need to drop last n anymore:

```
/// Play a full game using MCTS with the loaded model
    ///
    /// All neural network inference happens in Rust. Returns training data.
    ///
    /// Args:
    ///     max_moves: Maximum moves per game (default 100)
    ///     add_noise: Whether to add Dirichlet noise (for exploration)
    ///     drop_last_n: Drop last N moves from training data (incomplete values)
    ///
    /// Returns:
    ///     GameResult with training examples, or None if no model loaded
    #[pyo3(signature = (max_moves=100, add_noise=true, drop_last_n=10))]
    pub fn play_game(
        &self,
        max_moves: u32,
        add_noise: bool,
        drop_last_n: u32,
    ) -> Option<GameResult> {
```

Becuase we don´t do discounting now, and just predict the future cummulative attack.

Could we have a rust file with constants like these:

```

        let mut env = TetrisEnv::new(10, 20);
```

Instead of hardcoding them around.

This should never happen. If it does we should make a loud error:

```
 // Get action mask
            let mask = crate::nn::get_action_mask(&env);
            if !mask.iter().any(|&x| x) {
                break; // No valid actions
            }
```

Are we raising these errors? We don't want to fail silently on any of these steps!!:

```
 // Get NN policy and value for root
            let (policy, value) = match nn.predict_masked(&env, move_idx as usize, &mask) {
                Ok(pv) => pv,
                Err(_) => break,
            };
```

Rename:
get_all_placements -> get_possible_placements

Why are we saving policy?:

```
            // Store state before making move
            states.push((env.clone(), move_idx, policy.clone(), mask.clone()));
```

We want to train on the final post-search policy, not initial estimate.

So lets simplify this:

```

            // Get NN policy and value for root
            let (policy, value) = match nn.predict_masked(&env, move_idx as usize, &mask) {
                Ok(pv) => pv,
                Err(_) => break,
            };

            // Store state before making move
            states.push((env.clone(), move_idx, policy.clone(), mask.clone()));

            // Run MCTS search
            let result = self.search_internal(&env, policy, value, add_noise, move_idx);
```

By just having all the nn calls inside of search_internal. No need to make the first of those outside the function and pass it in, or make some other function that starts the search or whatever.

What is going on here:

```

            let (x, y, rot) = self.action_space.index_to_placement(result.action).unwrap_or((0, 0, 0));
```

Returninig (0, 0, 0) would be real bad right? Something is wrong with the code then? We should not silently fail on these things!!!

Why do we need the placements check here when we already applied a mask?:

```
// Get action mask
            let mask = crate::nn::get_action_mask(&env);
            if !mask.iter().any(|&x| x) {
                break; // No valid actions
            }

            // Get NN policy and value for root
            let (policy, value) = match nn.predict_masked(&env, move_idx as usize, &mask) {
                Ok(pv) => pv,
                Err(_) => break,
            };

            // Store state before making move
            states.push((env.clone(), move_idx, policy.clone(), mask.clone()));

            // Run MCTS search
            let result = self.search_internal(&env, policy, value, add_noise, move_idx);

            // Execute the selected action
            let (x, y, rot) = self.action_space.index_to_placement(result.action).unwrap_or((0, 0, 0));
            let placements = env.get_all_placements();
            let attack = if let Some(placement) = placements.iter().find(|p| {
                p.piece.x == x && p.piece.y == y && p.piece.rotation == rot
            }) {
                env.execute_placement(placement)
            } else {
                env.place_piece(x, y, rot)
            };
            attacks.push(attack);
```

Would that not rule out illegal moves?

Yeah when we change the state saving we can just save it the right way first time instead of having to update:

```

            // Update stored policy with MCTS policy
            if let Some(last) = states.last_mut() {
                last.2 = result.policy;
            }
```

Why would there be fewer attacks than num states? why that check?:

```
// Compute value targets (cumulative attack from each position)
        let num_states = states.len();
        let mut values = vec![0.0f32; num_states];
        let mut cumulative = 0u32;
        for i in (0..num_states).rev() {
            if i < attacks.len() {
                cumulative += attacks[i];
            }
            values[i] = cumulative as f32;
        }
```

Why some here?:

```
 for _ in 0..num_games {
            if let Some(result) = self.play_game(max_moves, add_noise, drop_last_n) {
                all_examples.extend(result.examples);
            }
        }
```

Does it not always return a result?

We need to split up the huge MCTS.rs file. The files are very very large, and we should just split it out into multiple files with their own responsibility.

Why would there be no visits here?:

```

        if total_visits > 0 {
            for (&action_idx, child) in &root.children {
                if self.config.temperature == 0.0 {
                    // Argmax
                    result_policy[action_idx] = if child.visit_count() == total_visits { 1.0 } else { 0.0 };
                } else {
                    // Proportional to visit_count ^ (1/temp)
                    result_policy[action_idx] = (child.visit_count() as f32).powf(1.0 / self.config.temperature);
                }
            }

```

So probably no need for:

```
} else {
            // Use prior policy if no visits
            result_policy = policy;
        }
```

I dont get this:

```
 // Argmax
                    result_policy[action_idx] = if child.visit_count() == total_visits { 1.0 } else { 0.0 };
```

Why would the child.visit_count equal the total visits? Argmax is just taking the one with the most visits? And it would not neceassiraly be the case that it had visited only a single child on every single visit (that would actually be somewhat strange)

Also we only need to normalize when not argmaxing right?:

```

            // Normalize
            let sum: f32 = result_policy.iter().sum();
            if sum > 0.0 {
                for p in &mut result_policy {
                    *p /= sum;
                }
            }
```

Seems quite complex:

```
// Select action
        let action = if self.config.temperature == 0.0 {
            // Argmax
            result_policy.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).map(|(i, _)| i).unwrap_or(0)
```

When we already looped through the children and could just have kept track of most visited child if argmaxing i.e. temp=0

Why are we returning root value?:

```
        let root_value = if root.visit_count > 0 {
            root.value_sum / root.visit_count as f32
        } else {
            0.0
        };

        MCTSResult {
            policy: result_policy,
            action,
            value: root_value,
            num_simulations: self.config.num_simulations,
        }
```

This should never happen right:

```

            if node.valid_actions.is_empty() {
                self.backup_with_value(&path, 0.0);
                return;
            }
```

Like we should throw error if this is the case, but not even sure if we need to check this explicitely?

Why defaulting here?:

```
/ Traverse to child - get attack at this step
            let step_attack = match node.children.get(&action_idx) {
                Some(MCTSNode::Chance(chance_node)) => chance_node.attack as f32,
                _ => 0.0,
            };
```

Why break here?:

```
match chance_node.children.get_mut(&piece) {
                        Some(MCTSNode::Decision(decision_node)) => {
                            current = decision_node as *mut DecisionNode;
                        }
                        _ => break,
                    }
```

NO silent failures!:

```
    /// Evaluate a leaf state with the neural network
    fn evaluate_leaf(&self, env: &TetrisEnv, move_number: u32) -> f32 {
        if let Some(ref nn) = self.nn {
            // Use NN to evaluate
            let mask = crate::nn::get_action_mask(env);
            if let Ok((_, value)) = nn.predict_masked(env, move_number as usize, &mask) {
                return value;
            }
        }
        // No NN or evaluation failed - return 0 (no estimated future value)
        0.0
    }
```

Why are we doing fall backs here?:

```
 // Find the matching placement to get move sequence for T-spin detection
        let placements = new_state.get_all_placements();
        let attack = if let Some(placement) = placements.iter().find(|p| {
            p.piece.x == x && p.piece.y == y && p.piece.rotation == rot
        }) {
            // Use execute_placement for proper T-spin detection
            new_state.execute_placement(placement)
        } else {
            // Fallback to direct placement
            new_state.place_piece(x, y, rot)
        };
```

Why would this need fallbacks?

We don't want simplified!!! Track the possible pieces!!:

```

        // Compute remaining bag (simplified - just use all pieces)
        // In a real implementation, we'd track the 7-bag state
        let bag_remaining: Vec<usize> = (0..NUM_PIECE_TYPES).collect();

```

So yeah maybe the round robin implementation I suggested is wrong and needs to be replaced with something that actually produces sensible chance pieces. This can be exactly inferred from the queue of pieces having been observed. I.e. at what place we are in the current queue, and how many are left in the next one, and so what pieces are possible to be observed next etc.

WHen would the network ever not be available??:

```
 /// Expand a chance node for a specific piece (creates decision node)
    fn expand_chance(&self, parent: &ChanceNode, piece: usize, move_number: u32) -> MCTSNode {
        let mut new_state = parent.state.clone();

        // Set the current piece to the sampled piece type
        // This allows MCTS to explore different possible next pieces
        new_state.set_current_piece_type(piece);

        let mut node = DecisionNode::new(new_state.clone(), move_number);

        // Set priors from neural network if available, otherwise uniform
        if let Some(ref nn) = self.nn {
            let mask = crate::nn::get_action_mask(&new_state);
            if let Ok((policy, _)) = nn.predict_masked(&new_state, move_number as usize, &mask) {
                node.set_priors(&policy);
            } else {
                let uniform = 1.0 / node.valid_actions.len().max(1) as f32;
                node.action_priors = vec![uniform; node.valid_actions.len()];
            }
        } else {
            let uniform = 1.0 / node.valid_actions.len().max(1) as f32;
            node.action_priors = vec![uniform; node.valid_actions.len()];
        }

        MCTSNode::Decision(node)
    }

```

Why would path be empty?:

```
fn backup_with_value(&self, path: &[(*mut DecisionNode, usize, f32)], leaf_value: f32) {
        if path.is_empty() {
            return;
        }
```

Are we double counting value here? Since we both add for the parent node and children, and than loop over the path? Seems to me like most nodes in the path exccept the first and last get their value and visit count doubly incremented:

```
     }

        // Compute total value = attack along path + leaf value estimate
        let total_attack: f32 = path.iter().map(|(_, _, reward)| reward).sum();
        let total_value = total_attack + leaf_value;

        // All nodes get the same total value
        for &(node_ptr, action_idx, _) in path.iter() {
            let node = unsafe { &mut *node_ptr };

            // Update child stats (the action we took from this node)
            if let Some(child) = node.children.get_mut(&action_idx) {
                match child {
                    MCTSNode::Decision(d) => {
                        d.visit_count += 1;
                        d.value_sum += total_value;
                    }
                    MCTSNode::Chance(c) => {
                        c.visit_count += 1;
                        c.value_sum += total_value;
                    }
                }
            }

            // Update parent (decision node) stats
            node.visit_count += 1;
            node.value_sum += total_value;
        }
```

Ahh I think the way we have done chance nodes is all wrong:

```

    /// Set the current piece to a specific type.
    ///
    /// This is used by MCTS to explore different possible next pieces
    /// at chance nodes. The piece spawns at the standard spawn position.
    ///
    /// Args:
    ///     piece_type: The piece type (0-6: I, O, T, S, Z, J, L)
    pub fn set_current_piece_type(&mut self, piece_type: usize) {
        if piece_type < 7 && !self.game_over {
            let spawn_x = (self.width as i32 - 4) / 2;
            let spawn_y = 0;
            self.current_piece = Some(Piece {
                piece_type,
                x: spawn_x,
                y: spawn_y,
                rotation: 0,
            });
        }
    }

```

We are NOT chancing the current piece. We are actually adding simulated pieces to the END of the queue. I.e. say we know what the next 5 pieces are. Then after we have placed a single piece, the next piece is unobserved (but restrained by the number of pieces left in that bag/chunk). So we have say 2 possibilies in this example, I and O. And so now we make two chance nodes, one where we placed the piece, and moved all the pieces forward in the queue, and then say I for the now visible 5th position in the queue, and one where we say O.

Why do we have both of these functions?:

```
   /// Directly place the current piece at the specified position and lock it.
    ///
    /// This is more efficient than stepping through individual moves when you
    /// already know the final placement from get_all_placements().
    ///
    /// Args:
    ///     x: The x position (column) for the piece
    ///     y: The y position (row) for the piece
    ///     rotation: The rotation state (0-3)
    ///
    /// Returns:
    ///     The attack gained from this placement (including line clears)
    ///
    /// Note: For proper T-spin detection including mini vs proper distinction,
    /// use execute_placement() with the full Placement object instead.
    pub fn place_piece(&mut self, x: i32, y: i32, rotation: usize) -> u32 {
        // Delegate to internal method with no move info
        self.place_piece_internal(x, y, rotation, None)
    }

    /// Execute a placement from get_all_placements() with full T-spin detection.
    ///
    /// This uses the move sequence to properly detect T-spins including
    /// the mini vs proper distinction based on which kick was used.
    ///
    /// Args:
    ///     placement: A Placement object from get_all_placements()
    ///
    /// Returns:
    ///     The attack gained from this placement (including line clears)
    pub fn execute_placement(&mut self, placement: &crate::moves::Placement) -> u32 {
        let x = placement.piece.x;
        let y = placement.piece.y;
        let rotation = placement.piece.rotation;
        self.place_piece_internal(x, y, rotation, Some(&placement.moves))
    }
```

Don't we only need one of them?

Why are we not throwing errors or something here:

```
/// Internal placement logic with optional move sequence for T-spin detection
    fn place_piece_internal(&mut self, x: i32, y: i32, rotation: usize, moves: Option<&[u8]>) -> u32 {
        if self.game_over {
            return 0;
        }

        if let Some(ref piece) = self.current_piece {
            let piece_type = piece.piece_type;
            let shape = &TETROMINOS[piece_type][rotation % 4];

            // Verify the position is valid
            if !self.is_valid_position_for_shape(shape, x, y) {
                return 0;
            }
```

Like an invalid piece placement just returns 0 attack??

When would this happen?:

```

            } else {
                // No move info - use heuristic for T pieces
                if piece_type == 5 {
                    self.last_move_was_rotation = true;
                    self.last_kick_index = 0;
                } else {
                    self.last_move_was_rotation = false;
                    self.last_kick_index = 0;
                }
            }
```

Hmm this does not sound that accurate:

```
    // Count consecutive rotations at the end to estimate kick index
                // More rotations in sequence = likely used a kick
                if self.last_move_was_rotation && piece_type == 5 {
                    let rotation_count = move_list.iter().rev()
                        .take_while(|&&m| m == 4 || m == 5 || m == 6)
                        .filter(|&&m| m == 4 || m == 5)
                        .count();
```

Why is this only "likely"? Are we gonna be scoring this correctly?
