//! Monte Carlo Tree Search for Tetris
//!
//! Implements AlphaZero-style MCTS with:
//! - Decision nodes (player moves)
//! - Chance nodes (piece spawns from 7-bag)
//! - Neural network priors
//! - PUCT selection

use pyo3::prelude::*;
use rand::prelude::*;
use rand_distr::{Distribution, Gamma};
use std::collections::HashMap;

use crate::env::TetrisEnv;
use crate::piece::NUM_PIECE_TYPES;

/// Configuration for MCTS
#[pyclass]
#[derive(Clone, Debug)]
pub struct MCTSConfig {
    /// Number of simulations per move
    #[pyo3(get, set)]
    pub num_simulations: u32,
    /// PUCT exploration constant
    #[pyo3(get, set)]
    pub c_puct: f32,
    /// Temperature for action selection (0 = argmax, higher = more exploration)
    #[pyo3(get, set)]
    pub temperature: f32,
    /// Dirichlet noise alpha (for root exploration)
    #[pyo3(get, set)]
    pub dirichlet_alpha: f32,
    /// Dirichlet noise weight (epsilon)
    #[pyo3(get, set)]
    pub dirichlet_epsilon: f32,
}

#[pymethods]
impl MCTSConfig {
    #[new]
    pub fn new() -> Self {
        MCTSConfig {
            num_simulations: 100,
            c_puct: 1.5,
            temperature: 1.0,
            dirichlet_alpha: 0.15,
            dirichlet_epsilon: 0.25,
        }
    }
}

impl Default for MCTSConfig {
    fn default() -> Self {
        MCTSConfig::new()
    }
}

/// Action index mapping
/// Maps (x, y, rotation) to action index 0-733
/// Built at module load time to match Python's action_space.py
#[derive(Clone)]
pub struct ActionSpace {
    pub action_to_placement: Vec<(i32, i32, usize)>, // (x, y, rotation)
    pub placement_to_action: HashMap<(i32, i32, usize), usize>,
}

impl ActionSpace {
    pub fn new() -> Self {
        let mut action_to_placement = Vec::new();
        let mut placement_to_action = HashMap::new();

        // Same logic as Python's action_space.py
        let x_min = -3i32;
        let x_max = 10i32;
        let y_min = -3i32;
        let y_max = 20i32;

        // Check which positions are valid for at least one piece
        let mut valid_positions: Vec<(i32, i32, usize)> = Vec::new();

        for y in y_min..y_max {
            for x in x_min..x_max {
                for rot in 0..4 {
                    // Check if any piece fits at this position on an empty board
                    for piece_type in 0..NUM_PIECE_TYPES {
                        if Self::is_valid_position_empty_board(piece_type, rot, x, y) {
                            valid_positions.push((x, y, rot));
                            break;
                        }
                    }
                }
            }
        }

        // Sort by rotation, then y, then x (to match Python)
        valid_positions.sort_by_key(|&(x, y, rot)| (rot, y, x));

        for (idx, pos) in valid_positions.iter().enumerate() {
            action_to_placement.push(*pos);
            placement_to_action.insert(*pos, idx);
        }

        ActionSpace {
            action_to_placement,
            placement_to_action,
        }
    }

    fn is_valid_position_empty_board(piece_type: usize, rotation: usize, x: i32, y: i32) -> bool {
        let shape = &crate::piece::TETROMINOS[piece_type][rotation];
        for dy in 0..4 {
            for dx in 0..4 {
                if shape[dy][dx] == 1 {
                    let cx = x + dx as i32;
                    let cy = y + dy as i32;
                    if cx < 0 || cx >= 10 || cy < 0 || cy >= 20 {
                        return false;
                    }
                }
            }
        }
        true
    }

    pub fn num_actions(&self) -> usize {
        self.action_to_placement.len()
    }

    pub fn placement_to_index(&self, x: i32, y: i32, rotation: usize) -> Option<usize> {
        self.placement_to_action.get(&(x, y, rotation)).copied()
    }

    pub fn index_to_placement(&self, idx: usize) -> Option<(i32, i32, usize)> {
        self.action_to_placement.get(idx).copied()
    }
}

impl Default for ActionSpace {
    fn default() -> Self {
        ActionSpace::new()
    }
}

/// Get the global action space
/// Note: Creates a new ActionSpace each call. In production, use lazy_static or OnceCell.
pub fn get_action_space() -> ActionSpace {
    ActionSpace::new()
}

/// Number of actions in the action space
pub const NUM_ACTIONS: usize = 734;

/// MCTS Node types
#[derive(Clone)]
pub enum MCTSNode {
    /// Decision node - player chooses an action
    Decision(DecisionNode),
    /// Chance node - random piece spawn
    Chance(ChanceNode),
}

/// Decision node where player chooses an action
#[derive(Clone)]
pub struct DecisionNode {
    /// Game state at this node
    pub state: TetrisEnv,
    /// Visit count
    pub visit_count: u32,
    /// Sum of values from all visits
    pub value_sum: f32,
    /// Prior probability from neural network
    pub prior: f32,
    /// Children: action index -> child node (ChanceNode after action)
    pub children: HashMap<usize, MCTSNode>,
    /// Valid action indices for this state
    pub valid_actions: Vec<usize>,
    /// Cached priors for valid actions (from neural network)
    pub action_priors: Vec<f32>,
    /// Whether this is a terminal state
    pub is_terminal: bool,
    /// Move number in the game
    pub move_number: u32,
}

/// Chance node for stochastic piece spawns
#[derive(Clone)]
pub struct ChanceNode {
    /// Game state (after piece placement, before new piece spawn)
    pub state: TetrisEnv,
    /// Visit count
    pub visit_count: u32,
    /// Sum of values from all visits
    pub value_sum: f32,
    /// Children: piece type -> child DecisionNode
    pub children: HashMap<usize, MCTSNode>,
    /// Attack gained from the action that led to this node
    pub attack: u32,
    /// Pieces remaining in current bag (for probability computation)
    pub bag_remaining: Vec<usize>,
    /// Randomized order for round-robin piece selection
    pub piece_order: Vec<usize>,
    /// Current index in round-robin sequence
    pub round_robin_idx: usize,
}

impl MCTSNode {
    /// Get visit count
    pub fn visit_count(&self) -> u32 {
        match self {
            MCTSNode::Decision(n) => n.visit_count,
            MCTSNode::Chance(n) => n.visit_count,
        }
    }

    /// Get mean value
    pub fn mean_value(&self) -> f32 {
        match self {
            MCTSNode::Decision(n) => {
                if n.visit_count > 0 {
                    n.value_sum / n.visit_count as f32
                } else {
                    0.0
                }
            }
            MCTSNode::Chance(n) => {
                if n.visit_count > 0 {
                    n.value_sum / n.visit_count as f32
                } else {
                    0.0
                }
            }
        }
    }
}

impl DecisionNode {
    pub fn new(state: TetrisEnv, move_number: u32) -> Self {
        let is_terminal = state.game_over;

        // Get valid actions
        let valid_actions = if is_terminal {
            Vec::new()
        } else {
            get_valid_action_indices(&state)
        };

        DecisionNode {
            state,
            visit_count: 0,
            value_sum: 0.0,
            prior: 1.0,
            children: HashMap::new(),
            valid_actions,
            action_priors: Vec::new(),
            is_terminal,
            move_number,
        }
    }

    /// Set priors from neural network output
    pub fn set_priors(&mut self, policy: &[f32]) {
        self.action_priors = self.valid_actions.iter().map(|&idx| policy[idx]).collect();

        // Normalize priors over valid actions
        let sum: f32 = self.action_priors.iter().sum();
        for p in &mut self.action_priors {
            *p /= sum;
        }
    }

    /// Add Dirichlet noise to priors (for root exploration)
    pub fn add_dirichlet_noise(&mut self, alpha: f32, epsilon: f32) {
        let noise = sample_dirichlet(alpha, self.action_priors.len());
        for (prior, n) in self.action_priors.iter_mut().zip(noise.iter()) {
            *prior = (1.0 - epsilon) * *prior + epsilon * n;
        }
    }

    /// Select best action using PUCT formula
    pub fn select_action(&self, c_puct: f32) -> usize {
        let sqrt_total = (self.visit_count as f32).sqrt();
        let mut best_action = self.valid_actions[0];
        let mut best_value = f32::NEG_INFINITY;

        for (i, &action_idx) in self.valid_actions.iter().enumerate() {
            let prior = self.action_priors[i];

            let (q, n) = if let Some(child) = self.children.get(&action_idx) {
                (child.mean_value(), child.visit_count())
            } else {
                (0.0, 0)
            };

            // PUCT formula: Q + c * P * sqrt(N_parent) / (1 + N_child)
            let u = c_puct * prior * sqrt_total / (1.0 + n as f32);
            let value = q + u;

            if value > best_value {
                best_value = value;
                best_action = action_idx;
            }
        }

        best_action
    }
}

impl ChanceNode {
    pub fn new(state: TetrisEnv, attack: u32, bag_remaining: Vec<usize>) -> Self {
        let mut rng = thread_rng();

        // Create randomized piece order for round-robin
        let mut piece_order: Vec<usize> = if bag_remaining.is_empty() {
            // New bag - all pieces
            (0..NUM_PIECE_TYPES).collect()
        } else {
            bag_remaining.clone()
        };
        piece_order.shuffle(&mut rng);

        ChanceNode {
            state,
            visit_count: 0,
            value_sum: 0.0,
            children: HashMap::new(),
            attack,
            bag_remaining,
            piece_order,
            round_robin_idx: 0,
        }
    }

    /// Select next piece using round-robin on randomized order
    /// Returns the piece type and advances the index
    pub fn select_piece_round_robin(&mut self) -> usize {
        let piece = self.piece_order[self.round_robin_idx % self.piece_order.len()];
        self.round_robin_idx += 1;
        piece
    }
}

/// Get valid action indices for a state
pub fn get_valid_action_indices(env: &TetrisEnv) -> Vec<usize> {
    let action_space = get_action_space();
    let placements = env.get_all_placements();

    let mut indices = Vec::new();
    for p in placements {
        let piece = &p.piece;
        if let Some(idx) = action_space.placement_to_index(piece.x, piece.y, piece.rotation) {
            indices.push(idx);
        }
    }

    indices
}

/// Sample from Dirichlet distribution
pub fn sample_dirichlet(alpha: f32, n: usize) -> Vec<f32> {
    let mut rng = thread_rng();
    let gamma = Gamma::new(alpha as f64, 1.0).unwrap();

    let samples: Vec<f64> = (0..n).map(|_| gamma.sample(&mut rng)).collect();
    let sum: f64 = samples.iter().sum();

    samples.into_iter().map(|x| (x / sum) as f32).collect()
}

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

#[pymethods]
impl MCTSResult {
    fn __repr__(&self) -> String {
        format!(
            "MCTSResult(action={}, value={:.3}, simulations={})",
            self.action, self.value, self.num_simulations
        )
    }
}

/// MCTS Agent for Tetris
#[pyclass]
pub struct MCTSAgent {
    config: MCTSConfig,
    action_space: ActionSpace,
    /// Optional neural network for leaf evaluation (pure Rust mode)
    nn: Option<crate::nn::TetrisNN>,
}

/// Training example returned from self-play
#[pyclass]
#[derive(Clone)]
pub struct TrainingExample {
    /// Board state flattened (20*10 = 200 values, 0 or 1)
    #[pyo3(get)]
    pub board: Vec<u8>,
    /// Current piece type (0-6)
    #[pyo3(get)]
    pub current_piece: usize,
    /// Hold piece type (0-6) or 7 if empty
    #[pyo3(get)]
    pub hold_piece: usize,
    /// Whether hold is available
    #[pyo3(get)]
    pub hold_available: bool,
    /// Next queue (up to 5 piece types)
    #[pyo3(get)]
    pub next_queue: Vec<usize>,
    /// Move number (0-99)
    #[pyo3(get)]
    pub move_number: u32,
    /// MCTS policy target (734 values)
    #[pyo3(get)]
    pub policy: Vec<f32>,
    /// Value target (cumulative attack from this point)
    #[pyo3(get)]
    pub value: f32,
    /// Action mask (734 values, true = valid)
    #[pyo3(get)]
    pub action_mask: Vec<bool>,
}

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

#[pymethods]
impl MCTSAgent {
    #[new]
    pub fn new(config: MCTSConfig) -> Self {
        MCTSAgent {
            config,
            action_space: ActionSpace::new(),
            nn: None,
        }
    }

    /// Load a neural network model
    ///
    /// Args:
    ///     path: Path to ONNX model file
    ///
    /// Returns:
    ///     True if loaded successfully, False otherwise
    pub fn load_model(&mut self, path: &str) -> bool {
        match crate::nn::TetrisNN::load(path) {
            Ok(nn) => {
                self.nn = Some(nn);
                true
            }
            Err(e) => {
                eprintln!("Failed to load model: {}", e);
                false
            }
        }
    }

    /// Check if a neural network model is loaded
    pub fn has_model(&self) -> bool {
        self.nn.is_some()
    }

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
        let nn = self.nn.as_ref()?;

        let mut env = TetrisEnv::new(10, 20);
        let mut states: Vec<(TetrisEnv, u32, Vec<f32>, Vec<bool>)> = Vec::new();
        let mut attacks: Vec<u32> = Vec::new();

        for move_idx in 0..max_moves {
            if env.game_over {
                break;
            }

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

            // Update stored policy with MCTS policy
            if let Some(last) = states.last_mut() {
                last.2 = result.policy;
            }
        }

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

        // Build training examples (drop last N moves)
        let usable = num_states.saturating_sub(drop_last_n as usize);
        let mut examples = Vec::with_capacity(usable);

        for i in 0..usable {
            let (ref state, move_num, ref policy, ref mask) = states[i];

            let board: Vec<u8> = state.get_board()
                .iter()
                .flat_map(|row| row.iter().copied())
                .collect();

            let current_piece = state.get_current_piece()
                .map(|p| p.piece_type)
                .unwrap_or(0);

            let hold_piece = state.get_hold_piece()
                .map(|p| p.piece_type)
                .unwrap_or(7); // 7 = empty

            let hold_available = !state.is_hold_used();
            let next_queue = state.get_queue(5);

            examples.push(TrainingExample {
                board,
                current_piece,
                hold_piece,
                hold_available,
                next_queue,
                move_number: move_num,
                policy: policy.clone(),
                value: values[i],
                action_mask: mask.clone(),
            });
        }

        let total_attack: u32 = attacks.iter().sum();
        let num_moves = states.len() as u32;

        Some(GameResult {
            examples,
            total_attack,
            num_moves,
        })
    }

    /// Generate multiple games of training data
    ///
    /// Args:
    ///     num_games: Number of games to play
    ///     max_moves: Maximum moves per game
    ///     add_noise: Whether to add Dirichlet noise
    ///     drop_last_n: Drop last N moves from each game
    ///
    /// Returns:
    ///     List of all training examples from all games
    #[pyo3(signature = (num_games, max_moves=100, add_noise=true, drop_last_n=10))]
    pub fn generate_games(
        &self,
        num_games: u32,
        max_moves: u32,
        add_noise: bool,
        drop_last_n: u32,
    ) -> Vec<TrainingExample> {
        let mut all_examples = Vec::new();

        for _ in 0..num_games {
            if let Some(result) = self.play_game(max_moves, add_noise, drop_last_n) {
                all_examples.extend(result.examples);
            }
        }

        all_examples
    }

}

// Internal implementation (not exposed to Python)
impl MCTSAgent {
    fn search_internal(
        &self,
        env: &TetrisEnv,
        policy: Vec<f32>,
        root_value: f32,
        add_noise: bool,
        move_number: u32,
    ) -> MCTSResult {
        // Create root node
        let mut root = DecisionNode::new(env.clone(), move_number);
        root.set_priors(&policy);

        if add_noise {
            root.add_dirichlet_noise(self.config.dirichlet_alpha, self.config.dirichlet_epsilon);
        }

        // Run simulations
        for _ in 0..self.config.num_simulations {
            self.simulate(&mut root, root_value, move_number);
        }

        // Build result policy from visit counts
        let mut result_policy = vec![0.0; NUM_ACTIONS];
        let total_visits: u32 = root.children.values().map(|c| c.visit_count()).sum();

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

            // Normalize
            let sum: f32 = result_policy.iter().sum();
            if sum > 0.0 {
                for p in &mut result_policy {
                    *p /= sum;
                }
            }
        } else {
            // Use prior policy if no visits
            result_policy = policy;
        }

        // Select action
        let action = if self.config.temperature == 0.0 {
            // Argmax
            result_policy.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).map(|(i, _)| i).unwrap_or(0)
        } else {
            // Sample from policy
            sample_action(&result_policy)
        };

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
    }

    /// Run a single MCTS simulation
    ///
    /// Args:
    ///     root: The root decision node
    ///     root_value: NN value estimate for root (used if no deeper evaluation available)
    ///     root_move_number: Move number at the root
    fn simulate(&self, root: &mut DecisionNode, root_value: f32, root_move_number: u32) {
        // Selection: traverse tree
        // Store (node_ptr, action_idx, attack_at_this_step)
        let mut path: Vec<(*mut DecisionNode, usize, f32)> = Vec::new();
        let mut current = root as *mut DecisionNode;
        let mut depth: u32 = 0;

        loop {
            let node = unsafe { &mut *current };

            if node.is_terminal {
                // Terminal - backpropagate with 0 future value (game over)
                self.backup_with_value(&path, 0.0);
                return;
            }

            if node.valid_actions.is_empty() {
                self.backup_with_value(&path, 0.0);
                return;
            }

            // Select action
            let action_idx = node.select_action(self.config.c_puct);

            // Check if child exists
            if !node.children.contains_key(&action_idx) {
                // Expansion: create new child
                let child = self.expand_action(node, action_idx);
                node.children.insert(action_idx, child);

                // Get attack and leaf state from the new node
                let (leaf_attack, leaf_value) = match node.children.get(&action_idx) {
                    Some(MCTSNode::Chance(chance_node)) => {
                        let attack = chance_node.attack as f32;
                        // Evaluate leaf with NN if available
                        let value = self.evaluate_leaf(&chance_node.state, root_move_number + depth + 1);
                        (attack, value)
                    }
                    _ => (0.0, 0.0),
                };

                // Add this step to path with its attack
                path.push((current, action_idx, leaf_attack));

                // Backpropagate: total = attack_along_path + leaf_value
                self.backup_with_value(&path, leaf_value);
                return;
            }

            // Traverse to child - get attack at this step
            let step_attack = match node.children.get(&action_idx) {
                Some(MCTSNode::Chance(chance_node)) => chance_node.attack as f32,
                _ => 0.0,
            };
            path.push((current, action_idx, step_attack));
            depth += 1;

            match node.children.get_mut(&action_idx) {
                Some(MCTSNode::Chance(chance_node)) => {
                    // Round-robin piece selection (randomized order, balanced exploration)
                    let piece = chance_node.select_piece_round_robin();

                    // Get or create decision node for this piece
                    if !chance_node.children.contains_key(&piece) {
                        let decision_child = self.expand_chance(chance_node, piece, root_move_number + depth);
                        chance_node.children.insert(piece, decision_child);
                    }

                    match chance_node.children.get_mut(&piece) {
                        Some(MCTSNode::Decision(decision_node)) => {
                            current = decision_node as *mut DecisionNode;
                        }
                        _ => break,
                    }
                }
                _ => break,
            }
        }

        // Terminal or no valid actions - backpropagate with root value as fallback
        self.backup_with_value(&path, root_value);
    }

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
            new_state.execute_placement(placement)
        } else {
            // Fallback to direct placement
            new_state.place_piece(x, y, rot)
        };

        // Compute remaining bag (simplified - just use all pieces)
        // In a real implementation, we'd track the 7-bag state
        let bag_remaining: Vec<usize> = (0..NUM_PIECE_TYPES).collect();

        MCTSNode::Chance(ChanceNode::new(new_state, attack, bag_remaining))
    }

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

    /// Backpropagate total game value through the path
    ///
    /// All nodes in the path receive the SAME value:
    ///   total = cumulative_attack_along_path + leaf_value
    ///
    /// This ensures consistent value estimates regardless of depth - a node at depth 5
    /// and depth 50 will have comparable Q values if the positions are equally good.
    ///
    /// Args:
    ///     path: List of (node_ptr, action_idx, attack_at_step)
    ///     leaf_value: NN value estimate of remaining game from the leaf state
    fn backup_with_value(&self, path: &[(*mut DecisionNode, usize, f32)], leaf_value: f32) {
        if path.is_empty() {
            return;
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
    }
}

/// Sample an action from a policy distribution
pub fn sample_action(policy: &[f32]) -> usize {
    let mut rng = thread_rng();
    let r: f32 = rng.gen();
    let mut cumsum = 0.0;

    for (i, &p) in policy.iter().enumerate() {
        cumsum += p;
        if r <= cumsum {
            return i;
        }
    }

    // Fallback to last action
    policy.len() - 1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_action_space() {
        let action_space = ActionSpace::new();
        assert_eq!(action_space.num_actions(), 734);

        // Test roundtrip
        for (idx, &(x, y, rot)) in action_space.action_to_placement.iter().enumerate() {
            let idx2 = action_space.placement_to_index(x, y, rot).unwrap();
            assert_eq!(idx, idx2);
        }
    }

    #[test]
    fn test_dirichlet() {
        let noise = sample_dirichlet(0.15, 10);
        assert_eq!(noise.len(), 10);

        let sum: f32 = noise.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_mcts_basic() {
        let config = MCTSConfig {
            num_simulations: 10,
            ..Default::default()
        };
        let agent = MCTSAgent::new(config);
        let env = TetrisEnv::new(10, 20);

        // Uniform policy
        let policy = vec![1.0 / NUM_ACTIONS as f32; NUM_ACTIONS];
        let result = agent.search(&env, policy, 0.0, false);

        assert!(result.action < NUM_ACTIONS);
        assert!((result.policy.iter().sum::<f32>() - 1.0).abs() < 1e-5);
    }
}
