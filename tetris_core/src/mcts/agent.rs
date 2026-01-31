//! MCTS Agent
//!
//! The main agent that runs MCTS search and self-play.

use pyo3::prelude::*;

use crate::constants::{BOARD_HEIGHT, BOARD_WIDTH, QUEUE_SIZE};
use crate::env::TetrisEnv;

use super::action_space::{get_action_space, NUM_ACTIONS};
use super::config::MCTSConfig;
use super::nodes::{ChanceNode, DecisionNode, MCTSNode};
use super::results::{GameResult, MCTSResult, MCTSTreeExport, TrainingExample, TreeNodeExport};
use super::utils::sample_action;

/// MCTS Agent for Tetris
#[pyclass]
pub struct MCTSAgent {
    config: MCTSConfig,
    /// Optional neural network for leaf evaluation (pure Rust mode)
    nn: Option<crate::nn::TetrisNN>,
}

#[pymethods]
impl MCTSAgent {
    #[new]
    pub fn new(config: MCTSConfig) -> Self {
        MCTSAgent {
            config,
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
    ///
    /// Returns:
    ///     GameResult with training examples, or None if no model loaded
    #[pyo3(signature = (max_moves=100, add_noise=true))]
    pub fn play_game(
        &self,
        max_moves: u32,
        add_noise: bool,
    ) -> Option<GameResult> {
        let nn = self.nn.as_ref()?;

        let mut env = TetrisEnv::new(BOARD_WIDTH, BOARD_HEIGHT);
        let mut states: Vec<(TetrisEnv, u32, Vec<f32>, Vec<bool>)> = Vec::new();
        let mut attacks: Vec<u32> = Vec::new();

        for move_idx in 0..max_moves {
            if env.game_over {
                break;
            }

            // Get action mask
            let mask = crate::nn::get_action_mask(&env);
            if !mask.iter().any(|&x| x) {
                // This should only happen if game is over - if not, it's a bug
                debug_assert!(env.game_over, "No valid actions but game not over - this is a bug");
                break;
            }

            // Get NN policy and value for root
            let (policy, nn_value) = nn.predict_masked(&env, move_idx as usize, &mask)
                .expect("Neural network prediction failed during self-play");

            // Store state before making move
            states.push((env.clone(), move_idx, policy.clone(), mask.clone()));

            // Run MCTS search
            let result = self.search(&env, policy, nn_value, add_noise, move_idx);

            // Execute the selected action
            let (x, y, rot) = get_action_space().index_to_placement(result.action)
                .expect("MCTS returned invalid action index");
            let placements = env.get_possible_placements();
            let placement = placements.iter().find(|p| {
                p.piece.x == x && p.piece.y == y && p.piece.rotation == rot
            }).expect("MCTS selected action not found in valid placements");
            let attack = env.execute_placement(placement);
            attacks.push(attack);

            // Update stored policy with MCTS policy
            if let Some(last) = states.last_mut() {
                last.2 = result.policy;
            }
        }

        // Compute value targets (cumulative attack from each position)
        let num_states = states.len();
        debug_assert_eq!(states.len(), attacks.len(), "States and attacks should have same length");

        let mut values = vec![0.0f32; num_states];
        let mut cumulative = 0u32;
        for i in (0..num_states).rev() {
            cumulative += attacks[i];
            values[i] = cumulative as f32;
        }

        // Build training examples (use all moves)
        let mut examples = Vec::with_capacity(num_states);

        for i in 0..num_states {
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
    ///
    /// Returns:
    ///     List of all training examples from all games
    #[pyo3(signature = (num_games, max_moves=100, add_noise=true))]
    pub fn generate_games(
        &self,
        num_games: u32,
        max_moves: u32,
        add_noise: bool,
    ) -> Vec<TrainingExample> {
        let mut all_examples = Vec::new();

        for _ in 0..num_games {
            if let Some(result) = self.play_game(max_moves, add_noise) {
                all_examples.extend(result.examples);
            }
        }

        all_examples
    }

    /// Run MCTS search and return both the result and the tree structure.
    ///
    /// This is for visualization/debugging - it exports the entire MCTS tree.
    ///
    /// Args:
    ///     env: The game state to search from
    ///     add_noise: Whether to add Dirichlet noise to root priors
    ///     move_number: The current move number in the game
    ///
    /// Returns:
    ///     Tuple of (MCTSResult, MCTSTreeExport) or None if no model loaded
    #[pyo3(signature = (env, add_noise=false, move_number=0))]
    pub fn search_with_tree(
        &self,
        env: &TetrisEnv,
        add_noise: bool,
        move_number: u32,
    ) -> Option<(MCTSResult, MCTSTreeExport)> {
        let nn = self.nn.as_ref()?;

        // Get action mask and initial policy
        let mask = crate::nn::get_action_mask(env);
        if !mask.iter().any(|&x| x) {
            return None;
        }

        let (policy, nn_value) = nn.predict_masked(env, move_number as usize, &mask)
            .expect("Neural network prediction failed");

        let (mcts_result, root) = self.search_internal(env, policy, nn_value, add_noise, move_number);

        // Export tree structure
        let mut nodes: Vec<TreeNodeExport> = Vec::new();
        self.export_decision_node(&root, None, None, &mut nodes);

        let tree_export = MCTSTreeExport {
            nodes,
            root_id: 0,
            num_simulations: self.config.num_simulations,
            selected_action: mcts_result.action,
            policy: mcts_result.policy.clone(),
        };

        Some((mcts_result, tree_export))
    }

    /// Run MCTS search and return the result (simpler than search_with_tree).
    ///
    /// This is the main method for selecting actions during play/evaluation.
    /// Handles NN inference internally.
    ///
    /// Args:
    ///     env: The game state to search from
    ///     add_noise: Whether to add Dirichlet noise to root priors
    ///     move_number: The current move number in the game
    ///
    /// Returns:
    ///     MCTSResult with selected action and policy, or None if no model loaded
    #[pyo3(signature = (env, add_noise=false, move_number=0))]
    pub fn select_action(
        &self,
        env: &TetrisEnv,
        add_noise: bool,
        move_number: u32,
    ) -> Option<MCTSResult> {
        let nn = self.nn.as_ref()?;

        // Get action mask and initial policy
        let mask = crate::nn::get_action_mask(env);
        if !mask.iter().any(|&x| x) {
            return None;
        }

        let (policy, nn_value) = nn.predict_masked(env, move_number as usize, &mask)
            .expect("Neural network prediction failed");

        let (mcts_result, _root) = self.search_internal(env, policy, nn_value, add_noise, move_number);
        Some(mcts_result)
    }
}

impl MCTSAgent {
    /// Get a reference to the neural network (if loaded).
    pub fn get_nn(&self) -> Option<&crate::nn::TetrisNN> {
        self.nn.as_ref()
    }

    /// Run MCTS search from a given state.
    pub fn search(
        &self,
        env: &TetrisEnv,
        policy: Vec<f32>,
        nn_value: f32,
        add_noise: bool,
        move_number: u32,
    ) -> MCTSResult {
        let (result, _root) = self.search_internal(env, policy, nn_value, add_noise, move_number);
        result
    }

    /// Run a single MCTS simulation
    ///
    /// Uses raw pointers for tree traversal to track the path from root to leaf.
    /// This is a common pattern in tree structures where we need mutable access
    /// to nodes at multiple levels simultaneously.
    ///
    /// # Safety
    /// The unsafe pointer operations are sound because:
    /// 1. All pointers are derived from valid mutable references to tree nodes
    /// 2. The tree structure is not modified during traversal (no reallocation)
    /// 3. Each node is accessed through exactly one pointer at a time
    /// 4. Pointers remain valid for the entire duration of a single simulation
    ///
    /// # Args
    /// * `root` - The root decision node
    /// * `root_move_number` - Move number at the root
    fn simulate(&self, root: &mut DecisionNode, root_move_number: u32) {
        // Selection: traverse tree, tracking path for backpropagation
        // Store (node_ptr, action_idx, attack_at_this_step)
        let mut path: Vec<(*mut DecisionNode, usize, f32)> = Vec::new();
        let mut current = root as *mut DecisionNode;
        let mut depth: u32 = 0;

        loop {
            // SAFETY: `current` is always derived from a valid &mut DecisionNode.
            // The tree structure doesn't change during simulation, so the pointer
            // remains valid. We only hold one mutable reference at a time.
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
                // Expansion: create new child (NN evaluation happens inside expand_action)
                let child = self.expand_action(node, action_idx, root_move_number + depth + 1);
                node.children.insert(action_idx, child);

                // Get attack and nn_value from the new node
                let chance_node = match node.children.get(&action_idx) {
                    Some(MCTSNode::Chance(cn)) => cn,
                    _ => panic!("BUG: expand_action should create ChanceNode"),
                };
                let leaf_attack = chance_node.attack as f32;
                let leaf_value = chance_node.nn_value;  // Use stored NN value

                // Add this step to path with its attack
                path.push((current, action_idx, leaf_attack));

                // Backpropagate: total = attack_along_path + leaf_value
                self.backup_with_value(&path, leaf_value);
                return;
            }

            // Traverse to child - get attack at this step
            let chance_node = match node.children.get_mut(&action_idx) {
                Some(MCTSNode::Chance(cn)) => cn,
                _ => panic!("BUG: Decision node child should be ChanceNode"),
            };
            let step_attack = chance_node.attack as f32;
            path.push((current, action_idx, step_attack));
            depth += 1;

            // Randomly select which piece outcome to explore
            let piece = chance_node.select_piece_random();

            // Get or create decision node for this piece
            if !chance_node.children.contains_key(&piece) {
                let decision_child = self.expand_chance(chance_node, piece, root_move_number + depth);
                chance_node.children.insert(piece, decision_child);
            }

            match chance_node.children.get_mut(&piece) {
                Some(MCTSNode::Decision(decision_node)) => {
                    current = decision_node as *mut DecisionNode;
                }
                _ => panic!("BUG: ChanceNode child should be DecisionNode"),
            }
        }
    }

    /// Evaluate a leaf state with the neural network
    fn evaluate_leaf(&self, env: &TetrisEnv, move_number: u32) -> f32 {
        let nn = self.nn.as_ref()
            .expect("Neural network required for MCTS leaf evaluation");
        let mask = crate::nn::get_action_mask(env);
        let (_, value) = nn.predict_masked(env, move_number as usize, &mask)
            .expect("Neural network prediction failed during leaf evaluation");
        value
    }

    /// Expand an action from a decision node (creates chance node)
    fn expand_action(&self, parent: &DecisionNode, action_idx: usize, move_number: u32) -> MCTSNode {
        let mut new_state = parent.state.clone();

        // Get placement coordinates from action index
        let (x, y, rot) = get_action_space().index_to_placement(action_idx)
            .expect("Invalid action index in expand_action");

        // Find the matching placement to get move sequence for T-spin detection
        let placements = new_state.get_possible_placements();
        let placement = placements.iter().find(|p| {
            p.piece.x == x && p.piece.y == y && p.piece.rotation == rot
        }).expect("Action not found in valid placements during expansion");
        let attack = new_state.execute_placement(placement);

        // Truncate to visible queue length FIRST.
        // This ensures expand_chance pushes to position 5 (the first "unseen" position).
        new_state.truncate_queue(QUEUE_SIZE);

        // Compute possible pieces: intersection of 7-bag constraints and visual constraints.
        // This ensures no duplicates in the visible window while respecting bag rules.
        let bag_remaining = new_state.get_possible_next_pieces_for_mcts();

        // Evaluate the NN on this state to get the value estimate
        let nn_value = self.evaluate_leaf(&new_state, move_number);

        MCTSNode::Chance(ChanceNode::new(new_state, attack, bag_remaining, nn_value))
    }

    /// Expand a chance node for a specific piece (creates decision node)
    ///
    /// The "piece" parameter represents the piece that appears at the END of the visible
    /// queue (the next unseen piece). This is the actual "chance" in Tetris - we know
    /// the current piece and visible queue, but not what comes after.
    fn expand_chance(&self, parent: &ChanceNode, piece: usize, move_number: u32) -> MCTSNode {
        let mut new_state = parent.state.clone();

        // Add the selected piece to the end of the queue
        // This represents the "chance" outcome - which piece appears next in the queue
        new_state.push_queue_piece(piece);

        let mut node = DecisionNode::new(new_state.clone(), move_number);

        // Set priors and value from neural network
        let nn = self.nn.as_ref()
            .expect("Neural network required for MCTS chance node expansion");
        let mask = crate::nn::get_action_mask(&new_state);
        let (policy, value) = nn.predict_masked(&new_state, move_number as usize, &mask)
            .expect("Neural network prediction failed during chance node expansion");
        node.set_nn_output(&policy, value);

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
    /// ## No double counting
    /// The path contains only DecisionNode pointers. For each entry (node, action, _):
    /// - We update `node.children[action]` (a ChanceNode)
    /// - We update `node` itself (a DecisionNode)
    ///
    /// These are DIFFERENT objects. The next entry's "node" is a child of the previous
    /// ChanceNode, not the ChanceNode itself:
    /// ```text
    /// root (DecisionNode) ──action_A──> ChanceNode_A ──piece_X──> node_B (DecisionNode)
    /// path[0] = (root, action_A)                                  path[1] = (node_B, action_B)
    /// ```
    /// Each node is updated exactly once.
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

        // Update each DecisionNode and its child ChanceNode (no double counting - see doc above)
        for &(node_ptr, action_idx, _) in path.iter() {
            // SAFETY: node_ptr was stored during the simulation traversal from valid
            // &mut DecisionNode references. The tree hasn't been modified, so pointers
            // remain valid. Each pointer in the path refers to a distinct node.
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

    /// Internal search implementation that returns both result and root node.
    fn search_internal(
        &self,
        env: &TetrisEnv,
        policy: Vec<f32>,
        nn_value: f32,
        add_noise: bool,
        move_number: u32,
    ) -> (MCTSResult, DecisionNode) {
        // Create root node (keep full queue - truncation breaks 7-bag tracking)
        let mut root = DecisionNode::new(env.clone(), move_number);
        root.set_nn_output(&policy, nn_value);

        if add_noise {
            root.add_dirichlet_noise(self.config.dirichlet_alpha, self.config.dirichlet_epsilon);
        }

        // Run simulations
        for _ in 0..self.config.num_simulations {
            self.simulate(&mut root, move_number);
        }

        // Build result policy from visit counts
        let mut result_policy = vec![0.0; NUM_ACTIONS];
        let total_visits: u32 = root.children.values().map(|c| c.visit_count()).sum();

        debug_assert!(total_visits > 0, "MCTS should have visits after simulations");

        let action = if self.config.temperature == 0.0 {
            let (best_action, _) = root.children.iter()
                .max_by_key(|(_, child)| child.visit_count())
                .map(|(&idx, child)| (idx, child.visit_count()))
                .expect("MCTS root should have children after simulations");
            result_policy[best_action] = 1.0;
            best_action
        } else {
            for (&action_idx, child) in &root.children {
                result_policy[action_idx] = (child.visit_count() as f32).powf(1.0 / self.config.temperature);
            }
            let sum: f32 = result_policy.iter().sum();
            if sum > 0.0 {
                for p in &mut result_policy {
                    *p /= sum;
                }
            }
            sample_action(&result_policy)
        };

        let root_value = if root.visit_count > 0 {
            root.value_sum / root.visit_count as f32
        } else {
            0.0
        };

        let mcts_result = MCTSResult {
            policy: result_policy,
            action,
            value: root_value,
            num_simulations: self.config.num_simulations,
        };

        (mcts_result, root)
    }

    /// Recursively export a decision node and its subtree
    fn export_decision_node(
        &self,
        node: &DecisionNode,
        parent_id: Option<usize>,
        edge_from_parent: Option<usize>,
        nodes: &mut Vec<TreeNodeExport>,
    ) -> usize {

        let id = nodes.len();
        let mean_value = if node.visit_count > 0 {
            node.value_sum / node.visit_count as f32
        } else {
            0.0
        };

        // Create the node (children will be filled in later)
        let export = TreeNodeExport {
            id,
            node_type: "decision".to_string(),
            visit_count: node.visit_count,
            value_sum: node.value_sum,
            mean_value,
            nn_value: node.nn_value,
            prior: node.prior,
            is_terminal: node.is_terminal,
            move_number: node.move_number,
            attack: 0,
            state: node.state.clone(),
            parent_id,
            edge_from_parent,
            children: Vec::new(),
            valid_actions: node.valid_actions.clone(),
            action_priors: node.action_priors.clone(),
        };

        nodes.push(export);

        // Export children (sorted by action index for deterministic order)
        let mut child_keys: Vec<usize> = node.children.keys().copied().collect();
        child_keys.sort();

        let mut child_ids = Vec::new();
        for action_idx in child_keys {
            if let Some(MCTSNode::Chance(chance_node)) = node.children.get(&action_idx) {
                let child_id = self.export_chance_node(chance_node, Some(id), Some(action_idx), nodes);
                child_ids.push(child_id);
            }
        }

        // Update our children list
        nodes[id].children = child_ids;

        id
    }

    /// Recursively export a chance node and its subtree
    fn export_chance_node(
        &self,
        node: &ChanceNode,
        parent_id: Option<usize>,
        edge_from_parent: Option<usize>,
        nodes: &mut Vec<TreeNodeExport>,
    ) -> usize {

        let id = nodes.len();
        let mean_value = if node.visit_count > 0 {
            node.value_sum / node.visit_count as f32
        } else {
            0.0
        };

        let export = TreeNodeExport {
            id,
            node_type: "chance".to_string(),
            visit_count: node.visit_count,
            value_sum: node.value_sum,
            mean_value,
            nn_value: node.nn_value,
            prior: 0.0,
            is_terminal: false,
            move_number: 0,
            attack: node.attack,
            state: node.state.clone(),
            parent_id,
            edge_from_parent,
            children: Vec::new(),
            valid_actions: Vec::new(),
            action_priors: Vec::new(),
        };

        nodes.push(export);

        // Export children (sorted by piece type for deterministic order)
        let mut child_keys: Vec<usize> = node.children.keys().copied().collect();
        child_keys.sort();

        let mut child_ids = Vec::new();
        for piece_type in child_keys {
            if let Some(MCTSNode::Decision(decision_node)) = node.children.get(&piece_type) {
                let child_id = self.export_decision_node(decision_node, Some(id), Some(piece_type), nodes);
                child_ids.push(child_id);
            }
        }

        nodes[id].children = child_ids;

        id
    }
}
