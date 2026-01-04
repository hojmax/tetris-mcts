# Tetris AlphaZero Training Plan

A comprehensive guide to implementing AlphaZero-style training for Tetris, with MCTS in Rust and neural network training in Python.

## Overview

We adapt [AlphaZero](alphazero.pdf) (Silver et al., 2017) for single-player stochastic Tetris:

- **Self-play**: Rust MCTS generates training data
- **Neural Network**: PyTorch CNN with policy and value heads
- **Inference**: Tiny network exported to run in Rust (CPU)
- **Key difference**: Tetris has stochastic piece spawns, so we use expectimax-style MCTS

---

## 1. Neural Network Architecture

### Input Representation

| Component      | Shape      | Encoding                        |
| -------------- | ---------- | ------------------------------- |
| Board state    | 20 x 10    | Binary (1 = filled, 0 = empty)  |
| Current piece  | 7          | One-hot encoded                 |
| Hold piece     | 8          | One-hot (7 pieces + empty)      |
| Hold available | 1          | Binary (can use hold this turn) |
| Next queue     | 5 x 7 = 35 | One-hot encoded per slot        |
| Move number    | 1          | Normalized: move_idx / 100      |

**Total input**: 200 + 7 + 8 + 1 + 35 + 1 = **252 features**

Move number lets the value head learn that later positions have less remaining future attack (due to 100-move episode cap).

### Network Structure

```
Input Board (20x10x1)
    │
    ├──► Conv2D(1, 4, kernel=3x3, padding=1) + BatchNorm2d + ReLU
    │        │
    │        ▼
    │    Conv2D(4, 8, kernel=3x3, padding=1) + BatchNorm2d + ReLU
    │        │
    │        ▼
    │    Flatten → 20*10*8 = 1,600
    │
Auxiliary Input (52 features: current + hold + hold_avail + next_queue + move_num)
    │
    └──► Concat with flattened board features
              │
              ▼
         FC(1652, 128) + LayerNorm + ReLU
              │
              ├──────────────────┐
              ▼                  ▼
         Policy Head        Value Head
         FC(128, 734)       FC(128, 1)
         Softmax            (linear)
              │                  │
              ▼                  ▼
         π(a|s)              V(s) = predicted attack
```

- **BatchNorm2d** after conv layers: stabilizes training, normalizes across batch
- **LayerNorm** after FC: ensures both heads receive well-scaled features

### Output Space

- **Policy head**: 734 outputs (all valid piece placements)
  - Each output corresponds to a unique (x, y, rotation) position
  - Invalid moves are masked before softmax
- **Value head**: 1 output (linear, no activation)
  - Predicts cumulative attack from current state to end of game
  - MSE loss against actual cumulative attack values

### Move Indexing

Create a lookup table mapping action index (0-733) to (x, y, rotation):

```python
# Generate all 734 valid positions (from count_reachable_states.py)
ACTION_TO_PLACEMENT = []  # List of (x, y, rotation) tuples
PLACEMENT_TO_ACTION = {}  # Dict mapping (x, y, rot) -> index

# Iterate through all positions, check if valid for ANY piece
for rot in range(4):
    for y in range(-3, 20):
        for x in range(-3, 10):
            if any_piece_fits(x, y, rot):
                idx = len(ACTION_TO_PLACEMENT)
                ACTION_TO_PLACEMENT.append((x, y, rot))
                PLACEMENT_TO_ACTION[(x, y, rot)] = idx
```

---

## 2. Move Masking

Not all 734 actions are valid for every game state. We must mask invalid moves.

### Masking Logic

For each state, generate a **mask tensor** of shape (734,):

```python
def get_action_mask(board, current_piece):
    """Returns binary mask: 1 = valid, 0 = invalid"""
    mask = torch.zeros(734)

    for action_idx, (x, y, rot) in enumerate(ACTION_TO_PLACEMENT):
        if piece_can_be_placed(board, current_piece, x, y, rot):
            mask[action_idx] = 1.0

    return mask

def apply_mask_to_policy(logits, mask):
    """Apply mask before softmax"""
    # Set invalid actions to -inf so softmax gives 0 probability
    masked_logits = logits.clone()
    masked_logits[mask == 0] = float('-inf')
    return F.softmax(masked_logits, dim=-1)
```

### Rust Integration

The Rust MCTS must:

1. Generate valid moves for current state (already implemented in `moves.rs`)
2. Map each valid move to its action index
3. Only expand MCTS nodes for valid actions

---

## 3. MCTS for Single-Player Stochastic Games

### Key Differences from AlphaZero

1. **No opponent**: Tree only has our moves + random piece spawns
2. **Stochastic transitions**: After placing a piece, next piece is random (7-bag)
3. **Expectimax structure**: Alternate between decision nodes and chance nodes

### Tree Structure

```
Decision Node (our move)
    │
    ├── Action a1 ──► Chance Node (piece spawn)
    │                     ├── piece I (prob 1/7) ──► Decision Node
    │                     ├── piece O (prob 1/7) ──► Decision Node
    │                     └── ... (7 outcomes)
    │
    ├── Action a2 ──► Chance Node
    └── ...
```

### MCTS Algorithm

```rust
struct MCTSNode {
    state: GameState,
    visit_count: u32,
    total_value: f32,
    children: Vec<(Action, MCTSNode)>,  // For decision nodes
    // OR
    chance_children: Vec<(Piece, f32, MCTSNode)>,  // For chance nodes (piece, prob, child)
    prior: f32,  // From policy network
    is_chance_node: bool,
}

fn mcts_search(root: &mut MCTSNode, num_simulations: u32, network: &Network) {
    for _ in 0..num_simulations {
        // Selection: traverse tree using UCB
        let (leaf, path) = select(root);

        // Expansion: expand leaf node
        if !leaf.is_terminal() {
            expand(leaf, network);
        }

        // Evaluation: get value from network (no rollout)
        let value = if leaf.is_terminal() {
            terminal_value(leaf)
        } else {
            network.evaluate(leaf.state).value
        };

        // Backup: propagate value up the tree
        backup(path, value);
    }
}

fn select(node: &MCTSNode) -> &MCTSNode {
    if node.is_chance_node {
        // Round-robin over pieces in randomized order (ensures balanced exploration)
        select_next_piece_round_robin(node)
    } else {
        // Use PUCT formula for action selection
        let best_action = argmax(|a| ucb_score(node, a));
        &node.children[best_action]
    }
}

fn ucb_score(node: &MCTSNode, action: Action) -> f32 {
    let child = &node.children[action];
    let q = child.total_value / child.visit_count.max(1) as f32;
    let u = C_PUCT * child.prior * (node.visit_count as f32).sqrt()
            / (1.0 + child.visit_count as f32);
    q + u
}
```

### Handling 7-Bag Randomization

The 7-bag system is NOT uniform random. Must track bag boundaries during lookahead.

**Key insight**: The queue shows 5 pieces, but we need to know where the current bag ends and the next begins to compute correct probabilities.

```rust
struct BagState {
    remaining: Vec<Piece>,  // Pieces left in current bag
}

impl BagState {
    fn piece_probabilities(&self) -> Vec<(Piece, f32)> {
        // Only pieces remaining in current bag have non-zero probability
        let count = self.remaining.len() as f32;
        self.remaining.iter()
            .map(|&p| (p, 1.0 / count))
            .collect()
    }

    fn consume_piece(&mut self, piece: Piece) {
        // Remove piece from current bag
        self.remaining.retain(|&p| p != piece);

        // If bag is empty, start new bag with all 7 pieces
        if self.remaining.is_empty() {
            self.remaining = vec![I, O, T, S, Z, J, L];
        }
    }
}
```

**Example lookahead**:

- Current bag: 5/7 consumed, remaining = [T, L]
- Queue shows: [T, L, I, O, S, ...]
- Pieces 0-1 (T, L) are deterministic from current bag
- Piece 2 (I) starts new bag: was uniform 1/7 when drawn
- Beyond visible queue: use bag state to compute probabilities

**During MCTS simulation**:

```rust
fn simulate_chance_node(state: &GameState) -> Vec<(Piece, f32, GameState)> {
    let bag = &state.bag_state;

    // Get probabilities from current bag state
    let probs = bag.piece_probabilities();

    probs.iter().map(|&(piece, prob)| {
        let mut next_state = state.clone();
        next_state.bag_state.consume_piece(piece);
        next_state.current_piece = next_state.queue.pop_front();
        next_state.queue.push_back(piece);  // New piece enters queue
        (piece, prob, next_state)
    }).collect()
}
```

**Chance node piece selection**: Rather than sampling by probability each simulation, the implementation uses **round-robin selection** over pieces in a randomized order. Each chance node shuffles the possible pieces once, then cycles through them. This ensures balanced exploration of all piece outcomes while still respecting the 7-bag distribution on average.

---

## 4. Self-Play Data Generation

### Data Collection Loop (Rust)

```rust
fn self_play_game(network: &Network, config: &Config) -> Vec<TrainingExample> {
    let mut examples = Vec::new();
    let mut game = GameState::new();

    while !game.is_game_over() {
        // Run MCTS from current position
        let mut root = MCTSNode::new(game.clone());

        // Add Dirichlet noise to root node priors (training only, not eval)
        add_dirichlet_noise_to_root(&mut root, config.dirichlet_alpha, config.noise_epsilon);

        mcts_search(&mut root, config.num_simulations, network);

        // Get improved policy from MCTS visit counts
        let pi = get_mcts_policy(&root, config.temperature);

        // Store training example (state, policy, placeholder value)
        examples.push(TrainingExample {
            state: game.encode(),
            policy: pi,
            value: 0.0,  // Fill in after game ends
        });

        // Select action (sample from pi during training, argmax during eval)
        let action = sample_action(&pi, config.temperature);
        game.apply_action(action);
    }

    // Fill in values: cumulative attack from each position to end
    for (i, example) in examples.iter_mut().enumerate() {
        example.value = compute_cumulative_attack(&game.attack_history, i);
    }

    examples
}

/// Add Dirichlet noise to root node priors for exploration during training
fn add_dirichlet_noise_to_root(root: &mut MCTSNode, alpha: f32, epsilon: f32) {
    let valid_actions: Vec<usize> = root.get_valid_action_indices();
    let n = valid_actions.len();

    // Sample from Dirichlet(alpha, alpha, ..., alpha)
    let noise = sample_dirichlet(alpha, n);

    // Mix network prior with noise: p' = (1 - epsilon) * p + epsilon * noise
    for (i, &action_idx) in valid_actions.iter().enumerate() {
        let prior = root.children[action_idx].prior;
        root.children[action_idx].prior = (1.0 - epsilon) * prior + epsilon * noise[i];
    }
}

fn sample_dirichlet(alpha: f32, n: usize) -> Vec<f32> {
    // Sample n values from Gamma(alpha, 1), then normalize
    let mut rng = thread_rng();
    let gamma = Gamma::new(alpha, 1.0).unwrap();

    let samples: Vec<f32> = (0..n).map(|_| gamma.sample(&mut rng) as f32).collect();
    let sum: f32 = samples.iter().sum();

    samples.into_iter().map(|x| x / sum).collect()
}

fn get_mcts_policy(root: &MCTSNode, temperature: f32) -> Vec<f32> {
    let mut policy = vec![0.0; 734];
    let total_visits: u32 = root.children.iter().map(|c| c.visit_count).sum();

    for (action, child) in &root.children {
        let action_idx = action.to_index();
        if temperature == 0.0 {
            // Deterministic: all mass on most visited
            policy[action_idx] = if child.visit_count == max_visits { 1.0 } else { 0.0 };
        } else {
            // Proportional to visit count ^ (1/temp)
            policy[action_idx] = (child.visit_count as f32).powf(1.0 / temperature);
        }
    }

    // Normalize
    let sum: f32 = policy.iter().sum();
    policy.iter_mut().for_each(|p| *p /= sum);
    policy
}
```

### Value Target Computation

The reward is **attack** (lines sent to opponent). This is what we maximize.

Attack values per action:

- Single: 0 attack
- Double: 1 attack
- Triple: 2 attack
- Tetris: 4 attack
- T-spin single: 2 attack
- T-spin double: 4 attack
- T-spin triple: 6 attack
- Back-to-back bonus: +1 attack
- Combo bonus: scales with combo count

```rust
const MAX_MOVES: usize = 100;

fn compute_training_examples(game: &Game) -> Vec<TrainingExample> {
    let mut examples = Vec::new();
    let num_states = game.states.len();

    // Compute cumulative attack from each position to end of game
    let mut cumulative_attack = vec![0.0f32; num_states];
    let mut running_total = 0u32;

    for i in (0..num_states).rev() {
        running_total += game.attacks[i];
        cumulative_attack[i] = running_total as f32;
    }

    // All moves are used for training
    for move_idx in 0..num_states {
        examples.push(TrainingExample {
            state: game.states[move_idx].clone(),
            policy: game.mcts_policies[move_idx].clone(),
            value: cumulative_attack[move_idx],
        });
    }

    examples
}
```

**Value targets use raw cumulative attack** (no discounting). The value for move N is the sum of all attack from move N to the end of the game. This provides a direct learning signal for "how much attack will I get from here?"

### Data Format (Save to Disk)

```python
# training_data.npz
{
    'boards': np.array of shape (N, 20, 10), dtype=bool,
    'current_pieces': np.array of shape (N, 7), dtype=float32,  # one-hot
    'hold_pieces': np.array of shape (N, 8), dtype=float32,     # one-hot + empty
    'hold_available': np.array of shape (N,), dtype=bool,
    'next_queue': np.array of shape (N, 5, 7), dtype=float32,   # one-hot
    'move_numbers': np.array of shape (N,), dtype=float32,      # normalized by 100
    'policy_targets': np.array of shape (N, 734), dtype=float32,
    'value_targets': np.array of shape (N,), dtype=float32,
    'action_masks': np.array of shape (N, 734), dtype=bool,
}
```

---

## 5. Training Loop (Python)

### Loss Function

```python
def compute_loss(model, batch):
    boards, aux_features, policy_targets, value_targets, masks = batch

    # Forward pass
    policy_logits, value_pred = model(boards, aux_features)

    # Apply mask and softmax
    masked_policy = apply_mask_to_policy(policy_logits, masks)

    # Policy loss: cross-entropy with MCTS policy
    policy_loss = -torch.sum(policy_targets * torch.log(masked_policy + 1e-8), dim=1).mean()

    # Value loss: MSE
    value_loss = F.mse_loss(value_pred.squeeze(), value_targets)

    # Total loss (AlphaZero uses equal weighting)
    total_loss = policy_loss + value_loss

    return total_loss, policy_loss, value_loss
```

### Training Configuration

```python
config = {
    # Network
    'conv_filters': [4, 8],
    'fc_hidden': 128,

    # Training
    'batch_size': 256,
    'learning_rate': 0.001,
    'weight_decay': 1e-4,
    'lr_schedule': 'cosine',

    # Self-play
    'num_simulations': 100,      # MCTS simulations per move
    'temperature': 1.0,          # For first 15 moves
    'temperature_drop': 0.1,     # After move 15
    'num_games_per_iteration': 100,

    # Replay buffer
    'buffer_size': 100000,
    'min_buffer_size': 10000,    # Start training after this many examples

    # Iteration
    'num_iterations': 100,
    'training_steps_per_iter': 1000,
    'checkpoint_interval': 10,
}
```

### Parallel Training Architecture

Self-play and training run in parallel:

- **Self-play process** (Rust): Continuously generates games, fills replay buffer
- **Training process** (Python): Continuously trains on buffer
- **Weight sync**: Every N training steps, export new weights for self-play

```
┌─────────────────┐         ┌─────────────────┐
│   Self-Play     │         │    Training     │
│    (Rust)       │         │    (Python)     │
│                 │         │                 │
│  Load weights ◄─┼─────────┼─ Export weights │
│       │         │         │       ▲         │
│       ▼         │         │       │         │
│  Play games     │         │  Train on batch │
│       │         │         │       ▲         │
│       ▼         │  shared │       │         │
│  Write to ──────┼─────────┼► Read from      │
│  buffer         │  buffer │  buffer         │
└─────────────────┘         └─────────────────┘
```

```python
# Shared replay buffer (e.g., memory-mapped file or Redis)
WEIGHT_SYNC_INTERVAL = 1000  # Sync weights every N training steps

def training_process():
    model = TetrisNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    replay_buffer = SharedReplayBuffer(config['buffer_size'])

    # Export initial weights
    export_weights(model, 'weights/latest.bin')

    step = 0
    while True:
        # Wait for minimum buffer size
        if replay_buffer.size() < config['min_buffer_size']:
            time.sleep(1)
            continue

        # Training step
        batch = replay_buffer.sample(config['batch_size'])
        optimizer.zero_grad()
        loss, p_loss, v_loss = compute_loss(model, batch)
        loss.backward()
        optimizer.step()
        step += 1

        # Logging
        wandb.log({
            'loss': loss.item(),
            'policy_loss': p_loss.item(),
            'value_loss': v_loss.item(),
            'step': step,
            'buffer_size': replay_buffer.size(),
        })

        # Sync weights to self-play
        if step % WEIGHT_SYNC_INTERVAL == 0:
            export_weights(model, 'weights/latest.bin')
            wandb.log({'weight_version': step // WEIGHT_SYNC_INTERVAL})

        # Evaluation
        if step % config['eval_interval'] == 0:
            metrics = evaluate_model(model)
            wandb.log(metrics)
            torch.save(model.state_dict(), f'checkpoints/model_{step}.pt')

def selfplay_process():
    """Rust process that continuously generates games"""
    # Runs: cargo run --release -- --weights weights/latest.bin --buffer shared_buffer
    # Automatically reloads weights when file changes
    pass
```

```rust
// Rust self-play worker
fn selfplay_loop(buffer_path: &str, weights_path: &str) {
    let mut model = load_weights(weights_path);
    let mut last_weights_modified = get_modified_time(weights_path);

    loop {
        // Check for new weights
        let current_modified = get_modified_time(weights_path);
        if current_modified > last_weights_modified {
            model = load_weights(weights_path);
            last_weights_modified = current_modified;
            println!("Loaded new weights");
        }

        // Play one game
        let examples = play_game(&model, MAX_MOVES);

        // Append to shared buffer
        append_to_buffer(buffer_path, &examples);
    }
}
```

---

## 6. Rust-Python Integration

### Option A: Export to ONNX, Run in Rust

```python
# Python: Export model
def export_model_for_rust(model, path):
    model.eval()
    dummy_board = torch.zeros(1, 1, 20, 10)
    dummy_aux = torch.zeros(1, 52)  # 7 + 8 + 1 + 35 + 1 = 52
    torch.onnx.export(model, (dummy_board, dummy_aux), path)
```

```rust
// Rust: Load and run ONNX
use ort::{Environment, SessionBuilder, Value};

fn load_model(path: &str) -> Session {
    let env = Environment::builder().build().unwrap();
    SessionBuilder::new(&env).with_model_from_file(path).unwrap()
}

fn evaluate(session: &Session, state: &GameState) -> (Vec<f32>, f32) {
    let board_tensor = state.board_to_tensor();
    let aux_tensor = state.aux_to_tensor();

    let outputs = session.run(vec![
        Value::from_array(board_tensor),
        Value::from_array(aux_tensor),
    ]).unwrap();

    let policy: Vec<f32> = outputs[0].try_extract().unwrap();
    let value: f32 = outputs[1].try_extract().unwrap()[0];

    (policy, value)
}
```

### Option B: Manual Weight Loading (Faster for Tiny Networks)

```rust
// Rust: Implement forward pass manually
struct TetrisNet {
    conv1_weights: Array4<f32>,
    conv1_bias: Array1<f32>,
    conv2_weights: Array4<f32>,
    conv2_bias: Array1<f32>,
    fc1_weights: Array2<f32>,
    fc1_bias: Array1<f32>,
    policy_weights: Array2<f32>,
    policy_bias: Array1<f32>,
    value_weights: Array2<f32>,
    value_bias: Array1<f32>,
}

impl TetrisNet {
    fn forward(&self, board: &Array2<f32>, aux: &Array1<f32>) -> (Array1<f32>, f32) {
        // Conv layers
        let x = conv2d(&board, &self.conv1_weights, &self.conv1_bias);
        let x = relu(&x);
        let x = conv2d(&x, &self.conv2_weights, &self.conv2_bias);
        let x = relu(&x);

        // Flatten and concat
        let x = x.into_shape(1600).unwrap();  // 20*10*8 = 1600
        let x = concatenate![Axis(0), x, aux];

        // FC layer
        let x = self.fc1_weights.dot(&x) + &self.fc1_bias;
        let x = relu(&x);

        // Heads
        let policy_logits = self.policy_weights.dot(&x) + &self.policy_bias;
        let value = (self.value_weights.dot(&x) + &self.value_bias)[0].tanh();

        (policy_logits, value)
    }
}
```

### Benchmark Both Options

For tiny networks (< 100K params), manual Rust implementation may be faster than ONNX runtime overhead, especially on CPU.

---

## 7. Exploration Strategies

### Dirichlet Noise at Root

Add noise to prior probabilities at root node for exploration:

```rust
fn add_dirichlet_noise(priors: &mut [f32], alpha: f32, epsilon: f32) {
    let noise = sample_dirichlet(alpha, priors.len());
    for (p, n) in priors.iter_mut().zip(noise.iter()) {
        *p = (1.0 - epsilon) * *p + epsilon * n;
    }
}

// AlphaZero uses alpha=0.3 for chess, epsilon=0.25
// For Tetris with 734 actions, try alpha=0.15, epsilon=0.25
```

---

## 8. Implementation Phases

### Phase 1: Infrastructure

- [ ] Create action index mapping (734 positions)
- [ ] Implement move masking in Rust
- [ ] Set up data serialization format
- [ ] PyTorch network skeleton
- [ ] Weight export/import pipeline

### Phase 2: Basic Training

- [ ] Random policy self-play (no MCTS)
- [ ] Train network on random games
- [ ] Verify training loop works
- [ ] WandB integration

### Phase 3: MCTS Integration

- [ ] Implement MCTS with chance nodes
- [ ] Handle 7-bag probabilities correctly
- [ ] PUCT selection with network priors
- [ ] Generate training data from MCTS games

### Phase 4: Full AlphaZero Loop

- [ ] Iterative self-play + training
- [ ] Replay buffer management
- [ ] Learning rate scheduling
- [ ] Checkpoint management

### Phase 5: Optimization

- [ ] Benchmark ONNX vs manual Rust inference
- [ ] Tune hyperparameters (MCTS sims, temperature, etc.)
- [ ] Profile and optimize bottlenecks

---

## 9. Metrics to Track (WandB)

### Training Metrics

- `loss`, `policy_loss`, `value_loss`
- `policy_entropy` (should decrease over training)
- `value_prediction_error`
- `learning_rate`
- `gradient_norm`

### Self-Play Metrics

- `avg_game_length` (pieces placed per game)
- `avg_attack` (total attack per game)
- `avg_attack_per_piece` (efficiency)
- `games_per_second`

### Evaluation Metrics (Fixed Seeds)

**Important**: Use the same set of seeds (e.g., seeds 0-19) for evaluation throughout training. Cap each game at 100 moves for consistent comparison.

```rust
const EVAL_SEEDS: [u64; 20] = [0, 1, 2, ..., 19];
const EVAL_MAX_MOVES: u32 = 100;

fn evaluate(model: &Model) -> EvalMetrics {
    let mut metrics = EvalMetrics::default();
    for seed in EVAL_SEEDS {
        let game = play_game_with_seed(model, seed, EVAL_MAX_MOVES, /*no exploration*/);
        metrics.accumulate(&game);
    }
    metrics.average()
}
```

**Core metrics** (over 100 moves per seed):

- `eval/avg_attack` - total attack averaged over eval games
- `eval/max_attack` - best single game
- `eval/avg_lines` - total lines cleared
- `eval/attack_per_piece` - attack / 100 moves

**Line clear breakdown** (count per game, averaged):

- `eval/clears_0` - pieces placed with no line clear
- `eval/clears_1` - singles
- `eval/clears_2` - doubles
- `eval/clears_3` - triples
- `eval/clears_4` - tetrises

**T-spin breakdown**:

- `eval/tspin_mini` - T-spin minis
- `eval/tspin_single` - T-spin singles
- `eval/tspin_double` - T-spin doubles
- `eval/tspin_triple` - T-spin triples

**Combo & back-to-back**:

- `eval/max_combo` - longest combo achieved
- `eval/avg_combo_length` - average combo length when combo > 0
- `eval/back_to_back_count` - number of back-to-back clears
- `eval/back_to_back_attack` - total attack from B2B bonuses

---

## 10. Hyperparameter Recommendations

| Parameter           | Initial Value | Notes                                   |
| ------------------- | ------------- | --------------------------------------- |
| MCTS simulations    | 100           | Increase for stronger play              |
| c_puct              | 1.5           | Exploration constant                    |
| Temperature         | 1.0           | Sampling from visit counts (0 = argmax) |
| Dirichlet alpha     | 0.15          | Lower than chess due to more actions    |
| Dirichlet epsilon   | 0.25          | Standard AlphaZero value                |
| Batch size          | 256           |                                         |
| Learning rate       | 0.001         | With Adam                               |
| Replay buffer       | 100K          | Examples, not games                     |
| Training steps/iter | 1000          |                                         |
| Games per iteration | 100           |                                         |

---

## Appendix: Key Differences from Original AlphaZero

1. **Single-player**: No adversarial opponent, value predicts attack not win/loss
2. **Stochastic**: Must handle random piece spawns with chance nodes
3. **Action space**: 734 valid placements (vs 4672 for chess)
4. **Tiny network**: Optimized for CPU inference in Rust
5. **7-bag randomizer**: Not uniform distribution, must track bag state
6. **No game symmetry**: Unlike chess/Go, limited board symmetries to exploit
7. **Fixed episode length**: All games capped at 100 moves
