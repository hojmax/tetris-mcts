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

| Component         | Shape      | Encoding                                  |
| ----------------- | ---------- | ----------------------------------------- |
| Board state       | 20 × 10    | Binary (1 = filled, 0 = empty)            |
| Current piece     | 7          | One-hot encoded                           |
| Hold piece        | 8          | One-hot (7 pieces + empty)                |
| Hold available    | 1          | Binary (can use hold this turn)           |
| Next queue        | 5 × 7 = 35 | One-hot encoded per slot                  |
| Placement count   | 1          | Normalized: count / max_placements        |
| Combo             | 1          | Normalized: combo / 12 (uncapped)         |
| Back-to-back      | 1          | Binary (1 = active)                       |
| Hidden piece dist | 7          | Probability distribution from 7-bag state |
| Column heights    | 10         | Normalized: height / 20 per column        |
| Max column height | 1          | Normalized: max height / 20               |
| Min column height | 1          | Normalized: min height / 20               |
| Row fill counts   | 20         | Normalized: fill / 10 per row             |
| Total blocks      | 1          | Normalized: count / 200                   |
| Bumpiness         | 1          | Normalized: Σ(Δh²) / 3600                 |
| Holes             | 1          | Normalized: sealed cavities / 190         |
| Overhang fields   | 1          | Normalized: empty-below-filled / 190      |
| **Total**         | **297**    | **(200 board + 97 auxiliary)**            |

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
Auxiliary Input (97 features)
    │
    └──► Concat with flattened board features
              │
              ▼
         FC(1652, 128) + LayerNorm + ReLU
              │
              ├──────────────────┐
              ▼                  ▼
         Policy Head        Value Head
         FC(128, 735)       FC(128, 1)
         Softmax            (linear)
              │                  │
              ▼                  ▼
         π(a|s)              V(s) = predicted attack
```

- **BatchNorm2d** after conv layers: stabilizes training, normalizes across batch
- **LayerNorm** after FC: ensures both heads receive well-scaled features

### Output Space

- **Policy head**: 735 outputs (734 valid piece placements + hold)
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

Not all 735 actions are valid for every game state. We must mask invalid moves.

### Masking Logic

For each state, generate a **mask tensor** of shape (735,):

```python
def get_action_mask(board, current_piece):
    """Returns binary mask: 1 = valid, 0 = invalid"""
    mask = torch.zeros(735)

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

        // Get improved policy target from MCTS visit counts.
        // Temperature shapes the training target distribution only.
        let pi = get_mcts_policy(&root, config.temperature);

        // Store training example (state, policy, placeholder value)
        examples.push(TrainingExample {
            state: game.encode(),
            policy: pi,
            value: 0.0,  // Fill in after game ends
        });

        // Select action to execute: always best move (argmax / most visited)
        let action = argmax_action(&pi);
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
    let mut policy = vec![0.0; 735];
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
const MAX_PLACEMENTS: usize = 100;

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
    'policy_targets': np.array of shape (N, 735), dtype=float32,
    'value_targets': np.array of shape (N,), dtype=float32,
    'action_masks': np.array of shape (N, 735), dtype=bool,
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

### Parallel Training Architecture

The implementation uses **integrated parallel game generation** via Rust `GameGenerator`:

- **GameGenerator** (Rust): Spawns worker threads that continuously generate self-play games using MCTS
- **In-memory buffer**: Training examples stored in shared memory (no disk I/O during training)
- **Direct sampling**: Python training loop samples batches directly from buffer via PyO3
- **Weight sync**: Every N training steps, export ONNX model; GameGenerator hot-swaps it

```
┌────────────────────────────────────────┐
│       Python Training Process          │
│                                        │
│  ┌──────────────────────────────────┐  │
│  │   Trainer.train()                │  │
│  │                                  │  │
│  │  1. generator.sample_batch(...)  │  │ ◄─── Direct sampling via PyO3
│  │  2. Train on batch               │  │
│  │  3. Export ONNX (periodic)       │  │
│  │  4. Log metrics                  │  │
│  └──────────────────────────────────┘  │
│            │                           │
│            │ PyO3 binding              │
└────────────┼───────────────────────────┘
             │
             ▼
┌────────────────────────────────────────┐
│    Rust GameGenerator (cdylib)         │
│                                        │
│  ┌─────────┐  ┌─────────┐              │
│  │ Worker  │  │ Worker  │  ...         │ ◄─── num_workers threads
│  │ Thread  │  │ Thread  │              │
│  └────┬────┘  └────┬────┘              │
│       │            │                   │
│       │  Play MCTS games               │
│       │  using current NN              │
│       ▼            ▼                   │
│  ┌──────────────────────────────────┐  │
│  │   Shared In-Memory Buffer        │  │ ◄─── RwLock<Vec<TrainingExample>>
│  │   (RingBuffer, max size)         │  │
│  └──────────────────────────────────┘  │
│       ▲                                │
│       │                                │
│       └── Hot-swap ONNX model          │ ◄─── When Python exports new weights
│           (atomic reload)              │
└────────────────────────────────────────┘
```

**Key benefits of this architecture:**

- No separate processes or IPC overhead
- No disk I/O during training (only periodic saves for resume)
- Simple weight synchronization (just export ONNX when needed)
- Game generation continues uninterrupted during training steps

```python
# Python training loop (simplified)
from tetris_core import GameGenerator, MCTSConfig
from tetris.config import TrainingConfig
from tetris.ml.network import TetrisNet

def train():
    config = TrainingConfig()
    model = TetrisNet(
        conv_filters=config.conv_filters,
        fc_hidden=config.fc_hidden,
        conv_kernel_size=config.conv_kernel_size,
        conv_padding=config.conv_padding,
    )
    mcts_config = MCTSConfig(num_simulations=400, ...)
    mcts_config.max_placements = config.max_placements

    # Create GameGenerator with worker threads
    generator = GameGenerator(
        model_path="parallel.onnx",
        training_data_path="training_runs/v0/training_data.npz",
        config=mcts_config,
        max_placements=config.max_placements,
        num_workers=5,
        max_examples=100_000,
    )

    for step in range(total_steps):
        # Sample directly from Rust buffer (no disk I/O)
        batch = generator.sample_batch(batch_size=256, max_placements=config.max_placements)
        if batch is None:
            continue

        # Train
        optimizer.zero_grad()
        loss = compute_loss(model, batch)
        loss.backward()
        optimizer.step()

        # Export ONNX periodically
        if step % model_sync_interval == 0:
            export_onnx(model, "parallel.onnx")
            # No explicit reload call needed: GameGenerator watches model_path
            # and hot-swaps atomically when the file timestamp changes.

        # Log metrics
        if step % log_interval == 0:
            stats = generator.get_stats()
            wandb.log({
                'loss': loss.item(),
                'games_generated': stats['games_generated'],
                'examples_generated': stats['examples_generated'],
                'buffer_size': stats['buffer_size'],
            })
```

**Rust GameGenerator implementation** (simplified):

- Worker threads continuously generate games via MCTS + current neural network
- Each completed game produces training examples that are added to shared buffer
- Buffer is a ring buffer (FIFO): old examples replaced when buffer is full
- Neural network reloaded atomically when Python exports new ONNX model
- Aggregate statistics (line clears, T-spins, combos) tracked with atomic counters

---

## 6. Rust-Python Integration

### ONNX Export and Inference

The implementation uses **tract-onnx** for fast CPU inference in Rust:

**Python: Export ONNX Model** (`weights.py`)

```python
def export_onnx(model: TetrisNet, path: Path) -> None:
    """Export PyTorch model to ONNX for Rust inference."""
    model.eval()
    dummy_board = torch.zeros(1, 1, 20, 10)
    dummy_aux = torch.zeros(1, 97)  # 97 auxiliary features

    torch.onnx.export(
        model,
        (dummy_board, dummy_aux),
        path,
        input_names=["board", "aux"],
        output_names=["policy_logits", "value"],
        dynamic_axes={
            "board": {0: "batch_size"},
            "aux": {0: "batch_size"},
        },
    )
```

**Rust: Load and Run ONNX** (`nn.rs`)

```rust
use tract_onnx::prelude::*;

pub struct TetrisNN {
    model: Arc<TypedRunnableModel<TypedModel>>,
}

impl TetrisNN {
    /// Load ONNX model from file
    pub fn load<P: AsRef<Path>>(path: P) -> TractResult<Self> {
        let model = tract_onnx::onnx()
            .model_for_path(path)?
            .into_optimized()?
            .into_runnable()?;

        Ok(TetrisNN {
            model: Arc::new(model),
        })
    }

    /// Run inference with action mask applied
    pub fn predict_masked(
        &self,
        env: &TetrisEnv,
        move_number: usize,
        action_mask: &[bool],
    ) -> TractResult<(Vec<f32>, f32)> {
        // Encode state to tensors
        let (board_vec, aux_vec) = encode_state(env, move_number);

        // Create tract tensors
        let board = tract_ndarray::Array4::from_shape_vec(
            (1, 1, BOARD_HEIGHT, BOARD_WIDTH),
            board_vec,
        )?.into_tensor();

        let aux = tract_ndarray::Array2::from_shape_vec(
            (1, AUX_FEATURES),
            aux_vec,
        )?.into_tensor();

        // Run inference
        let result = self.model.run(tvec!(board, aux))?;

        // Extract outputs
        let policy_logits = result[0].to_array_view::<f32>()?;
        let value = result[1].to_array_view::<f32>()?[[0]];

        // Apply softmax with masking
        let policy = apply_mask_and_softmax(policy_logits.as_slice().unwrap(), action_mask);

        Ok((policy, value))
    }
}
```

**Why tract-onnx?**

- Fast CPU inference with SIMD optimizations
- Small dependency footprint (no Python runtime needed)
- Supports all operations used by TetrisNet (Conv2d, BatchNorm, LayerNorm, Linear)
- Model optimization at load time
- Thread-safe (can share across workers)

---

## 6.5. Training Directory Structure

Training runs are organized with automatic versioning:

```
training_runs/
├── v0/                    # First training run
│   ├── config.json        # Saved hyperparameters
│   ├── training_data.npz  # Periodic backup of training data
│   └── checkpoints/
│       ├── step_1000.pt
│       ├── step_2000.pt
│       └── ...
├── v1/                    # Second training run
│   └── ...
└── v2/                    # And so on...
```

**Key features:**

- Automatic version incrementing (v0, v1, v2, ...)
- Each run gets isolated directory with checkpoints and config
- Resume capability: `--resume-dir training_runs/v0`
- Config saved as JSON for reproducibility
- NPZ backup files for recovery (training primarily uses in-memory buffer)

**Usage:**

```bash
# New run (auto-creates training_runs/v0)
python scripts/train.py --total-steps 100000

# Resume from checkpoint
python scripts/train.py --resume-dir training_runs/v0
```

---

## 6.6. Available Tools & Scripts

### Training

- **`train.py`** - Main training script with WandB logging
- **`evaluate.py`** - Evaluate trained model on fixed seeds

### Visualization

- **`tetris_game.py`** - Interactive Pygame Tetris (manual play or MCTS agent)
- **`scripts/inspection/mcts_visualizer.py`** - Dash app for MCTS tree visualization (port 8050)
- **`scripts/inspection/buffer_viewer.py`** - Inspect GameGenerator's in-memory training buffer

### Data Analysis

- **`scripts/inspection/inspect_training_data.py`** - View contents of training_data.npz files
- **`scripts/inspection/analyze_training_data.py`** - Compute statistics over training data
- **`scripts/inspection/count_reachable_states.py`** - Enumerate all 734 valid piece placements
- **`scripts/inspection/profile_games.py`** - Performance profiling of game generation

### Ablations & Utilities

- **`scripts/ablations/compare_offline_architectures.py`** - Baseline vs gated offline architecture benchmark
- **`scripts/ablations/compare_offline_feature_ablation.py`** - Offline sweep over state-feature ablations
- **`scripts/ablations/compare_offline_network_scaling.py`** - Default vs scaled trunk/post-fusion offline benchmark

**Quick access via Makefile:**

```bash
make play      # Launch interactive Tetris
make viz       # Launch MCTS tree visualizer
make test      # Run Rust tests
make build     # Rebuild Rust extension
```

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
// For Tetris with 735 actions (734 placements + hold), try alpha=0.15, epsilon=0.25
```

---

## 9. Metrics to Track (WandB)

### Training Metrics

- `train/loss`, `train/policy_loss`, `train/value_loss`
- `train/learning_rate`, `train/grad_norm`
- `batch/value_target_mean`, `batch/value_target_std`
- `batch/overhang_fields_mean`, `batch/valid_actions_mean`

### Self-Play Metrics

- `replay/buffer_size`, `replay/games_generated`, `replay/examples_generated`
- `throughput/games_per_second`, `throughput/steps_per_second`
- `incumbent/model_step`, `incumbent/uses_network`
- `incumbent/lifetime_games`, `incumbent/lifetime_avg_attack`
- `model_gate/*` (candidate vs incumbent promotion decisions)
- `game/*` (per-completed-game stats, including `game/attack_per_move`)

### Evaluation Metrics (Fixed Seeds)

**Important**: Use the same set of seeds (e.g., seeds 0-19) for evaluation throughout training. Cap each game at 100 moves for consistent comparison.

```rust
const EVAL_SEEDS: [u64; 20] = [0, 1, 2, ..., 19];
const EVAL_MAX_PLACEMENTS: u32 = 100;

fn evaluate(model: &Model) -> EvalMetrics {
    let mut metrics = EvalMetrics::default();
    for seed in EVAL_SEEDS {
        let game = play_game_with_seed(model, seed, EVAL_MAX_PLACEMENTS, /*no exploration*/);
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

## Appendix: Key Differences from Original AlphaZero

1. **Single-player**: No adversarial opponent, value predicts attack not win/loss
2. **Stochastic**: Must handle random piece spawns with chance nodes
3. **Action space**: 735 actions total (734 valid placements + hold; vs 4672 for chess)
4. **Tiny network**: Optimized for CPU inference in Rust
5. **7-bag randomizer**: Not uniform distribution, must track bag state
6. **No game symmetry**: Unlike chess/Go, limited board symmetries to exploit
7. **Fixed episode length**: All games capped at 100 moves
