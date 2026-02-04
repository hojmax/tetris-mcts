# Configuration Parameters

All training and architecture parameters are now centralized in `tetris_mcts/config.py`.

## Network Architecture

| Parameter | Default | Description |
|-----------|---------|-------------|
| `conv_filters` | `[2, 4]` | Number of filters in each conv layer |
| `fc_hidden` | `64` | Hidden units in fully connected layer |
| `conv_kernel_size` | `3` | Kernel size for conv layers (3x3) |
| `conv_padding` | `1` | Padding for conv layers (maintains spatial dims) |
| `max_moves` | `100` | Maximum moves per game (for game length, not normalization) |

**Note:** Move number normalization in the network uses `MAX_MOVES = 100` constant (in `network.py`) as part of the architecture spec. The `config.max_moves` controls actual game length during training/eval.

## Optimizer

| Parameter | Default | Description |
|-----------|---------|-------------|
| `batch_size` | `256` | Training batch size |
| `learning_rate` | `0.001` | Initial learning rate |
| `weight_decay` | `1e-4` | L2 regularization weight |
| `grad_clip_norm` | `1.0` | Gradient clipping threshold |

## Learning Rate Schedule

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lr_schedule` | `"cosine"` | Schedule type: `"cosine"`, `"step"`, or `"none"` |
| `lr_decay_steps` | `10000` | Steps for full cosine cycle or step decay |
| `lr_min_factor` | `0.01` | Min LR as fraction of initial (cosine only) |
| `lr_step_gamma` | `0.1` | LR decay multiplier (step only) |
| `lr_step_divisor` | `3` | Decay every `lr_decay_steps // divisor` (step only) |

## MCTS / Self-Play

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_simulations` | `400` | MCTS simulations per move |
| `c_puct` | `1.5` | PUCT exploration constant |
| `temperature` | `1.0` | Action selection temperature |
| `dirichlet_alpha` | `0.15` | Dirichlet noise concentration |
| `dirichlet_epsilon` | `0.25` | Dirichlet noise weight |
| `num_workers` | `5` | Parallel game generation threads |

## Training Loop

| Parameter | Default | Description |
|-----------|---------|-------------|
| `total_steps` | `100000` | Total training steps |
| `model_sync_interval` | `2000` | Steps between ONNX exports |
| `checkpoint_interval` | `1000` | Steps between checkpoints |
| `eval_interval` | `200000` | Steps between evaluations |
| `log_interval` | `100` | Steps between logging |

## Replay Buffer

| Parameter | Default | Description |
|-----------|---------|-------------|
| `buffer_size` | `100000` | Maximum examples in memory |
| `min_buffer_size` | `100` | Minimum examples before training |
| `games_per_save` | `2000` | Games between disk saves (0=disable) |

## Usage Examples

### Use default small architecture
```bash
python tetris_mcts/scripts/train.py --training.total-steps 100000
```

### Use original large architecture
```bash
python tetris_mcts/scripts/train.py \
    --training.conv-filters 4 8 \
    --training.fc-hidden 128 \
    --training.total-steps 100000
```

### Faster training with more MCTS simulations
```bash
python tetris_mcts/scripts/train.py \
    --training.num-simulations 800 \
    --training.num-workers 10 \
    --training.batch-size 512
```

### Tune learning rate schedule
```bash
python tetris_mcts/scripts/train.py \
    --training.learning-rate 0.0005 \
    --training.lr-schedule step \
    --training.lr-decay-steps 5000 \
    --training.lr-step-gamma 0.5
```

### Experiment with different architectures
```bash
# Larger kernels (5x5)
python tetris_mcts/scripts/train.py \
    --training.conv-kernel-size 5 \
    --training.conv-padding 2

# Tiny model for fast experimentation
python tetris_mcts/scripts/train.py \
    --training.conv-filters 2 2 \
    --training.fc-hidden 32
```

## Parameter Recommendations

### Fast Experimentation
- `conv_filters=[2, 2]`, `fc_hidden=32` (fastest)
- `num_simulations=100` (reduce search time)
- `batch_size=128` (faster updates)

### Production Training
- `conv_filters=[2, 4]`, `fc_hidden=64` (default, good speed/quality)
- `num_simulations=400-800` (better policy targets)
- `batch_size=256-512` (stable gradients)

### Maximum Quality
- `conv_filters=[4, 8]`, `fc_hidden=128` (original, slower)
- `num_simulations=1600` (very accurate MCTS)
- `batch_size=512` (large batches)
