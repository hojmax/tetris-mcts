# Configuration Centralization Changes

## Summary

All configurable parameters have been moved to `tetris_mcts/config.py` for centralized management. This eliminates scattered hardcoded values and makes experimentation easier.

## New Configuration Parameters

### Architecture Parameters (NEW)
- `conv_kernel_size` (default: 3) - Convolutional kernel size
- `conv_padding` (default: 1) - Convolutional padding
- `max_moves` (default: 100) - Maximum moves per game

**Note:** The network uses a separate `MAX_MOVES = 100` constant for move number normalization (part of the architecture spec). The config `max_moves` controls actual game length.

### Optimizer Parameters (NEW)
- `grad_clip_norm` (default: 1.0) - Gradient clipping threshold
- `lr_min_factor` (default: 0.01) - Minimum LR as fraction of initial (cosine schedule)
- `lr_step_gamma` (default: 0.1) - LR decay multiplier (step schedule)
- `lr_step_divisor` (default: 3) - Steps between decays (step schedule)

### MCTS Parameters (NEW)
- `c_puct` (default: 1.5) - PUCT exploration constant

## Modified Files

### `tetris_mcts/config.py`
- Added 9 new configuration parameters
- Updated default architecture: `[2, 4]` filters, `64` hidden units

### `tetris_mcts/ml/network.py`
- Added `conv_kernel_size` and `conv_padding` parameters to `TetrisNet.__init__()`
- Updated default values to match new config

### `tetris_mcts/ml/training.py`
- Pass `conv_kernel_size` and `conv_padding` to `TetrisNet`
- Use `config.grad_clip_norm` instead of hardcoded `1.0`
- Use `config.lr_min_factor`, `config.lr_step_gamma`, `config.lr_step_divisor` in schedulers
- Pass `config.c_puct` to `MCTSConfig`
- Use `config.max_moves` instead of `MAX_MOVES` for game generation
- Pass `max_moves` to `Evaluator`

### `tetris_mcts/ml/evaluation.py`
- Added `max_moves` parameter to `Evaluator.__init__()`
- Use `self.max_moves` instead of importing `MAX_MOVES`

### Documentation
- Updated `CLAUDE.md` with new architecture defaults
- Created `PARAMETERS.md` with comprehensive parameter reference
- Created `CHANGELOG_CONFIG.md` (this file)

### New Scripts
- `tetris_mcts/scripts/compare_architectures.py` - Benchmark different architectures
- `tetris_mcts/scripts/verify_config.py` - Verify config completeness
- `tetris_mcts/scripts/analyze_board_reuse.py` - Analyze board state reuse (for future caching experiments)

## Architecture Change

**Default architecture changed from:**
- Conv: 1→4→8 (307K parameters)
- FC: 1652→128
- Time: 0.40ms/batch

**To:**
- Conv: 1→2→4 (103K parameters)
- FC: 852→64
- Time: 0.30ms/batch

**Benefits:**
- **3x fewer parameters** (307K → 103K)
- **33% faster inference** (0.40ms → 0.30ms per batch)
- Smaller model size for deployment
- Faster training iterations

## Usage Examples

### Use defaults (new small architecture)
```bash
python tetris_mcts/scripts/train.py --training.total-steps 100000
```

### Revert to original large architecture
```bash
python tetris_mcts/scripts/train.py \
    --training.conv-filters 4 8 \
    --training.fc-hidden 128
```

### Experiment with different conv kernels
```bash
python tetris_mcts/scripts/train.py \
    --training.conv-kernel-size 5 \
    --training.conv-padding 2
```

### Tune MCTS exploration
```bash
python tetris_mcts/scripts/train.py \
    --training.c-puct 2.0 \
    --training.num-simulations 800
```

### Adjust gradient clipping
```bash
python tetris_mcts/scripts/train.py \
    --training.grad-clip-norm 0.5
```

### Modify learning rate schedule
```bash
python tetris_mcts/scripts/train.py \
    --training.lr-schedule step \
    --training.lr-step-gamma 0.5 \
    --training.lr-min-factor 0.001
```

## Verification

Run verification script to ensure all parameters are properly wired:
```bash
python tetris_mcts/scripts/verify_config.py
```

## Backwards Compatibility

**Breaking change:** Models trained with old architecture (`[4, 8]`, `128`) cannot load into networks with new architecture (`[2, 4]`, `64`) due to different parameter counts.

**Solution:** Specify architecture explicitly when loading old checkpoints:
```bash
python tetris_mcts/scripts/train.py \
    --resume-dir training_runs/v0 \
    --training.conv-filters 4 8 \
    --training.fc-hidden 128
```

## Future Work

Possible optimizations to explore:
1. **Batched inference** - Accumulate multiple MCTS leaf expansions for batch processing
2. **Model quantization** - FP16/INT8 for 1.5-2x speedup
3. **GPU inference** - Switch from tract-onnx to ONNX Runtime with CUDA
4. **Board state caching** - Cache CNN features (low hit rate expected, but worth profiling)
5. **Progressive widening** - Limit expansions per node to reduce NN calls

## Testing

All changes tested with:
- Architecture comparison benchmark (✓)
- Config verification script (✓)
- Network parameter verification (✓)
- Default values validation (✓)
