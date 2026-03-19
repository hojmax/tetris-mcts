# Network Architecture

This document describes the current neural network architecture implemented in this repo, including the training-time model, the runtime split-export path, tensor shapes, feature layout, and the alternative baseline architecture.

The code is the source of truth. If this document and the code disagree, trust:

- `tetris_bot/ml/network.py`
- `tetris_bot/ml/config.py`
- `tetris_bot/ml/weights.py`
- `tetris_core/src/inference/mod.rs`
- `tetris_core/src/game/action_space.rs`
- `tetris_bot/ml/loss.py`

## Current Default at a Glance

The default network instantiated by `NetworkConfig()` is:

- Architecture: `gated_fusion`
- Board input: `1 x 20 x 10`
- Auxiliary input: `80` features
- Auxiliary split: `61` piece/game features + `19` board-stat features
- Outputs: `735` policy logits + `1` scalar value
- Default hyperparameters:
  - `trunk_channels=16`
  - `num_conv_residual_blocks=3`
  - `reduction_channels=32`
  - `fc_hidden=128`
  - `aux_hidden=64`
  - `num_fusion_blocks=1`
  - `conv_kernel_size=3`
  - `conv_padding=1`
- Total trainable parameters: `511,136`

The repo also includes a simpler baseline architecture, `simple_aux_mlp`, described near the end of this document.

## High-Level Design

The current default model is intentionally split into a board-dependent path and a piece/game-context path:

- The board occupancy tensor goes through a convolutional trunk.
- Board-derived scalar statistics are concatenated only after the conv trunk.
- Piece/game context never enters the conv trunk. It is processed separately and only interacts with the board representation in the fusion stage.
- At runtime, Rust caches the board embedding keyed only by board occupancy, so repeated searches on the same board do not re-run the expensive conv path.

That split is the main architectural idea in this repo.

## Input Contract

The model consumes two inputs:

- `board`: shape `(B, 1, 20, 10)`
- `aux_features`: shape `(B, 80)`

`board` is the raw binary occupancy grid:

- `1.0` for filled cells
- `0.0` for empty cells

`aux_features` is stored as one flat `80`-dimensional vector in training data, but the model splits it internally into:

- `piece_aux`: first `61` features
- `board_stats`: last `19` features

### Piece Order

Piece-type indices come from `tetris_core/src/game/constants.rs`:

- `0 = I`
- `1 = O`
- `2 = T`
- `3 = S`
- `4 = Z`
- `5 = J`
- `6 = L`

That ordering is used anywhere the network expects one-hot piece encodings or piece-distribution vectors.

### Exact Auxiliary Layout

| Full aux indices | Size | Name | Encoding / normalization |
| --- | ---: | --- | --- |
| `0..6` | 7 | Current piece | One-hot over the 7 tetromino types |
| `7..14` | 8 | Hold piece | One-hot over 7 tetromino types plus one extra `empty` slot |
| `15` | 1 | Hold available | Binary flag: `1.0` if hold may still be used this turn |
| `16..50` | 35 | Next queue | `5` queue slots, each one-hot over the 7 tetromino types |
| `51` | 1 | Placement count | `env.placement_count / max_placements` |
| `52` | 1 | Combo | `combo / 4.0` |
| `53` | 1 | Back-to-back | Binary flag |
| `54..60` | 7 | Hidden next-piece distribution | Uniform distribution over the currently possible hidden pieces implied by the 7-bag state |
| `61..70` | 10 | Column heights | Each column height divided by `8.0` |
| `71` | 1 | Max column height | Max height divided by `20.0` |
| `72..75` | 4 | Bottom row fill counts | The bottom `4` row-fill counts, each divided by board width `10` |
| `76` | 1 | Total blocks | Total filled cells divided by `60.0` |
| `77` | 1 | Bumpiness | Sum of squared adjacent-height deltas divided by `200.0` |
| `78` | 1 | Holes | Hole count divided by `20.0` |
| `79` | 1 | Overhang fields | Overhang-field count divided by `25.0` |

Important details:

- These are normalized ratios, not hard-clamped `[0, 1]` features. Some features can exceed `1.0` on extreme boards because the code just divides by a fixed normalization constant.
- `combo` is explicitly linear and uncapped.
- The hidden-piece distribution is computed after truncating the visible queue to `QUEUE_SIZE=5`, then enumerating which pieces are still possible from the 7-bag state.
- `board_stats` are board-only features by design. None of them depend on current piece, queue contents, hold availability, combo, or back-to-back.

### Board-Stats Definitions

The 19 board-stat features are computed in Rust and reused by the runtime caching path:

- `column_heights`: normalized current heights for each of the 10 columns
- `max_column_height`: normalized maximum column height
- `row_fill_counts`: normalized fill counts for the bottom 4 rows only
- `total_blocks`: normalized filled-cell count
- `bumpiness`: normalized `sum((h[i] - h[i+1])^2)` across adjacent columns
- `holes`: empty cells below a filled cell in the same column that are not reachable from top-row air via 4-neighbor flood fill
- `overhang_fields`: empty cells below a filled cell in the same column, whether reachable or not

The normalization divisors are empirical constants in `tetris_core/src/game/constants.rs`:

- Column heights: `8.0`
- Max column height: `20.0`
- Total blocks: `60.0`
- Bumpiness: `200.0`
- Holes: `20.0`
- Overhang fields: `25.0`

## Default Architecture: `gated_fusion`

### End-to-End Tensor Flow

```text
board (B,1,20,10)
  -> Conv2d(1,16,3,pad=1)
  -> GroupNorm(16 groups, 16 channels)
  -> SiLU
  -> 3 x ResidualConvBlock(16 channels)
  -> Conv2d(16,32,3,pad=1,stride=2)
  -> GroupNorm(32 groups, 32 channels)
  -> SiLU
  -> flatten
  -> concat(board_stats[19])
  -> Linear(1619,128)
  = board_h (B,128)

piece_aux (B,61)
  -> Linear(61,64)
  -> LayerNorm(64)
  -> SiLU
  = aux_h (B,64)

aux_h
  -> Linear(64,128) -> sigmoid = gate (B,128)
  -> Linear(64,128) = aux_bias (B,128)

fused_0 = board_h * (1 + gate) + aux_bias
fused_0 -> LayerNorm(128) -> SiLU
        -> 1 x ResidualFusionBlock(128)
        = fused (B,128)

fused -> Linear(128,256) -> SiLU -> Linear(256,735) = policy_logits
fused -> Linear(128,64)  -> SiLU -> Linear(64,1)    = value
```

### Stage-by-Stage Shapes

For batch size `B`, the default `gated_fusion` model has the following shapes:

| Stage | Operation | Output shape | Notes |
| --- | --- | --- | --- |
| 1 | Input board | `(B, 1, 20, 10)` | Binary occupancy tensor |
| 2 | `conv_initial` | `(B, 16, 20, 10)` | `Conv2d(1,16,3,padding=1)` |
| 3 | Initial norm/activation | `(B, 16, 20, 10)` | `GroupNorm -> SiLU` |
| 4 | `3 x ResidualConvBlock` | `(B, 16, 20, 10)` | Spatial resolution unchanged |
| 5 | `conv_reduce` | `(B, 32, 10, 5)` | `Conv2d(16,32,3,padding=1,stride=2)` |
| 6 | Reduction norm/activation | `(B, 32, 10, 5)` | `GroupNorm -> SiLU` |
| 7 | Flatten | `(B, 1600)` | `32 x 10 x 5 = 1600` |
| 8 | Concat board stats | `(B, 1619)` | `1600 + 19` |
| 9 | `board_proj` | `(B, 128)` | This is `board_h` |
| 10 | `aux_fc + aux_ln + SiLU` | `(B, 64)` | Piece/game-context path |
| 11 | `gate_fc -> sigmoid` | `(B, 128)` | Channelwise gate |
| 12 | `aux_proj` | `(B, 128)` | Channelwise additive term |
| 13 | Fusion | `(B, 128)` | `board_h * (1 + gate) + aux_proj(aux_h)` |
| 14 | `fusion_ln + SiLU` | `(B, 128)` | Prepares fused state for residual MLP |
| 15 | `1 x ResidualFusionBlock` | `(B, 128)` | Preserves width |
| 16 | `policy_fc + SiLU` | `(B, 256)` | Hidden width is `2 x fc_hidden` |
| 17 | `policy_head` | `(B, 735)` | Raw logits |
| 18 | `value_fc + SiLU` | `(B, 64)` | Hidden width is `fc_hidden / 2` |
| 19 | `value_head` | `(B, 1)` | Scalar value |

### Exact Connection Equations

Using the code’s actual variable names:

```text
x0 = SiLU(GN(conv_initial(board)))
x1 = ResidualConvBlock_1(x0)
x2 = ResidualConvBlock_2(x1)
x3 = ResidualConvBlock_3(x2)
x4 = SiLU(GN(conv_reduce(x3)))

flat = flatten(x4)
board_h = board_proj(concat(flat, board_stats))

aux_h = SiLU(LayerNorm(aux_fc(piece_aux)))
gate = sigmoid(gate_fc(aux_h))
fused = board_h * (1 + gate) + aux_proj(aux_h)
fused = SiLU(LayerNorm(fused))
fused = ResidualFusionBlocks(fused)

policy_logits = policy_head(SiLU(policy_fc(fused)))
value = value_head(SiLU(value_fc(fused)))
```

Important connection details:

- `board_stats` are injected exactly once, at `board_proj`.
- `piece_aux` never touches the conv trunk.
- The gate is multiplicative but only in the form `1 + sigmoid(...)`, so it scales each `board_h` channel by a factor in `(1, 2)` before the additive aux term is applied.
- The fusion stage is both multiplicative and additive:
  - multiplicative via `board_h * (1 + gate)`
  - additive via `aux_proj(aux_h)`

### Residual Block Definitions

The two residual block types are different.

#### `ResidualConvBlock`

Each conv residual block is:

```text
input
  -> GroupNorm
  -> SiLU
  -> Conv2d(channels, channels, kernel=3, padding=1)
  -> GroupNorm
  -> SiLU
  -> Conv2d(channels, channels, kernel=3, padding=1)
  + skip connection
```

This is a pre-activation-style residual block with no channel change and no spatial downsampling.

#### `ResidualFusionBlock`

Each fusion residual block is:

```text
input
  -> LayerNorm
  -> SiLU
  -> Linear(hidden, hidden)
  -> LayerNorm
  -> SiLU
  -> Linear(hidden, hidden)
  + skip connection
```

With `fc_hidden=128`, each fusion block adds `33,536` parameters.

### Normalization and Activation

The current architecture does not use BatchNorm or ReLU.

- Conv path: `GroupNorm + SiLU`
- Aux/fusion/head MLP path: `LayerNorm + SiLU`

`_make_group_norm()` chooses the largest divisor from `(32, 16, 8, 4, 2, 1)` that divides the channel count:

- For default `16`-channel tensors, it uses `GroupNorm(16, 16)`.
- For default `32`-channel tensors, it uses `GroupNorm(32, 32)`.

### Parameter Counts for Default `gated_fusion`

| Subsystem | Parameters |
| --- | ---: |
| Initial conv + initial GroupNorm | 192 |
| 3 conv residual blocks | 14,112 |
| Reduction conv + reduction GroupNorm | 4,704 |
| `board_proj` | 207,360 |
| Aux path (`aux_fc`, `aux_ln`, `gate_fc`, `aux_proj`) | 20,736 |
| Fusion LayerNorm | 256 |
| 1 fusion residual block | 33,536 |
| Policy head stack | 221,919 |
| Value head stack | 8,321 |
| **Total** | **511,136** |

Two modules dominate parameter count:

- `board_proj`, because it maps `1619 -> 128`
- The policy head, because `735` actions is a wide output layer

## Output Contract

The network returns:

- `policy_logits`: shape `(B, 735)`
- `value`: shape `(B, 1)`

### Action Space Mapping

The 735 policy outputs correspond to:

- `0..733`: placement actions
- `734`: hold action

Placement actions are generated in `tetris_core/src/game/action_space.rs` by enumerating all `(x, y, rotation)` tuples that are valid for at least one piece on an empty board, then sorting by:

- rotation
- y
- x

### Masking

The network itself returns raw logits. Invalid actions are masked outside the model:

- Training masks invalid logits to `-inf` before `log_softmax`
- Rust runtime computes softmax only over valid actions

This means the architecture always emits a fixed-width `735`-logit vector even though each concrete game state has only a subset of legal moves.

## Training-Time Losses

The architecture is trained with two heads and two losses:

- Policy loss: cross-entropy against the MCTS target distribution
- Value loss: mean-squared error against the replay `value_targets`

The training loss in `tetris_bot/ml/loss.py` is:

```text
total_loss = policy_loss + value_loss_weight * value_loss
```

`value_loss_weight` is not fixed. It is adapted online from rolling averages:

```text
value_loss_weight = avg(policy_loss) / avg(value_loss)
```

That is a training detail rather than a network-layer detail, but it matters for how the two heads are balanced in practice.

## Export and Runtime Inference Path

The repo does not run the monolithic training model directly inside Rust for normal inference. Instead it exports a split runtime representation:

- `latest.conv.onnx`
- `latest.heads.onnx`
- `latest.fc.bin`

### What Each Artifact Contains

#### `conv.onnx`

Exported from `ConvBackbone`.

For `gated_fusion`, it contains:

- `conv_initial`
- initial `GroupNorm`
- all conv residual blocks
- `conv_reduce`
- reduction `GroupNorm`
- flatten

Input:

- `board`: `(B, 1, 20, 10)`

Output:

- `conv_out`: `(B, 1600)` for the default config

#### `fc.bin`

This is not ONNX. It is the raw `board_proj` weight and bias written as:

```text
[rows u32 LE][cols u32 LE][weight row-major f32][bias f32]
```

For the default config:

- rows = `128`
- cols = `1619`

Rust loads this file and computes:

```text
board_h = board_proj([conv_out ; board_stats])
```

directly in Rust, without needing another ONNX session for that layer.

#### `heads.onnx`

Exported from `HeadsModel`.

For `gated_fusion`, it contains:

- `aux_fc`
- `aux_ln`
- `gate_fc`
- `aux_proj`
- `fusion_ln`
- `fusion_blocks`
- `policy_fc`
- `policy_head`
- `value_fc`
- `value_head`

Inputs:

- `board_h`: `(B, 128)`
- `piece_aux`: `(B, 61)`

Outputs:

- `policy_logits`: `(B, 735)`
- `value`: `(B, 1)`

### Rust Runtime Cache

Rust wraps the split model in `TetrisNN` and caches only the board embedding:

- Cache key: the `200` binary board cells packed losslessly into `[u64; 4]`
- Cache value: `board_h`

Inference flow in Rust is:

1. Encode the board occupancy tensor.
2. Encode the 19 board-stat features.
3. If the packed board is not in cache:
   - run `conv.onnx`
   - concatenate `conv_out` with the 19 board stats
   - apply the `board_proj` weights from `fc.bin`
   - store `board_h` in the cache
4. Encode the 61 piece/game features.
5. Run `heads.onnx` on `(board_h, piece_aux)`.
6. Apply softmax over only valid actions.

This is why the feature split matters:

- board-only information is cached
- piece/game context stays uncached and cheap to recompute

## Alternative Architecture: `simple_aux_mlp`

The repo also supports `architecture="simple_aux_mlp"`.

This architecture keeps the same external model interface and the same split-export contract, but it does not use the board tensor for actual prediction.

### Forward Path

```text
board -> summed to a dummy scalar -> multiplied by 0
board_stats -> passed through a frozen identity-like board_proj
piece_aux + board_h(=board_stats) -> concat to full_aux[80]
full_aux -> Linear(80,128) -> LayerNorm -> SiLU
        -> Linear(128,735) = policy_logits
        -> Linear(128,1)   = value
```

More precisely:

- `board_proj` is defined as `Linear(1 + 19, 19)`
- its weights are frozen
- its first input column, the dummy board scalar, is forced to zero contribution
- its remaining `19` columns are set to the identity matrix

So in effect:

```text
board_h == board_stats
full_aux == concat(piece_aux, board_stats)
```

### Why It Exists

It is a baseline architecture that:

- preserves the same ONNX signatures as the default model
- preserves the same Rust split-runtime contract
- provides a lower-capacity comparison point

### Parameter Count

With the default `fc_hidden=128`:

- Total parameters: `105,967`
- Trainable parameters: `105,568`

The difference is the frozen `board_proj` identity map.

## Differences From Older Repo Docs

If you have seen older architecture descriptions in this repo, the main changes are:

- Current input contract is `280` total features, not `297`.
- Current default model is `gated_fusion`, not a simple board-flatten-then-concat MLP.
- Current conv path uses `GroupNorm + SiLU + residual blocks`, not `BatchNorm + ReLU`.
- Current runtime uses split export plus a cached `board_h` embedding in Rust.
- Current board-stat vector keeps only the bottom `4` row-fill features, not all `20` rows.

## Practical Summary

If you want the one-sentence description of the production model used here, it is:

> A split `gated_fusion` policy/value network that turns the board into a cached `128`-dimensional embedding, modulates that embedding with `61` piece/game features, and predicts `735` action logits plus a scalar value.

If you change any of the following, you must keep Python and Rust in sync:

- feature ordering or feature count
- `board_proj` input or output size
- the shape of `board_h`
- the action-space size
- ONNX split-export assumptions
