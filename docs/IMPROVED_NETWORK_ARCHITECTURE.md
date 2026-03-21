Absolutely — here is a more detailed write-up of that concrete version.

---

# Proposed Architecture: Cached Spatial Backbone + Late FiLM + Spatial Policy

This version keeps the best part of the current design — **board-only caching** — but makes the prediction path more spatial and simpler.

The main idea is:

- compute a **board-only spatial embedding**
- optionally condition that embedding with **board-only handcrafted stats**
- **cache** that embedding
- inject **piece / queue / hold / combo / B2B** only **after** the cache point
- produce:
  - a **spatial placement policy** over `4 x 20 x 10`
  - a separate **hold logit**
  - a **scalar value**

This is intended as a replacement for the current flatten-then-fuse `gated_fusion` setup.

---

## High-level summary

**Inputs**

- board occupancy: `(B, 1, 20, 10)`
- board stats: `(B, 19)` — board-only handcrafted features
- dynamic aux: `(B, 61)` — current piece, hold, queue, combo, B2B, etc.

**Outputs**

- placement logits: `(B, 4, 20, 10)` → flattened to `800`
- hold logit: `(B, 1)`
- final policy logits: `(B, 801)`
- value: `(B, 1)`

**Cache**

- cache a spatial board embedding of shape:

[
(B, 16, 10, 5)
]

That is `16 * 10 * 5 = 800` floats per cached board.

This is much larger than the current `128`-vector cache, but still much smaller than caching a full-resolution `32 x 20 x 10` map.

---

# 1. Input contract

## 1.1 Board input

Instead of feeding only occupancy, I would feed **occupancy + coordinates**.

### Raw board

- shape: `(B, 1, 20, 10)`
- values:
  - `1.0` for filled
  - `0.0` for empty

### Coordinate channels

Add two fixed channels:

- **row coordinate**: shape `(1, 20, 10)`
  - normalized from bottom to top or top to bottom

- **column coordinate**: shape `(1, 20, 10)`
  - normalized left to right

So actual conv input is:

[
board_in \in \mathbb{R}^{B \times 3 \times 20 \times 10}
]

This helps because Tetris is not translation-invariant:

- vertical position matters a lot
- column position matters somewhat
- local patterns mean different things at different heights

A reasonable normalization is:

- row coord in `[-1, 1]`
- col coord in `[-1, 1]`

---

## 1.2 Auxiliary split

Keep the same semantic split as the current model:

### `board_stats` — board-only, cache-compatible

shape `(B, 19)`

These are the handcrafted board features such as:

- column heights
- max height
- bottom row fill counts
- total blocks
- bumpiness
- holes
- overhang fields

These can be used **before the cache point** because they depend only on the board.

### `dynamic_aux` — not cache-compatible

shape `(B, 61)`

These include:

- current piece
- hold piece
- hold availability
- next queue
- placement count
- combo
- B2B
- hidden bag distribution

These must only be used **after** the cache point.

---

# 2. Proposed architecture

Here is the full tensor flow.

---

## 2.1 Pre-cache board backbone

### Stage A: stem

```text
board_in (B,3,20,10)
 -> Conv2d(3,32,kernel=3,padding=1)
 -> GroupNorm
 -> SiLU
 = x0 : (B,32,20,10)
```

### Stage B: pre blocks at full resolution

```text
x0
 -> 4 x ResidualConvBlock(32)
 = x1 : (B,32,20,10)
```

### Stage C: downsample to cache resolution

```text
x1
 -> Conv2d(32,16,kernel=3,stride=2,padding=1)
 -> GroupNorm
 -> SiLU
 = x2 : (B,16,10,5)
```

### Stage D: a couple of pre-cache blocks

```text
x2
 -> 2 x ResidualConvBlock(16)
 = x3 : (B,16,10,5)
```

At this point, the board has been processed into a compact spatial representation.

---

## 2.2 Board-stats conditioning before cache

Encode the `19` board-only handcrafted features with a small MLP:

```text
board_stats (B,19)
 -> Linear(19,32)
 -> LayerNorm(32)
 -> SiLU
 = z_board : (B,32)
```

Then use `z_board` to FiLM-modulate the two `16`-channel pre-cache blocks.

For each of the 2 pre-cache residual blocks, produce per-channel FiLM parameters:

```text
z_board
 -> Linear(32,32)
 -> split into gamma_i (16), beta_i (16)
```

Apply them channelwise:

[
\mathrm{FiLM}(h, z) = \gamma(z) \odot h + \beta(z)
]

with broadcasting over spatial dimensions.

### Important detail

I would parameterize this as an **identity-centered** modulation:

[
\gamma(z) = 1 + \Delta\gamma(z)
]

and initialize `Δγ` and `β` near zero.

That makes the network start close to a plain ResNet.

So conceptually, pre-cache blocks become:

```text
input
 -> GroupNorm
 -> FiLM(board_stats)
 -> SiLU
 -> Conv
 -> GroupNorm
 -> FiLM(board_stats)
 -> SiLU
 -> Conv
 + skip
```

This lets board stats refine the cached representation, while keeping the cache board-only.

---

## 2.3 Cache point

Cache the tensor:

[
cached_board = x3 \in \mathbb{R}^{B \times 16 \times 10 \times 5}
]

This is the runtime cache value keyed only by board occupancy.

### Cache size

- floats per board: `16 * 10 * 5 = 800`
- if `float32`: about `3.2 KB`
- if `float16`: about `1.6 KB`

So this is much bigger than the current `128`-float cache, but still fairly manageable.

---

# 3. Post-cache shared trunk

After retrieving the cached board embedding, inject dynamic features and lift back toward full spatial resolution.

---

## 3.1 Decode cached map back to richer spatial features

```text
cached_board (B,16,10,5)
 -> Upsample(scale_factor=2, mode='nearest' or bilinear)
 -> Conv2d(16,32,kernel=3,padding=1)
 -> GroupNorm
 -> SiLU
 = y0 : (B,32,20,10)
```

This step reconstructs a full-resolution spatial map for the heads.

I would usually prefer:

- nearest-neighbor upsampling + `3x3` conv
  over
- transposed convolution

because it is simpler and less artifact-prone.

---

## 3.2 Dynamic aux encoder

Encode the 61 dynamic features:

```text
dynamic_aux (B,61)
 -> Linear(61,64)
 -> LayerNorm(64)
 -> SiLU
 -> Linear(64,64)
 -> LayerNorm(64)
 -> SiLU
 = z_aux : (B,64)
```

This `z_aux` is the main conditioning signal for:

- current piece
- hold
- queue
- combo/B2B
- hidden bag information

---

## 3.3 Shared post-cache FiLM ResBlocks

Run a shared post-cache trunk before splitting into policy and value heads:

```text
y0
 -> 2 x FiLMResidualConvBlock(32, conditioned on z_aux)
 = y_shared : (B,32,20,10)
```

For each of the 2 shared post-cache blocks, generate FiLM parameters from `z_aux`:

```text
z_aux
 -> Linear(64,64)
 -> split into gamma_i (32), beta_i (32)
```

again using identity-centered scaling:
[
\gamma_i(z) = 1 + \Delta\gamma_i(z)
]

This is the point where:

- same board
- different current piece / queue / hold
  can start producing different spatial interpretations.

That is exactly what you want.

---

# 4. Policy head

The policy has two parts:

- **placement logits** over `4 x 20 x 10`
- **hold logit** as a separate scalar

---

## 4.1 Policy spatial trunk

```text
y_shared
 -> 2 x FiLMResidualConvBlock(32, conditioned on z_aux)
 = y_policy : (B,32,20,10)
```

I would condition these policy blocks on `z_aux` as well, because:

- current piece strongly affects which placements matter
- queue / hold also affect preference among placements

---

## 4.2 Placement logits

Project to 4 rotation channels:

```text
y_policy
 -> Conv2d(32,4,kernel=1)
 = placement_logits : (B,4,20,10)
```

This is the cleanest final projection.

Then flatten:

```text
placement_logits_flat = reshape(B, 800)
```

### Meaning of the 4 channels

Each channel corresponds to a rotation index.

So each logit corresponds to:

[
(rotation,\ y,\ x)
]

This gives 800 candidate placements.

Invalid placements are masked out outside the network.

---

## 4.3 Hold head

The hold action is not a placement, so it should be a separate branch.

I would make it depend on both:

- pooled policy/shared spatial features
- dynamic aux directly

### Hold head flow

```text
y_policy : (B,32,20,10)
 -> GlobalAveragePool
 = pooled_policy : (B,32)

hold_input = concat([pooled_policy, z_aux]) : (B,96)

hold_input
 -> Linear(96,32)
 -> SiLU
 -> Linear(32,1)
 = hold_logit : (B,1)
```

This is better than using only pooled spatial features, because hold depends heavily on:

- hold availability
- current piece
- held piece
- queue

---

## 4.4 Final policy output

Concatenate:

```text
policy_logits = concat([placement_logits_flat, hold_logit], dim=-1)
```

So final shape is:

[
policy_logits \in \mathbb{R}^{B \times 801}
]

---

# 5. Value head

The value head should stay spatial longer, but not collapse to 1 channel too early.

---

## 5.1 Value spatial trunk

```text
y_shared
 -> 2 x FiLMResidualConvBlock(32, conditioned on z_aux)
 = y_value : (B,32,20,10)
```

---

## 5.2 Value reduction

Instead of reducing directly to 1 channel, reduce to 8 channels:

```text
y_value
 -> Conv2d(32,8,kernel=1)
 -> GroupNorm
 -> SiLU
 = v0 : (B,8,20,10)
```

This preserves more information before flattening.

---

## 5.3 Value MLP

Flatten:

```text
v0 -> flatten
= v_flat : (B,1600)
```

Then concatenate `z_aux`:

```text
value_input = concat([v_flat, z_aux]) : (B,1664)
```

Then:

```text
value_input
 -> Linear(1664,128)
 -> SiLU
 -> Linear(128,1)
 = value : (B,1)
```

This keeps value sensitive to:

- spatial board structure
- dynamic context

which is important in Tetris.

---

# 6. Residual block definitions

I would use one standard residual block for both shared trunk and heads, with optional FiLM.

---

## 6.1 Plain residual conv block

```text
input
 -> GroupNorm
 -> SiLU
 -> Conv2d(C,C,3,padding=1)
 -> GroupNorm
 -> SiLU
 -> Conv2d(C,C,3,padding=1)
 + skip
```

---

## 6.2 FiLM residual conv block

Same block, but insert FiLM after normalization:

```text
input
 -> GroupNorm
 -> FiLM(z)
 -> SiLU
 -> Conv2d(C,C,3,padding=1)
 -> GroupNorm
 -> FiLM(z)
 -> SiLU
 -> Conv2d(C,C,3,padding=1)
 + skip
```

where

[
\mathrm{FiLM}(h,z) = \gamma(z)\odot h + \beta(z)
]

with:

- `γ(z)` and `β(z)` shaped `(B, C)`
- broadcast across `H, W`

---

# 7. Full end-to-end tensor flow

Here is the entire proposed architecture in one block.

```text
board_raw (B,1,20,10)
coords -> append
board_in (B,3,20,10)

board_in
 -> Conv2d(3,32,3,pad=1)
 -> GN -> SiLU
 -> 4 x ResidualConvBlock(32)
 -> Conv2d(32,16,3,stride=2,pad=1)
 -> GN -> SiLU
 -> 2 x FiLMResidualConvBlock(16, z_board)
 = cached_board (B,16,10,5)

board_stats (B,19)
 -> Linear(19,32) -> LN -> SiLU
 = z_board (B,32)

CACHE HERE

cached_board
 -> Upsample x2
 -> Conv2d(16,32,3,pad=1)
 -> GN -> SiLU
 = y0 (B,32,20,10)

dynamic_aux (B,61)
 -> Linear(61,64) -> LN -> SiLU
 -> Linear(64,64) -> LN -> SiLU
 = z_aux (B,64)

y0
 -> 2 x FiLMResidualConvBlock(32, z_aux)
 = y_shared (B,32,20,10)

POLICY:
y_shared
 -> 2 x FiLMResidualConvBlock(32, z_aux)
 = y_policy (B,32,20,10)
 -> Conv2d(32,4,1)
 = placement_logits (B,4,20,10)
 -> flatten
 = placement_logits_flat (B,800)

GAP(y_policy) = pooled_policy (B,32)
concat([pooled_policy, z_aux]) (B,96)
 -> Linear(96,32) -> SiLU -> Linear(32,1)
 = hold_logit (B,1)

policy_logits = concat([placement_logits_flat, hold_logit])
 = (B,801)

VALUE:
y_shared
 -> 2 x FiLMResidualConvBlock(32, z_aux)
 = y_value (B,32,20,10)
 -> Conv2d(32,8,1)
 -> GN -> SiLU
 = v0 (B,8,20,10)
 -> flatten
 = v_flat (B,1600)

concat([v_flat, z_aux]) = (B,1664)
 -> Linear(1664,128)
 -> SiLU
 -> Linear(128,1)
 = value (B,1)
```

---

# 8. Runtime split / export path

This architecture still supports a clean split runtime path.

---

## 8.1 Pre-cache export

### `conv.onnx`

Contains:

- input coord augmentation can happen outside ONNX or inside Python before export
- stem conv
- 4 pre blocks
- stride-2 reduction
- 2 pre-cache blocks
- board-stats conditioning path if you want it exported here

Inputs:

- `board`: `(B, 1, 20, 10)` or `(B, 3, 20, 10)` depending on where coords are added
- `board_stats`: `(B, 19)` if board-stats FiLM is inside this graph

Output:

- `cached_board`: `(B,16,10,5)`

### Cache key

Still keyed only by board occupancy.

Because `board_stats` are deterministic from the board, they are cache-compatible.

---

## 8.2 Heads export

### `heads.onnx`

Contains:

- upsample / decode
- dynamic aux encoder
- shared post-cache blocks
- policy trunk
- hold head
- value head

Inputs:

- `cached_board`: `(B,16,10,5)`
- `dynamic_aux`: `(B,61)`

Outputs:

- `policy_logits`: `(B,801)`
- `value`: `(B,1)`

---

# 9. Action-space mapping

This version changes the policy layout.

Instead of:

- 734 placement logits
- 1 hold logit

you would now have:

- 800 spatial placement logits
- 1 hold logit

So total:

- `801`

The runtime must define a fixed mapping from `(rotation, y, x)` to policy index.

A natural flattening is:

[
index = rotation * 200 + y * 10 + x
]

with:

- `rotation in [0,3]`
- `y in [0,19]`
- `x in [0,9]`

Then:

- `0..799` are placements
- `800` is hold

Invalid placements are masked exactly as before.

---

# 10. Why this version is better than the current one

I think this version improves the current model in four ways.

### 1. Policy is spatial

The policy head now matches the structure of the task:

- piece placement is spatial
- output is spatial

That is a better inductive bias than `128 -> 256 -> 735`.

### 2. Caching is preserved

You still avoid recomputing the expensive board trunk for every queue / hold / combo variation on the same board.

### 3. Dynamic features are injected where they matter

Queue/current-piece/hold do not contaminate the cache.
They only affect post-cache interpretation of the board.

### 4. Fusion becomes simpler and more expressive

FiLM over spatial features is more natural than:

- compress board to vector
- multiply by a gate in `(1,2)`
- add another vector
- hope the MLP sorts it out

---

# 11. What I would treat as the default hyperparameters

If you want a concrete default config, I would start with:

```text
stem_channels = 32
cache_channels = 16

num_pre_blocks_fullres = 4
num_pre_blocks_cached = 2
num_shared_post_blocks = 2
num_policy_blocks = 2
num_value_blocks = 2

board_stats_hidden = 32
dynamic_aux_hidden = 64
hold_hidden = 32
value_hidden = 128

activation = SiLU
norm_conv = GroupNorm
norm_mlp = LayerNorm
```

That is a very reasonable first version.

---

# 12. A few ablations I would test first

The most important ones would be:

### A. With vs without board-stats FiLM

This tells you whether handcrafted board stats still help once the spatial trunk is stronger.

### B. `cached_board = 16x10x5` vs `8x10x5`

This directly tests cache size vs strength.

### C. Value head flatten vs global pool

You may find that a simpler value head works almost as well.

### D. With vs without coord channels

I would expect coord channels to help, but it is cheap to verify.

---

# 13. My one-line recommendation

If I had to commit to one architecture right now, it would be this exact family:

> **board-only conv backbone → cache `16x10x5` spatial embedding → late FiLM conditioning by dynamic aux → spatial `4x20x10` placement head + separate hold head + modest spatial value head**

That is the version I would build first.

I can also turn this into a repo-style architecture doc matching the format of your current markdown.
