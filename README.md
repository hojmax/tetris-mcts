# tetris-mcts

An AlphaZero-style reinforcement learning system for Tetris. A neural network
guides a Monte Carlo Tree Search that plays the game, generates its own training
data through self-play, and improves by learning from it.

- **Rust** for the game engine, MCTS, and inference hot path.
- **Python / PyTorch** for the network, training loop, and analysis tooling.
- **PyO3 + ONNX** bridge the two: Python trains, Rust plays.

## How it fits together

```
   ┌─────────────────────┐   ONNX export    ┌──────────────────────┐
   │   PyTorch trainer   │ ───────────────▶ │  Rust self-play +    │
   │  (policy + value)   │                  │  MCTS workers        │
   └──────────┬──────────┘                  └──────────┬───────────┘
              ▲                                        │
              │         replay buffer                  │
              └────────────────────────────────────────┘
                 (boards, MCTS visit policies, values)
```

Workers play self-play games in Rust using an ONNX copy of the current network.
Each game writes `(board, aux features, MCTS visit distribution, value target)`
tuples into a shared ring buffer. The Python trainer samples from that buffer,
updates the policy + value heads, and periodically exports a new ONNX
candidate. The candidate only replaces the incumbent used by workers if it
wins a fixed-seed evaluation window against the current incumbent — so
regressions never poison the replay buffer.

## MCTS

Tetris is single-player and stochastic, so the search tree alternates two node
types:

- **Decision nodes** where the agent picks an action (a placement cell or
  hold).
- **Chance nodes** where the 7-bag draws the next piece.

Selection at decision nodes uses the standard **PUCT** rule from AlphaZero:

```
a* = argmax_a  Q(s, a) + c_puct · P(s, a) · √N(s) / (1 + N(s, a))
```

- `P(s, a)` is the network's masked policy prior over legal actions.
- `Q(s, a)` is the mean backed-up value for action `a`. During search, Q terms
  are min-max normalized globally so PUCT stays well-scaled across trees.
- `c_puct` defaults to `1.5`; `num_simulations` defaults to `2000` per move.

## Network

The network is deliberately small (~1.3–1.8M parameters depending on config)
and built around one idea: **the board embedding should be cached, and
everything else should be cheap to recompute**.

**Inputs.**

- `board`: `(B, 1, 20, 10)` binary occupancy.
- `aux_features`: `(B, 80)`, split internally as `61` piece/game features
  (current piece, hold, 5-piece queue, combo, B2B, hidden-piece distribution,
  …) and `19` board-derived stats (column heights, holes, bumpiness, …).

## Training loop

- **Self-play workers** (Rust) play full games to game-over and write tuples
  into a ring buffer (default size ~7M).
- **Trainer** (Python) samples mini-batches (default `2048`) and minimizes
  `policy_loss + value_loss`
- **Candidate gating.** The trainer periodically exports an ONNX candidate
  and runs a fixed-seed eval window (default 20 games, no Dirichlet noise,
  fixed MCTS seed) against the current incumbent. The candidate promotes
  only if it wins on average; eval trajectories are _not_ added to the replay
  buffer. Set `self_play.use_candidate_gating=false` to bypass this and push
  new exports directly to all workers.

## Quick start

Requires Python 3.12 or 3.13 (not 3.14 — `onnxruntime` has no wheels yet).

```bash
make install     # uv sync + Rust toolchain + debug extension build
make build       # optimized release build of the Rust extension
make play        # interactive Tetris (human play)
make viz         # MCTS tree visualizer
make test        # Rust + Python test suites
make check       # ruff + pyright + rustfmt + clippy
```

`make install` bootstraps `uv`, a Rust toolchain via `rustup`, Python deps, and
a debug PyO3 extension. On Linux it also tries to install ORT build
prerequisites (`pkg-config`, OpenSSL headers, `patchelf`); pass
`AUTO_INSTALL_SYSTEM_DEPS=0` to skip.

## Training a model

```bash
# tune worker/backend/build settings for this machine (cached per-machine)
make optimize

# start training from scratch
make train

# or drive the script directly
uv run python tetris_bot/scripts/train.py --config config.yaml
```

## Documentation

- [`paper/paper.pdf`](paper/paper.pdf) — write-up, in progress.
- [`CLAUDE.md`](CLAUDE.md) — for AI assistants working in the repo.
