# Next Steps

- [ ] Continue training with more steps
- [ ] Look at tetris_mcts/scripts/buffer_viewer.py
- [ ] Caching board representation from network in inference.
- [ ] Storing boards in three 64bit numbers (1 bit per field, 20x10=200 fields, 200 bits). Useful for caching.
- [ ] Performance profiler on 10 games generated.

- [ ] Visualizing MCTS search and verifying correctness
- [ ] Looking at evaluation data on wandb
- [ ] Code for visualizing trajectories
- [ ] Looking at training data
- [ ] Play the tetris game to ensure the environment is working correctly
- [ ] Save a full rollout tree from during training and inspect it with `make viz` tool
- [ ] Look at training rollouts
- [ ] Speeding up masking of moves
- [ ] Reading through and validating all code
- [ ] Stress test next possible pieces with unit tests, like that it can do wild twists and stuff and that it correctly decides those possible locations
- [ ] Benchmarking and improving speed of MCTS search
- [ ] Testing speed on different hardware then the Macbook Air
