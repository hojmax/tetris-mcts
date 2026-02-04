# Next Steps

- [ ] Performance profiler on 10 games generated.
- [ ] is_valid_position_at: 6.5% on 85 calls??
- [ ] find_all_placements: Optimize find all placements (63% of runtime?)
- [ ] predict_masked: 17% of runtime
- [ ] load_model: Load model called 32 times??

- [ ] Fix the quicktest failures.
- [ ] Look around and add more tests.


- [ ] Continue training with more steps
- [ ] Look at tetris_mcts/scripts/buffer_viewer.py
- [ ] Caching board representation from network in inference.
- [ ] Storing boards in four 64bit numbers (1 bit per field, 20x10=200 fields, 200 bits). Useful for caching.

- [ ] Optimize int4 and int8 everywhere.
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
