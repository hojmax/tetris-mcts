
# Next Steps

- [ ] Fix hold move bug in mcts. Like why are we tracking x and y and rotation, and just raw move indices? Why are we not passing a single int around corresponding to a specific move? We have the lookup for this. And hold should be a move.
- [ ] Take an interestnig state like tetris_mcts/scripts/outputs/game_3194.gif at step 14 (oppurtinity for t spin), and see how the model evaluates that state? Is it not reached or something?
- [ ] No need to show Value and attack in tetris_mcts/scripts/outputs/game_3194.gif? They are the same? Just skip the value one, and show the attack. They are reading from the same field right? Or where is the attack ocming from?
- [ ] When inspecting games, with the training viewer, I need to see network predictions. At least value predictions.
- [ ] What happened with the LR lol? Why did it go so low, even though we trained with 0.5 lr min?
- [ ] Do we have a mismatch between the action somewhere? Like training on one set of indeces and taking actions with another? This would shift all the agents agents / randomize it? Need to check for this.
- [ ] tetris_mcts/scripts/inspect_training_data.py not showing can hold?
- [ ] game/avg_moves, game/max_moves and not eval/...
- [ ] I don't think the eval/trajectory matches the eval/max_attack?? Is the replay correct?
- [ ] Verify that the inference and training netwroks are tacking the exavt same inputs in the exavt same ordering and the exavt same scaling and values and all that. Make unit tests for this as well. Also the masked softmax and all that. We need tests that compare the two netwrok files and check equal outputs. Might need to be a pytest? Add pytest for this that runs both and compares inputs and outputs that are written to files or something like that.

# Deep Review

## tetris_core/src/scoring.rs ✅

## tetris_core/src/piece.rs ✅

## tetris_core/src/nn.rs ✅

## tetris_core/src/moves.rs 🟨

# Backlog

- [ ] Proper network split caching. Caching board CNN head, and optimizing such that we only run last part of network.
- [ ] Do we need a larger network?
- [ ] What alternative hardware could we run on?
- [ ] Would an instance with a GPU be better? What is the major bottleneck, CPU or GPU? Like maybe AWS instance with a ton of CPUs for deep tree search.
- [ ] Looking at training data
- [ ] Caching board representation from network in inference.
- [ ] Optimize int4 and int8 everywhere.
- [ ] More rust profiling
- [ ] Better splitting up of rust between the environment and the MCTS. Two different packages / folders.
- [ ] Visualizing MCTS search and verifying correctness
- [ ] Play the tetris game to ensure the environment is working correctly
- [ ] Save a full rollout tree from during training and inspect it with `make viz` tool
- [ ] Reading through and validating all code
- [ ] Stress test next possible pieces with unit tests, like that it can do wild twists and stuff and that it correctly decides those possible locations
- [ ] Benchmarking and improving speed of MCTS search
- [ ] Testing speed on different hardware then the Macbook Air
