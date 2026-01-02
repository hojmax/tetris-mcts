So we train want to figure out how to run some CNN quickly from Rust. So we want to be able to import some specified pytorch weights into a rust and run it over a state quickly, and some units tests that the pytorch application of this and the rust one does the same. We probbaly want to just run this on a CPU.

So the architecture is something like:

- CNN with two kernels over the board.
- Flatten
- Concat history of one-hotnencoding of next 5 pieces
- Fully connected layer to 734 (all possible placements of a piece) outputs. Plus softmax to get distribution.
* And another head for value estimate.
* So two heads, policy and value head.

Input:
* Board (20 * 10 bools)
* Hold piece (one hot encoded 7 values)
* Next 5 queue pieces (one hot encoded 5 * 7 values)
* Whether hold has been used. (single bool)

Output:
- 1 (value estimate)
* 734 (all possible placements of a piece)


Training:
* Random initialization
* Play a bunch of rust games.
* Some epsilon for a random piece placement.
* Save the game states to file.
* Train model to predict value estimates
* WAndb logging of everything
* Alpha zero style training. So value estimate predicts the reward + discounted future reward.
* Policy head predicts the distributions over the nodes visisted during search.
* Please read 1712.01815v1.pdf for more details.
* Difference is we train a tiny network which is then saved to disk and loaded and run with rust. We do the tree search in rust only, and training in python. So the tree structure
* I wonder if faster to call the gpu from rust for some of the kernel operations? Like how fast is it to do CPU only for these small networks, vs. calling gpu from rust.