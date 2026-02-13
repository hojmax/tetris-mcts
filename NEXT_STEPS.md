# Next Steps

- [ ] Deeper search depth potentially.
- [ ] What AWS instance would be well suited for this workload?
- [ ] Do offline learning experiments. Things to validate:
  - Learning rate and scheduler
  - Model size
  - Batch size
  - Policy / Value loss weighting
  - Weight decay
  - Optimizer
  - Architecture choices
  - Huber loss vs. MSE loss

- [ ] Run with updated feature set, integrate all the new features.
- [ ] Should the network be predicting the overhang penalty? Or is that just a search heuristic?
- [ ] Another round of benchmarking and optimizing?
- [ ] Benchmark conv net depth impact on speed, since caching so high. 96% caching.
- [ ] Adding in alpha downweighting of value loss?
- [ ] predict "n-step bootstrapped return" instead of "cummulative reward"

# Deep Review

Done: ✅
In progress: 🟨

## tetris_core/src/scoring.rs ✅

## tetris_core/src/piece.rs ✅

## tetris_core/src/nn.rs ✅

## tetris_core/src/moves.rs ✅

## tetris_core/src/kicks.rs ✅

## tetris_core/src/constants.rs ✅

## tetris_core/src/env/board.rs ✅

## tetris_core/src/env/clearing.rs ✅

# Backlog

- [ ] Visualizing MCTS search and verifying correctness
- [ ] Reading through and validating all code
- [ ] Proper network split caching. Caching board CNN head, and optimizing such that we only run last part of network.
- [ ] Do we need a larger network?
- [ ] What alternative hardware could we run on?
- [ ] Would an instance with a GPU be better? What is the major bottleneck, CPU or GPU? Like maybe AWS instance with a ton of CPUs for deep tree search.
- [ ] Looking at training data
- [ ] Caching board representation from network in inference.
- [ ] Optimize int4 and int8 everywhere.
- [ ] More rust profiling
- [ ] Better splitting up of rust between the environment and the MCTS. Two different packages / folders.
- [ ] Play the tetris game to ensure the environment is working correctly
- [ ] Save a full rollout tree from during training and inspect it with `make viz` tool
- [ ] Stress test next possible pieces with unit tests, like that it can do wild twists and stuff and that it correctly decides those possible locations
- [ ] Benchmarking and improving speed of MCTS search
- [ ] Testing speed on different hardware then the Macbook Air
- [ ] Sweep over best value head weighting.
- [ ] Maybe this is just a hella slow learning algorithm, and we need to scale up compute.
- [ ] Adding hand crafted heuristics to offload work off the neural network
  - can t-spin filter?
- [ ] Try to handcraft great Tetris bot.
- [ ] All the steps setttings depend on batch size which is kind of annoying. Maybe wall clock is nicer for all these settings?

## Test Environment State

Current Piece:

```
T
```

Queue:

```
I,O,L,S,Z
```

Board:

```
..........
..........
..........
..........
..........
..........
..........
..........
..........
..........
..........
..........
..........
..........
..........
..........
..........
....LLL...
LL...LLLLL
LLL.LLLLLL
```

# Backwards Compatibility

- [ ] Remove the backwards compatibility code for the raw values.:

```

• Edited tetris_core/src/generator/npz.rs (+69 -3)
    235          read_npy_array::<f32>(&mut archive, "value_targets.npy")?;
    236 -    let (raw_value_targets, raw_value_targets_shape) =
    237 -        read_npy_array::<f32>(&mut archive, "raw_value_targets.npy")?;
    236 +    let raw_value_targets_optional =
    237 +        read_optional_npy_array::<f32>(&mut archive, "raw_value_targets
         .npy")?;
    238      let (action_masks, action_masks_shape) =
        ⋮
    272      validate_shape("value_targets", &value_targets_shape, &[n as u64])?
         ;
    273 -    validate_shape("raw_value_targets", &raw_value_targets_shape, &[n a
         s u64])?;
    273 +    let raw_value_targets = if let Some((values, shape)) = raw_value_ta
         rgets_optional {
    274 +        validate_shape("raw_value_targets", &shape, &[n as u64])?;
    275 +        values
    276 +    } else {
    277 +        vec![0.0; n]
    278 +    };
    279      validate_shape(
        ⋮
    416
    417 +fn read_optional_npy_array<T: npyz::Deserialize>(
    418 +    archive: &mut ZipArchive<File>,
    419 +    entry_name: &str,
    420 +) -> Result<Option<(Vec<T>, Vec<u64>)>, String> {
    421 +    let file = match archive.by_name(entry_name) {
    422 +        Ok(file) => file,
    423 +        Err(zip::result::ZipError::FileNotFound) => return Ok(None),
    424 +        Err(error) => return Err(error.to_string()),
    425 +    };
    426 +    let npy = NpyFile::new(file).map_err(|e| e.to_string())?;
    427 +    let shape = npy.shape().to_vec();
    428 +    let values = npy.into_vec::<T>().map_err(|e| e.to_string())?;
    429 +    Ok(Some((values, shape)))
    430 +}
    431 +
    432  fn read_npy_array_bool_like(
        ⋮
    523
    524 +    fn write_npz_without_entry(source: &PathBuf, destination: &PathBuf,
          missing_entry: &str) {
    525 +        let input = File::open(source).expect("source npz should exist"
         );
    526 +        let mut input_zip = ZipArchive::new(input).expect("source npz s
         hould be valid zip");
    527 +        let output = File::create(destination).expect("destination npz
         should be writable");
    528 +        let mut output_zip = ZipWriter::new(output);
    529 +        let options = FileOptions::default().compression_method(Compres
         sionMethod::Deflated);
    530 +
    531 +        for i in 0..input_zip.len() {
    532 +            let mut entry = input_zip
    533 +                .by_index(i)
    534 +                .expect("entry index should be valid in source npz");
    535 +            let entry_name = entry.name().to_string();
    536 +            if entry_name == missing_entry {
    537 +                continue;
    538 +            }
    539 +            output_zip
    540 +                .start_file(entry_name, options)
    541 +                .expect("writing destination entry should succeed");
    542 +            std::io::copy(&mut entry, &mut output_zip).expect("copying
         entry bytes should work");
    543 +        }
    544 +
    545 +        output_zip
    546 +            .finish()
    547 +            .expect("finishing destination npz should succeed");
    548 +    }
    549 +
    550      #[test]
        ⋮
    580      #[test]
    581 +    fn test_read_npz_defaults_missing_raw_value_targets_to_zero() {
    582 +        let source_path = unique_temp_path("missing_raw_source");
    583 +        let compat_path = unique_temp_path("missing_raw_compat");
    584 +        let examples = vec![make_example(0, 7), make_example(88, 5)];
    585 +
    586 +        write_examples_to_npz(&source_path, &examples, 100).expect("wri
         te should succeed");
    587 +        write_npz_without_entry(&source_path, &compat_path, "raw_value_
         targets.npy");
    588 +
    589 +        let loaded = read_examples_from_npz(&compat_path, 100).expect("
         read should succeed");
    590 +
    591 +        fs::remove_file(&source_path).expect("source temp file cleanup
         should succeed");
    592 +        fs::remove_file(&compat_path).expect("compat temp file cleanup
         should succeed");
    593 +
    594 +        assert_eq!(loaded.len(), examples.len());
    595 +        for loaded_example in loaded {
    596 +            assert_eq!(loaded_example.raw_value, 0.0);
    597 +        }
    598 +    }
    599 +
    600 +    #[test]
    601      fn test_validate_shape_with_dynamic_batch_rejects_mismatched_dims()
          {

────────────────────────────────────────────────────────────────────────────────

• I’ve implemented the loader fallback in Rust and added a test for missing
  raw_value_targets.npy. I’ll also add the same fallback in the analysis script
  and update AGENTS context so behavior is documented.

• Edited tetris_mcts/scripts/analyze_training_data.py (+4 -1)
    69      value_targets = data["value_targets"]
    70 -    raw_value_targets = data["raw_value_targets"]
    70 +    if "raw_value_targets" in data:
    71 +        raw_value_targets = data["raw_value_targets"]
    72 +    else:
    73 +        raw_value_targets = np.zeros(n_examples, dtype=np.float32)
    74      if len(move_numbers) != n_examples:

• Edited CLAUDE.md (+1 -0)
    269  10. `training_data.npz` snapshots include `raw_value_targets` (per-stat
         e cumulative raw attack), `game_numbers` (1-indexed WandB game ids), an
         d `game_total_attacks` (raw per-game attack) for exact replay/WandB ali
         gnment
    270 +11. Backward compatibility: if loading an older snapshot without `raw_v
         alue_targets.npy`, Rust loader defaults `raw_value` to `0.0` for all ex
         amples
    271

────────────────────────────────────────────────────────────────────────────────

```
