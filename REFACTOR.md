# Refactor

## Points of improvement

- Remove old bloat
  - Tetris environment has a lot of old code and options that are no longer used.
  - Like the manual playing of the environment, drop all of that.
  - In the python code, we drop all of the scripts and settings and random nonsense that isnt used.
  - There are also soooo many checks and asserts, where we could just let it fail naturally. Especially on inputs. 
  - I want better thought out classes with clear responsibilites.
  - Better reuse of code, like the generator worker should be a thing.
  - Also the makefile is horrendous, I want clean simple commands that install and build and find optimal settings for the machine, thats it.
  - So we drop all the scripts essentially, and most of makefile is gone, and I don't want so much damn makefile boiler plate. Maybe we could move that logic into python scripts or something more suitable?
  - Also there are all sorts of strange functions like is_friendly_run_id. That is just used for testing?? But like obviously not needed? Like by construction it is a valid run id, so it should be valid.
  - Just like very very simple overall, we need to cut the bloat.
  - Also massive functions like apply_checkpoint_search_overrides. This should be like a class that handles settings or something. Things should be thought through explcitly and cleanly as classes with seperate responsibilites.
  - So screw all of ablations and inspection and scripts.
  - Also all the candidate gating logic with like storing params on the models should be dropped now in favor of the step based logic.
  - Also like cleaner way of parallelizing the game generation, so they all funnel in exactly like the main thread. I.e. I don't want patches and hacks to have the remote games be treated the same way as on machine games, they should all funnel in and be treated the same, in terms of logging and metrics and so on.
  - Get rid of all the candidate gating.
  - And like you really really dont need all that input validation like in NNValueWeightScheduleConfig. Like we are not taking in arbitrary inputs, we can just fail if something is set wrong the usual way! no need to verify all those things.
  - Also there are a lot of complicated metrics in the Rust that we could probably drop, like the rmse and variance along the trajectory and others. we should list out all of the different metrics, and then together decide which are actually useful and which are not.
  - Also things are duplicated like is build_aux_features used only in a script?
  - Also like the put everything on gpu idea is maybe not great, because then on smaller gpus, I am limited in terms of replay buffer size. So maybe we should just load in from cpu, is that much slower?
  - There should never be files that are like 1000+ lines long like the trainer.
  - Drop all backwards compatibility code. Like legacy actions and masks and so on. Like we should just have a fresh start, and no bloat from various formats etc.
  - Also think through if there would be a better format for saving and loading the replay buffers, that would make syncing faster for example or something.
  - Also tetris_bot is a not great name. Suggest alternatives.
  - Also the file organisation is not great in tetris_bot, with random top level files, and then just a big dump in tetris_bot/ml.