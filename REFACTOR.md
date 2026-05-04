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