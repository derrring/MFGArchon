- **`MeanFieldActorCritic` now fails loud when the observation carries no population channel**
  (Issue #1568, extends #1508). `_extract_population` silently zero-filled the mean-field state when
  the observation lacked a `local_density`/`population` key (dict) or a tail slice past `state_dim`
  (array) — training on an identically-zero mean field and returning a policy the user trusts as
  MFG-coupled while the coupling that DEFINES the game was never present. #1508 fixed exactly this
  for DDPG/TD3/SAC (env-side `get_population_state` guard) but did not cover ActorCritic, which reads
  the population from the observation rather than querying the env. It now raises `AttributeError`
  citing #1508/#1568, with a hint that adapts to whether the env exposes `get_population_state()`.
  Note: the raise keys on "observation has no population channel" (not on "env lacks
  get_population_state"), so it also catches the shipped `crowd_navigation_env` pairing — that env
  *does* expose `get_population_state()` yet its observation omits the population, which the issue's
  literal env-only condition would have missed.
