- **Pin mypy `<2.2` in the dev extras.** mypy 2.2.0 false-positives on valid `list[str]` slices
  (e.g. `config/omegaconf_manager.py` `keys[:i+1]`) via a typeshed `slice`-overload regression that
  2.1.0 does not have, reddening the blocking config type-gate on every PR. Narrow upper bound until
  mypy 2.2.1 fixes it (then lift to `<2.3`).
