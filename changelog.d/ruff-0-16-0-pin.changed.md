- **The ruff version is pinned in CI, and markdown is exempt from formatting** (ruff 0.16.0).
  `ci.yml`'s Quick Validation job ran a bare `pip install ruff`, so it silently drifted off the
  version pinned in `.pre-commit-config.yaml`; when 0.16.0 shipped fenced-code-block formatting for
  markdown, the job went red on four README files nobody had touched, on every open PR at once. The
  job now reads the pin from `.pre-commit-config.yaml`, fails loudly if that pin cannot be read, and
  that file is now a CI trigger path -- otherwise a pin change merges without ever running the
  version it selects. `scripts/local_ci.sh` warns when the ruff on PATH disagrees with the pin,
  since `pyproject.toml` and `environment.yml` specify a floor (`ruff>=0.6.0`) rather than a pin.
  `[tool.ruff.format] exclude = ["*.md"]` keeps documentation examples laid out for reading.
