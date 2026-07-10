- **plugin_system migrated off the removed pkg_resources** (Issue #1540). `mfgarchon.core.plugin_system`
  did a module-level `import pkg_resources`, which raises `ModuleNotFoundError` on setuptools >= 81
  (pkg_resources removed) — undetected because nothing in the package imports the module (CI, incl. the
  compat matrix, never loaded it). It now uses the stdlib `importlib.metadata.entry_points(group=...)`
  (py3.10+ selectable API; same `.name`/`.load()` interface). Added an import + discover smoke test.
