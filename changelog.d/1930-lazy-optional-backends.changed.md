`import mfgarchon` no longer imports PyTorch.

torch is optional — declared in the `[all]` extra, not in `dependencies` — but two eager sites
imported it for anyone who touched the package: `backends/__init__.py` registered the torch and
jax backends at import, and `utils/acceleration/__init__.py` re-exported `torch_utils`, which
imports torch (inside a `try/except ImportError`, so that module loads fine without it — but the
import still runs when torch is installed).

Both are now on demand. `create_backend("torch")` already carried the import-and-register path;
eager registration was what made it unreachable. `mfgarchon.utils.acceleration` resolves its
torch names through a module `__getattr__`, and `TORCH_UTILS_AVAILABLE` is resolved the same way
— so it still answers the question it always answered ("did `import torch` succeed"), computed on
first read rather than at import.

Measured: `import mfgarchon` 4.12s → 3.27s, with torch absent from `sys.modules`. NumPy stays
eager: it is a hard dependency and costs 0.07s.

One observable consequence: `get_backend_info()["registered_backends"]` now reports
`["numpy"]` at import rather than every installed backend. Asking for a backend still
registers it, so the list grows as backends are used. (#1930)
