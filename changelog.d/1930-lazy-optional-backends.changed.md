`import mfgarchon` no longer imports PyTorch.

torch is optional — declared in the `[all]` extra, not in `dependencies` — but two eager sites
imported it for anyone who touched the package: `backends/__init__.py` registered the torch and
jax backends at import, and `utils/acceleration/__init__.py` re-exported `torch_utils`, whose
first statement is `import torch`.

Both are now on demand. `create_backend("torch")` already carried the import-and-register path;
eager registration was what made it unreachable. `mfgarchon.utils.acceleration` resolves its
torch names through a module `__getattr__`, so `TORCH_UTILS_AVAILABLE` still answers correctly —
it uses `importlib.util.find_spec`, which does not import anything.

Measured: `import mfgarchon` 4.12s → 3.27s, with torch absent from `sys.modules`. NumPy stays
eager: it is a hard dependency and costs 0.07s. (#1930)
