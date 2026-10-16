- **The mypy scope baseline records alg 649.** #2706 and #2711 each removed one type error in `alg` and each recorded
  alg 651 -> 650 against their common base, so their merges kept one decrement: main had 649 errors against a recorded
  650, and the ratchet (`scripts/check_mypy_scope.py`) failed every pre-push gate. No code changes.
