Document which tree `./scripts/local_ci.sh` reads when it runs from a `git worktree`, because the
answer differs between the suite step and the ratchet steps.

Measured in a throwaway worktree with a control assertion that fired: the suite step
(`PYTHONSAFEPATH=1 "$PY" -P -m pytest tests/ -n auto`) reads the worktree, because pytest inserts
the rootdir at `sys.path[0]` -- `tests/__init__.py` exists -- and setuptools' editable install only
appends its finder to `sys.meta_path`, so `PathFinder` resolves first. Every `"$PY" scripts/*.py`
ratchet reads the main checkout instead: there `sys.path[0]` is the worktree's `scripts/`, which
holds no `mfgarchon`, so the editable finder answers with the original tree.

Seven files under `scripts/` import the package, `check_doc_api.py` among them, and that one runs
under `--fast`. A lane's `GATE GREEN` is therefore half its own tree and half whichever branch the
main checkout is on, with nothing in the output naming which. The fix is per-worktree isolation --
its own environment, or `PYTHONPATH` for the ratchet steps -- not `MFG_PYTHON`, which selects the
interpreter and not the tree.

Documentation only; no behaviour changes.
