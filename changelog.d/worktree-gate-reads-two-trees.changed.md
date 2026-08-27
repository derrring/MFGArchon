Document which tree `./scripts/local_ci.sh` reads when it runs from a `git worktree`: most steps
read the worktree, three read the main checkout.

The suite step reads the worktree, because pytest puts the tree root at `sys.path[0]` -- there is a
`tests/__init__.py` -- and setuptools' editable install only appends its finder to `sys.meta_path`,
so `PathFinder` answers first. Twelve of the fifteen `scripts/*.py` steps also read the worktree:
they never import the package, they walk `--path .` from the gate's own `cd`. The three that do
import -- `check_internal_deprecation.py --self-test` under `--fast`, and `capability_matrix.py`
twice in the full gate -- resolve through the editable finder, which is hard-wired to the original
checkout. Measured with a blocking meta-path finder over every step, its control firing on the
importer.

A lane must set both `PYTHONPATH=<worktree>`, which fixes the tree, and `MFG_PYTHON` pointed at the
gate's own environment, which fixes the interpreter: the gate's candidate search tries PATH first,
so an activated virtualenv is selected and its older pytest and ruff turn a documentation diff into
a red gate.

Documentation only; no behaviour changes.
