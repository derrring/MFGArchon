- **`environment.yml` is deleted** (Issue #2167, step 6). `pyproject.toml` + `uv.lock` are the one
  dependency owner. #2164 reconciled the two manifests four days after #1687 had reconciled one of
  them alone; the fork is closed rather than patched again.
- **Swapping the BLAS implementation does not need a second manifest**, which was the objection that
  nearly kept the file. conda supplies a *substrate* — an interpreter and a BLAS-linked numpy/scipy,
  about thirty packages — and `pyproject.toml` still supplies the dependencies, because
  `uv pip install` treats a conda-installed numpy as satisfying and leaves it alone.
  `scripts/manage_environments.sh`'s `create-dev` now builds exactly that.
- **One qualification, measured after review said the original claim was too strong.** "Leaves it
  alone" holds for an ordinary install and **not** under `--reinstall-package`: measured,
  `uv pip install --reinstall-package numpy "numpy==2.4.6"` — the *same version* — replaces the
  conda-built wheel with the PyPI one, which is precisely losing the BLAS link the substrate exists
  to provide. An earlier version of this entry said the opposite, from a probe that passed
  `--reinstall-package` for a *dependent* package and never for numpy itself. Do not force-reinstall
  numpy or scipy in a substrate environment.
- **`check_manifests.py` loses its second direction with the file it read.** `DECLARED-BUT-MISSING`
  compared `pyproject.toml` against `environment.yml`; there is no second manifest to disagree with.
  `IMPORTED-BUT-UNDECLARED` — an unguarded module-level import no manifest declares, the #1687 shape
  — stays, and is what the gate step added in #2164 runs. `CONDA_TO_PYPI` went too: a conda-to-PyPI
  rename has nothing left to rename, and `_normalise` no longer applies one.
- `scripts/manage_environments.sh`'s `create-performance` reads `conda_performance.yml`, which was
  deleted in `dc47e5f7` and has been broken since. Not touched here — it is not code this change
  made unreachable, and what a "performance environment" should now mean is its own question.
  Recorded as #2178, along with the observation that it is the #2170 shape again: an artifact names
  something and nothing checks that the thing exists.
