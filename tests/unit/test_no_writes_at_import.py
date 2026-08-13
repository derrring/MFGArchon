"""Importing this package writes nothing to the caller's working directory.

#1917 measured 20,322 UUID-named directories accumulated in this repository in two weeks, a
sample of 500 of them empty. `import mfgarchon.workflow` ran an initialiser that built a manager
anchored to `Path.cwd()`, created `.mfg_workflows/`, and persisted an example workflow -- with a
step -- as a side effect of reading the module. `import mfgarchon` separately created
`performance_data/`, because `global_performance_monitor = PerformanceMonitor()` runs at module
level and that constructor called `mkdir`. Neither needed a user to do anything.

The reason this file exists rather than just the fixes: #1674 filed the `performance_data/` half
BEFORE those 20,322 directories were created, and it stayed open while they accumulated. A fix
with no test is what that looks like from the outside. What is asserted here is the whole
directory listing, not the two names that are known to have gone wrong -- a filter would pass over
the next site rather than catch it.

Each test runs the import in a subprocess with a fresh interpreter, because by the time this file
is collected the package is already imported and the side effect, if any, has already happened in
the pytest process's cwd.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]

# Entry points a user or a tool actually reaches for. `mfgarchon.workflow` is the site #1917
# measured; the others are here because the defect was never specific to it -- any module-level
# singleton whose constructor writes has the same shape.
IMPORTS = [
    "mfgarchon",
    "mfgarchon.workflow",
    "mfgarchon.utils.performance",
    "mfgarchon.geometry",
    "mfgarchon.backends",
]


def _listing_after(code: str, cwd: Path) -> tuple[set[str], subprocess.CompletedProcess]:
    """Run `code` with `cwd` as the working directory and report what appeared there.

    HOME points somewhere else on purpose. Pointing it at `cwd` was the first version, and it
    turned matplotlib's ordinary `~/.matplotlib` cache into a reported defect in five tests --
    a harness artefact that reads exactly like the bug under test. Writing to HOME is legitimate;
    writing to the directory you were launched from is not, and only the second is measured here.
    """
    import tempfile

    with tempfile.TemporaryDirectory() as home:
        proc = subprocess.run(
            [sys.executable, "-c", code],
            cwd=cwd,
            env={
                "PATH": "/usr/bin:/bin",
                "PYTHONPATH": str(REPO),
                "HOME": home,
                "MPLCONFIGDIR": home,
                "XDG_CACHE_HOME": home,
            },
            capture_output=True,
            text=True,
            timeout=300,
        )
    return {p.name for p in cwd.iterdir()}, proc


@pytest.mark.parametrize("module", IMPORTS)
def test_importing_writes_nothing_to_the_working_directory(module, tmp_path):
    """The whole listing, not a denylist of the two names already known to be wrong."""
    before = {p.name for p in tmp_path.iterdir()}
    assert before == set(), "tmp_path was not empty; the measurement below would mean nothing"

    after, proc = _listing_after(f"import {module}", tmp_path)
    assert proc.returncode == 0, f"import {module} failed, so nothing was measured:\n{proc.stderr[-1500:]}"
    assert after == set(), (
        f"import {module} created {sorted(after)} in the caller's working directory. "
        f"A library may not write where it was launched from; anchor to an explicit path or "
        f"create the directory at the point of an actual write."
    )


def test_naming_a_workflow_writes_nothing():
    """Constructing a `Workflow` used to `mkdir` and open a log file inside it, so the test
    suite's own `Workflow(name="test")` calls were part of the 20,322."""
    code = "from mfgarchon.workflow import Workflow\nw = Workflow(name='probe')\nassert w.name == 'probe'\n"
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        cwd = Path(d)
        after, proc = _listing_after(code, cwd)
    assert proc.returncode == 0, f"construction failed, so nothing was measured:\n{proc.stderr[-1500:]}"
    assert after == set(), f"naming a workflow created {sorted(after)}"


def test_the_probe_would_notice_a_write():
    """Positive control. Every assertion above is that a set is empty, and an empty set is what a
    broken probe returns too -- a subprocess that never ran, a cwd that is not the one inspected,
    an exception swallowed into returncode 0. This writes on purpose and must be caught.
    """
    code = "import pathlib\npathlib.Path('sentinel_dir').mkdir()\n"
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        cwd = Path(d)
        after, proc = _listing_after(code, cwd)
    assert proc.returncode == 0, proc.stderr[-800:]
    assert after == {"sentinel_dir"}, (
        f"the probe reported {sorted(after)} for a program that certainly wrote one directory; "
        f"it is not measuring the directory it claims to measure"
    )


def test_saving_a_result_does_create_the_directory():
    """The other direction: deferring the write must not mean never writing.

    A fix that simply stopped persisting would pass every assertion above, so assert that the
    directory appears at the point it is supposed to -- when a workflow is actually saved.
    """
    code = (
        "from pathlib import Path\n"
        "from mfgarchon.workflow import WorkflowManager\n"
        "m = WorkflowManager(workspace_path=Path('ws'))\n"
        "assert not Path('ws').exists(), 'the manager created its workspace at construction'\n"
        "w = m.create_workflow('real', 'persisted on purpose')\n"
        "meta = Path('ws') / f'workflow_{w.id}' / 'metadata.json'\n"
        "assert meta.exists(), f'create_workflow did not persist metadata: {sorted(Path().rglob(\"*\"))}'\n"
    )
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        cwd = Path(d)
        after, proc = _listing_after(code, cwd)
    assert proc.returncode == 0, f"the persistence path is broken:\n{proc.stderr[-1500:]}"
    assert after == {"ws"}, f"expected only the workspace the caller named, got {sorted(after)}"
