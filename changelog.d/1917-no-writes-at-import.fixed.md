Importing `mfgarchon` or `mfgarchon.workflow` no longer writes to the caller's working
directory.

`import mfgarchon.workflow` ran an initialiser that built a workflow manager anchored to
`Path.cwd()`, created `.mfg_workflows/`, and persisted an example workflow as a side effect of
reading the module; `import mfgarchon` separately created `performance_data/` from a
module-level `PerformanceMonitor()` whose constructor called `mkdir`. Constructing a `Workflow`
or a `WorkflowManager` also created directories. 21,498 empty directories had accumulated in the
development tree, and on a read-only working directory the import raised `PermissionError`.

`Workflow`, `WorkflowManager` and `PerformanceMonitor` now compute their paths at construction
and create the directory at the point of an actual write. `initialize_default_workspace()` and
`_create_example_workflows()` are removed with the initialiser that was their only caller.

The same constructor-time pattern survives in `Experiment`, `ExperimentTracker` and
`SweepConfiguration`, which still create directories under the current working directory when
they are constructed; those are tracked in #1929. (#1917, #1674)
