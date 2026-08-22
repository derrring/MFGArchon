`MFGFormatter` is now the only owner of the log line format (#2058). Two hand-rolled copies of
`"%(asctime)s - %(name)s - %(levelname)s - %(message)s"` — in `LoggingHook`
(`hooks/visualization.py`) and `setup_workflow_logging` (`workflow/common.py`) — are replaced by
`MFGFormatter()`, and no rival format string remains in the package.

Each copy departed from the owner in two independent ways, either of which alone defeats
`LogAnalyzer`: no `datefmt`, so `asctime` carries milliseconds and the timestamp field never
matches, and no `-8s` padding, so the level field never matches. Measured through `LoggingHook`
before the change: **0 entries at every level**, failing at the timestamp before the level was
examined.

**This was latent, not live**, and the issue overstated it in both directions before it was
measured. `setup_workflow_logging` builds no handler at all for production callers — `get_logger`
always attaches a StreamHandler, so its `if not logger.handlers:` guard is False and neither the
FileHandler nor the console handler is constructed; its copy was never installed. `LoggingHook`
does install its formatter, but only when a caller passes `log_file`, and the only site that does
is inside a docstring Example. So nothing currently writes an unreadable log — a rival string was
simply standing ready to.

Fixed as a single-source change rather than by adding `datefmt=` to each copy: one owner, both
rivals deleted in the same change, which is what stops the drift recurring rather than repairing
this instance of it.
