`MFGFormatter` is now the only owner of the log line format, and is exported from
`mfgarchon.utils.mfg_logging` (#2058). Three rival copies of
`"%(asctime)s - %(name)s - %(levelname)s - %(message)s"` are gone: two hand-rolled formatters in
`LoggingHook` (`hooks/visualization.py`) and `setup_workflow_logging` (`workflow/common.py`) now use
`MFGFormatter()`, and the third — an inert `logging.format` key in `config/configs/experiment.yaml`
that nothing reads — is deleted. No copy of that literal remains under `mfgarchon/`.

Each copy departed from the owner in two ways, of which **one** matters. The missing `datefmt` makes
`asctime` carry milliseconds, so `LogAnalyzer`'s timestamp field never matches and the line is
dropped before its level is examined. The missing `-8s` padding costs nothing, because #2056 widened
the reader's level group to accept the single space an unpadded levelname leaves. Measured 2×2
against the current reader: `datefmt` alone gives 5/5 at every level with or without padding, no
`datefmt` gives 0/5 either way.

**Latent, not live.** `setup_workflow_logging` builds no handler at all for production callers —
`get_logger` always attaches a StreamHandler, so its `if not logger.handlers:` guard is False;
measured with `log_file` set and `console=True`, it returns one StreamHandler carrying the correct
format and writes no file. `LoggingHook` does install its formatter, but only when a caller passes
`log_file`, and outside this change's own test nothing does. So no unreadable log is being written
today; a rival string was standing ready to write one.

Fixed as a single-source change rather than by adding `datefmt=` to each copy: one owner, every
rival deleted in the same change, which is what stops the drift recurring.

**This does not make workflow logs readable**, and #2062 tracks the reason: workflow loggers are
named `mfg_workflow.<uuid>`, and `LogAnalyzer`'s logger-name group is `([^-]+?)`, which cannot match
a hyphen. Measured through the real reader: `MFGSolver` and `mfg_workflow_manager` parse, while
`mfg_workflow.48151c9d-a6d5-...` and a user-chosen `my-experiment` both yield zero entries. Any
logger name containing a hyphen is silently dropped.
