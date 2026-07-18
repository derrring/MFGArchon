"""DEPRECATED: Use mfgarchon.utils.iteration.schedules instead.

This module is a compatibility shim. Will be removed in v0.22.0.
"""

import warnings

warnings.warn(
    "Importing from mfgarchon.alg.iterative.schedules is deprecated. "
    "Use mfgarchon.utils.iteration.schedules instead. "
    "Will be removed in v0.22.0.",
    DeprecationWarning,
    stacklevel=2,
)

from mfgarchon.utils.iteration.schedules import *  # noqa: E402, F403
