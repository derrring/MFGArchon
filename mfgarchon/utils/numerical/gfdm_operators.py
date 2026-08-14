"""
Deprecated: GFDMOperator has moved to _compat/gfdm_operators.py.

Use TaylorOperator from gfdm_strategies.py instead.

This file preserves the import path for backward compatibility:
    from mfgarchon.utils.numerical.gfdm_operators import GFDMOperator
"""

import warnings

warnings.warn(
    "mfgarchon.utils.numerical.gfdm_operators is deprecated since v0.17.15. "
    "Use mfgarchon.alg.numerical.gfdm_components.gfdm_strategies.TaylorOperator instead. "
    "GFDMOperator has been moved to _compat and will be removed in v0.25.0.",
    DeprecationWarning,
    stacklevel=2,
)

from mfgarchon.utils.numerical._compat.gfdm_operators import GFDMOperator  # noqa: E402

__all__ = ["GFDMOperator"]
