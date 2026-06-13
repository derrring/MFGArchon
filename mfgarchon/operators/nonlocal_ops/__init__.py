"""
Deprecated alias for :mod:`mfgarchon.operators.integro_diff`.

``operators.nonlocal_ops`` has been renamed to ``operators.integro_diff`` to
separate non-local *PDE structure* (Lévy integro-differential operators, graphon
coupling) from the *game-coupling* non-local operators in
``operators/interaction/``. See issue #1024.

This module re-exports the public API and aliases the submodules so existing
import paths keep working; importing it emits a ``DeprecationWarning``. The shim
will be removed in v0.22.0.
"""

import sys
import warnings

from mfgarchon.operators import integro_diff as _integro_diff
from mfgarchon.operators.integro_diff import (  # noqa: F401
    CompoundPoissonJumps,
    GaussianJumps,
    LevyIntegroDiffOperator,
    LevyMeasure,
)
from mfgarchon.operators.integro_diff import (
    graphon_coupling as _graphon_coupling,
)
from mfgarchon.operators.integro_diff import (
    graphon_kernels as _graphon_kernels,
)
from mfgarchon.operators.integro_diff import (
    levy_integro_diff as _levy_integro_diff,
)
from mfgarchon.operators.integro_diff import (
    levy_measures as _levy_measures,
)

warnings.warn(
    "mfgarchon.operators.nonlocal_ops is renamed to "
    "mfgarchon.operators.integro_diff; will be removed in a future release "
    "(v0.22.0). Update imports to mfgarchon.operators.integro_diff.",
    DeprecationWarning,
    stacklevel=2,
)

# Alias submodules so old dotted paths (e.g.
# ``mfgarchon.operators.nonlocal_ops.levy_integro_diff``) resolve to the renamed
# modules without re-importing or duplicating module objects.
sys.modules[__name__ + ".levy_integro_diff"] = _levy_integro_diff
sys.modules[__name__ + ".levy_measures"] = _levy_measures
sys.modules[__name__ + ".graphon_coupling"] = _graphon_coupling
sys.modules[__name__ + ".graphon_kernels"] = _graphon_kernels

__all__ = _integro_diff.__all__
