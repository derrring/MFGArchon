from __future__ import annotations

# mfgarchon/utils/aux_func.py
import numpy as np

# Original definition was -np.minimum(x,0), which is equivalent to np.maximum(-x,0)
# For consistency with ppart(x) = max(x,0), npart(x) = max(-x,0)
# If p_fwd = (U[i+1]-U[i])/Dx, then npart(p_fwd) is for flow to the right if p_fwd < 0.
# If p_bwd = (U[i]-U[i-1])/Dx, then ppart(p_bwd) is for flow to the left if p_bwd > 0.


def ppart(x: float | np.ndarray) -> float | np.ndarray:
    """
    Positive part of x.
    Works element-wise for NumPy arrays.
    """
    return np.maximum(x, 0.0)


def npart(x: float | np.ndarray) -> float | np.ndarray:
    """
    Negative part of x, defined as max(0, -x).
    This means (npart(x))^2 is equivalent to (x^-)^2 where x^- = min(0, x).
    Works element-wise for NumPy arrays.
    """
    return np.maximum(-x, 0.0)
