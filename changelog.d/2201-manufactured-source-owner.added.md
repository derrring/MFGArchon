`mfgarchon.utils.manufactured` — one owner for assembling MFG manufactured-solution (MMS) source
terms from an exact `(u*, m*)` pair, for coupled HJB–FP fixtures on a separable Hamiltonian.

Every convention the assembly needs is taken from its existing owner rather than re-derived: `H`
from `HamiltonianBase.evaluate_H`, the FP drift `alpha*` from the Hamiltonian's own
`optimal_control` (so the transport term follows the optimization sense instead of hardcoding the
MINIMIZE relation), and `sigma -> D` from `diffusion_from_volatility`, including its `kind`
argument and its refusal to guess a `(d, d)` tensor from a spatially varying field. The diffusion
term is written `tr(D . Hess)` rather than `(sigma^2/2) Lap`, so an anisotropic `Sigma` can express
the cross-derivative term (#2198), which a Laplacian-shaped source cannot.

`check_pair` audits a pair's eight analytic derivative callables against a finite difference of `u`
and `m`. This is the pair's only non-circular check — a residual built from the pair's own
derivatives is an algebraic identity that returns zero under a deliberately sign-flipped convention
— and it is the only check that sees a wrong cross-derivative, which an isotropic `Sigma`
multiplies by exactly zero.
