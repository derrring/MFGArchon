`scripts/capability_census.py` and its ratchet `tests/unit/test_capability_census.py`: what each
class **declares**, and which wall each FP path actually **imposes**.

Both lanes answer a question the 2026-08-13 design census could not. All four of its lanes look for
reality falling *short* of a claim, and it found 77 over-claims; nobody counted the other
direction, and the other direction made #1975 wrong twice. Over-claiming makes a user's code fail
loudly; under-claiming makes a maintainer "fix" something that works.

Lane 1 discovers its population with `walk_packages` + `issubclass` — a predicate independent of
every declaration it audits — so "declares nothing" is a recorded row rather than an absence. Own
declarations are separated from inherited ones by walking the MRO: `honors_inhomogeneous_neumann`
defaults to `True` on `BaseMFGSolver`, so 19 solvers claim to honour an inhomogeneous Neumann flux
by a default nobody chose.

Lane 2's verdict is the wall ratio `d_n m / ((v_n/D) m_wall)` and its **trend under refinement**,
not mass conservation. Mass conservation is neither sufficient (streamline diffusion conserves to
1e-12 while the ratio collapses 0.967 -> 0.414) nor necessary (`FPSLJacobianSolver` is the
Lagrangian form, non-conservative by construction, deprecated for adjoint inconsistency rather than
for mass). Mass drift is reported as a form property beside the verdict, never as it.

The instrument was rebuilt after eight defects, each of which produced a confident verdict rather
than a failure — four found by independently measuring the paths it could not construct, four more
only by re-running it against a path whose answer was already known. The sharpest: it passed a
potential to the three solvers whose declared `_drift_convention` is `VELOCITY`, so the
wall-normal drift vanished at the wall the mass reached and the discriminating property was absent
while a verdict printed anyway. (#1975, #1977)
