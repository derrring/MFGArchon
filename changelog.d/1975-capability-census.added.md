`scripts/capability_census.py` and its ratchet `tests/unit/test_capability_census.py`: what each
class **declares**, and what the FP paths actually **do** at a wall.

Both lanes answer a question the 2026-08-13 design census could not. All four of its lanes look for
reality falling *short* of a claim, and it found 77 over-claims; nobody counted the other
direction. The other direction is what made #1975 wrong twice — `FPFEMSolver` implements a general
Robin wall and declares nothing, and the default FDM scheme imposes `J.n = 0` structurally with no
branch naming it. Over-claiming makes a user's code fail loudly; under-claiming makes a maintainer
"fix" something that works.

Lane 1 discovers its population with `walk_packages` + `issubclass` — a predicate independent of
every declaration it audits — so "declares nothing" is a recorded row rather than an absence. Own
declarations are separated from inherited ones by walking the MRO: `honors_inhomogeneous_neumann`
defaults to `True` on `BaseMFGSolver`, so 19 solvers claim to honour an inhomogeneous Neumann flux
by a default nobody chose.

Lane 2 is the oracle this area was missing: mass conservation at a wall with wall-normal drift,
a law of the equation computed without reference to any scheme. Five FP paths conserve to machine
precision — several imposing `J.n = 0` structurally, one by Skorokhod reflection. The zero-drift
control fired immediately: `FPSLJacobianSolver` gains mass at `O(h)` with no drift at all
(+0.1396 / +0.0689 / +0.0343 % at Nx = 41 / 81 / 161), so its drifted figure says nothing about its
wall — the quantified reason its deprecation notes never gave. Six paths are `NOT_MEASURED`, which
is recorded as a finding rather than a pass. (#1975, #1977)
