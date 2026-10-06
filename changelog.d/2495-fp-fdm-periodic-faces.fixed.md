FP-FDM reads periodicity from a boundary condition's faces, not from how it is spelt. One PERIODIC
segment per face, or an empty segment list over a periodic `default_bc`, solved as no-flux on every
face, bit for bit and with no error: the periodic test was `bc.is_uniform and bc.type ==
"periodic"`, and every wall node of a non-uniform BC went to the no-flux handler. On the implicit
path -- `FPFDMSolver.solve_fp_system`, and the coupled `FDM_UPWIND` solve -- both spellings
now solve bit-identically to `periodic_bc`, a BC handed to the solver without a bound dimension
included. On a fixture of the issue's shape (coupled 1-D `FDM_UPWIND`) the gap was 5.2e-02 in U and
3.2 in M on main. One face loop, `conditions.periodic_faces`, decides it: `periodic_on_every_face`
reads it for `periodic_axis_span`, the assembly's wall routing and the linearised operator, and the
refusal below reads it too.

A BC periodic on some faces only -- a channel, or an axis periodic on one face -- was solved the same
way, as no-flux everywhere. FP-FDM wraps every axis or none, so it now refuses such a BC, at
construction and where the wall handlers are dispatched; per-axis FP-FDM is #2505. It also refuses a
periodic segment the face reader cannot place (`bc_utils.refuse_unplaceable_segments`, now shared
with the semi-Lagrangian pair), in a mix of operations or with no `default_bc`: one with no
`boundary` -- placed by `normal_direction`, say -- reads as periodic on every face, and would be
wrapped on faces it does not reach, where HJB-FDM walls them over a default or finds no BC and raises
without one. The semi-Lagrangian pair now refuses that no-default shape too. A BC that declares
periodicity but shows it on no face -- a periodic default that segments naming every face override --
is still solved as no-flux; an unrestricted catch-all segment beside such a default is refused with
the unplaceable ones, by #2467's rule.

Not changed: the callable-drift, tensor-diffusion and strict-adjoint (`solve_fp_step_adjoint_mode`)
paths build their operators from the BC's uniform type, and refuse an all-periodic BC spelt per face
on main and here alike; that is #2505's too.
