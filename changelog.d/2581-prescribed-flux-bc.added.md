- **A prescribed mass flux through a Fokker-Planck wall: `BCType.FLUX` and `flux_bc(q)`** (#2581). `J.n = q`, with
  `J = alpha*m - D*grad(m)` the total flux and `q` mass per unit time per unit boundary measure, outward positive;
  `q = 0` is NO_FLUX, and `q` may be a constant or a callable `q(t, x)`, read at the level being solved. It is the FP
  solver's own BC: `FPFDMSolver` (its implicit step) and `FPFEMSolver` honour it, a FLUX on the problem's shared BC
  is refused at `MFGProblem.get_boundary_conditions`, and every other solver refuses it by name. The mass changes by
  exactly `-int q dS` per unit time on a step that conserves mass under NO_FLUX, with no Dirichlet face, where the
  positivity clip leaves the step alone. A step is refused rather than clipped where the clip would fabricate mass the
  flux decides (`refuse_what_a_flux_takes_negative`): a node the step without the flux leaves non-negative goes
  negative, or the step without it passes the clip gate and the flux takes it past. A step whose own clip is past the
  gate is the scheme's, flux included: FP-FDM's guard refuses it, FP-FEM clips it with its warning. The FDM ghost layer
  (`pad_array_with_ghosts` and the operators behind it), the calculator route, the geometric fold and the legacy ghost
  values refuse a FLUX by name rather than write another condition.
