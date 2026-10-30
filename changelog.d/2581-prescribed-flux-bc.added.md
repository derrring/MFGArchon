- **A prescribed mass flux through a Fokker-Planck wall: `BCType.FLUX` and `flux_bc(q)`** (#2581). `J.n = q`, with
  `J = alpha*m - D*grad(m)` the total flux and `q` mass per unit time per unit boundary measure, outward positive;
  `q = 0` is NO_FLUX, and `q` may be a constant or a callable `q(t, x)`, read at the level being solved. It is the FP
  solver's own BC: `FPFDMSolver` (its implicit step) and `FPFEMSolver` honour it, a FLUX on the problem's shared BC
  is refused at `MFGProblem.get_boundary_conditions`, and every other solver refuses it by name. Each step changes the
  mass by exactly `-dt int q(t_{k+1}) dS` on the scheme's own face quadrature, on a step that conserves mass under
  NO_FLUX, with no Dirichlet face, where the positivity clip leaves the step alone. A step is refused rather than
  clipped past the gate where the clip would fabricate mass the flux decides (`refuse_what_a_flux_takes_negative`):
  the negatives the flux makes where the step without it is non-negative would fabricate more than the gate allows, or
  the step without it passes the clip gate and either the flux deepens its negatives past it or the step with the
  flux fails it. Any other step whose own clip is past the gate
  is the scheme's, flux included: FP-FDM's guard refuses it where clipping would fabricate more than the gate allows,
  FP-FEM clips it. The FDM ghost layer (`pad_array_with_ghosts` and the operators behind it), the calculator route, the
  geometric fold and the legacy ghost values refuse a FLUX by name rather than write another condition.
