- **A prescribed mass flux through a Fokker-Planck wall: `BCType.FLUX` and `flux_bc(q)`** (#2581). `J.n = q`, with
  `J = alpha*m - D*grad(m)` the total flux and `q` mass per unit time per unit boundary measure, outward positive;
  `q = 0` is NO_FLUX, and `q` may be a constant or a callable `q(t, x)`, read at the level being solved. It is the FP
  solver's own BC: `FPFDMSolver` (its implicit step) and `FPFEMSolver` honour it, a FLUX on the problem's shared BC
  is refused at `MFGProblem.get_boundary_conditions`, and every other solver refuses it by name. Each step changes the
  mass by exactly `-dt int q(t_{k+1}) dS` on the scheme's own face quadrature, on a step that conserves mass under
  NO_FLUX, with no Dirichlet face, and on FP-FDM where its positivity clip leaves the step alone. A step is refused
  where the flux takes the density negative past the gate (`refuse_what_a_flux_takes_negative`): the negatives the flux
  makes where the step without it is non-negative are more than the gate allows, or the step without it is within the
  gate and either the flux deepens its negatives past it or the step with the flux is past it. Any other step whose own
  negatives are past the gate is the scheme's, flux included: FP-FDM's guard refuses it, and FP-FEM keeps its negative
  values and reports them. The FDM ghost layer (`pad_array_with_ghosts` and the operators behind it), the calculator route, the
  geometric fold and the legacy ghost values refuse a FLUX by name rather than write another condition.
