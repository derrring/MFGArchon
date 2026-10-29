- **A prescribed mass flux through a Fokker-Planck wall: `BCType.FLUX` and `flux_bc(q)`** (#2581). `J.n = q`, with
  `J = alpha*m - D*grad(m)` the total flux and `q` mass per unit time per unit boundary measure, outward positive;
  `q = 0` is NO_FLUX, and `q` may be a constant or a callable `q(t, x)`, read at the level being solved. It is the FP
  solver's own BC: `FPFDMSolver` (its implicit step) and `FPFEMSolver` honour it, a FLUX on the problem's shared BC
  is refused at `MFGProblem.get_boundary_conditions`, and every other solver refuses it by name. The mass changes by
  exactly `-int q dS` per unit time; a step whose flux part would take the density negative is refused, never
  clipped. The FDM ghost layer (`pad_array_with_ghosts` and the operators behind it), the calculator route, the
  geometric fold and the legacy ghost values refuse a FLUX by name rather than write another condition.
