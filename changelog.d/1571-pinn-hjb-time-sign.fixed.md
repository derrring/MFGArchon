- **PINN HJB residual uses the backward-HJB time sign** (Issue #1571). Both `HJBPINNSolver` and
  `MFGPINNSolver` assembled `+du/dt + H - (sigma^2/2) u_xx = 0`, mirroring the FORWARD Fokker-Planck
  diffusion sign onto the time derivative of the BACKWARD HJB (terminal condition at t=T, t fed
  un-reversed). That fits the wrong PDE (e.g. H=const c, D=0 gave u(0)=g+cT instead of g-cT). The
  residual is now `-du/dt + H - (sigma^2/2) u_xx = 0`. A discriminating test pins the sign via the
  residual difference of `u=+c*t` vs `u=-c*t`. Off the published FDM/GFDM/particle paths.
