- **`PenaltyHJBSolver` docstrings no longer describe a working wrapper** (#2002). The module docstring, the class
  docstring and the `hjb_solvers` package listing still presented the retired wrapper as giving any HJB solver
  variational-inequality capability, wrote its penalty with the wrong sign (`+ (1/eps) * max(0, Psi - v)`, which
  lowers v where it should raise it), and called the lower obstacle `v >= Psi` optimal stopping, which under the
  library's cost convention is the upper obstacle `v <= Psi`. They now state the retirement, the sign, the
  convention and the `ObstacleConstraint` route. Docstrings only; behaviour is unchanged.
