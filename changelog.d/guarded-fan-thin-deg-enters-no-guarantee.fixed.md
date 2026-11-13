- **`GuardedFanParameters.thin_deg` no longer claims to enter the scheme's constant.** Its docstring said the
  smallest gap's sine bounds a control's sector coordinates. In a sector of aperture `theta` with rays `z0, z1`, a
  control `alpha = l0 z0 + l1 z1` with `l0, l1 >= 0` has `|alpha|^2 >= (l0|z0|)^2 + (l1|z1|)^2` when
  `theta <= 90` degrees and `l_k |z_k| sin(theta) <= |alpha|` above. So `l0|z0| + l1|z1|` is at most `2|alpha|`,
  or `2|alpha|/sin(theta)` with `theta` the fan's widest gap, which at a node given a widened cap can exceed
  `theta0_deg`. The monotonicity argument and the fan constant do not use `thin_deg`, a selection rule. Behaviour
  is unchanged.
