One solve now commits to one interpolant. `interpolation_method` named four different
interpolants across the semi-Lagrangian family, and which one ran was decided **per timestep**
by the CFL number rather than by configuration: one `cubic` solve at `Nx=41, Nt=8` constructed
`CubicSpline(not-a-knot)` 7 times and `PchipInterpolator` 81 times. The two disagree by ~1.9e-2
on coarse grids and differ in monotonicity -- the property Issue #583/#1033 replaced
`CubicSpline` for after the Towel-on-Beach blow-up. Three of the four dispatch sites had taken
that fix; the default batch path had not, so on that path this is a deliberate numerical change,
not a byte-identical refactor.

`sl_backend()` in `hjb_sl_interpolation` is now the single owner of method-to-backend, taking the
monotone requirement as an explicit policy argument, and the four private ladders are deleted.
Only the backend choice changes: both interpolants the batch path previously built extrapolated
out-of-domain feet, and it still does, so the out-of-bounds policy is untouched.

A monotone scheme overriding the declared method is now disclosed rather than applied silently.
`diffusion_method="stochastic"` warned only for `cubic`/`quintic`, via a hardcoded pair that was a
third restatement of the same policy; `nearest` and `slinear` are honoured at nD, are equally
non-monotone, and were remapped to linear with no warning at all (measured spread on one 7x7
profile: `nearest` 0.0688 against `linear` 0.2732).
