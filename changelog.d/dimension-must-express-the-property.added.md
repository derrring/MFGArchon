`AGENTS.md` gains a testing rule: **the dimension must be able to express the property under test.**

Not "prefer 2D". The point is that a test in a dimension that cannot express the property is not a
weak test — it cannot fail at all, and it reads exactly like a passing one. A 1D wall's normal is
always the coordinate axis and there is no tangential component, so a scheme that mishandles the
tangential part passes 1D by construction.

The discriminator is measurable rather than arguable: write a mutation that breaks the property and
run it in 1D. Measured on the FP wall study (#1728/#2006), three tangential mutants — dropping the
tangential advection at wall rows, flipping its sign, reading the potential along the wrong axis —
all give `max|diff| = 0.000e+00` in 1D and two of three separate in 2D. `max|diff| = 0` means the
test must go up a dimension.

The burden runs both ways: 2D costs real time (the same FP MMS study is 0.1 s at `d=1` against 1.5 s
at `d=2`), so a 2D test whose 1D reduction separates the same mutants owes its runtime an
explanation, and a 1D test of a directional property owes a mutant showing it can fail.
