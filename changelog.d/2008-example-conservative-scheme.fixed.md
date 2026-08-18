`examples/basic/three_mode_api_demo.py` selected the non-conservative `gradient_upwind` advection
scheme in its expert-mode demonstration and lost 98.17% of its probability mass (#2008).

Measured on the example's own problem: `gradient_upwind` runs `1.000000 -> 0.018302`, while
`divergence_upwind` holds `1.000000` to `-0.00%`. The scheme is documented non-conservative at
no-flux walls (#1075) and the constructor warns about it, but `pytest.ini` is `testpaths = tests`,
so nothing in any gate runs `examples/`.

`divergence_centered` is not an alternative here: it raises at timestep 3 because the density goes
negative at cell-Peclet > 2, which the mass-fabrication gate correctly refuses to clip.

`max_iterations` moves 20 -> 60. The old value did not converge either (`converged=False`,
`max_error=7.184e-06`); the conservative solve reaches tolerance at iteration 49.

What this does not fix, and #2008 keeps open: `err_M` measures an increment, so it read `9.27e-07`
-- three orders under the `1e-6` tolerance -- while the mass drained, and no diagnostic in the
coupled loop is a function of total mass.
