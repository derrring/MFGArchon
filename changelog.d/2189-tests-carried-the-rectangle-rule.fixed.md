Two integration tests built their expectation with the measure the library had stopped using, and
went red on `main` after #2145 and #1887.

`test_jacobian_transpose_converges` reconstructed the initial density as `m0 /= np.sum(m0) * dx`
and compared at `atol=1e-15`. It now asserts with `assert_array_equal` that `result.M[0]` comes
back exactly as handed in, with no normalisation -- which is what #1887 promises, is bitwise true
as measured, and is the form the same test already uses for `U[-1]`.

`test_coupled_2d_no_flux_converges_at_first_order` asserted `0.8 <= EOC(m) <= 1.3`. Removing the
rectangle-rule error cut the m error ~10x (9.876e-03 -> 9.187e-04 at Nx=11), and what remains is
first order approached from below: the order rises 0.758 -> 0.882 -> 0.916 over Nx = 11/21/31/41.
The bound is now 0.70.

**What justifies 0.70, measured across both mutant families rather than inferred between samples.**
The clean run itself sits at 0.758, so no admissible bound exists above it and the old 0.80 was not
a discriminator at the tip -- it rejected the correct solution. The live neighbourhood is narrow:
sigma k = 1.015 survives at 0.707, k = 1.020 fails at 0.691, lambda = 1.02 survives at 0.760 and
lambda = 1.05 fails at 0.696. So 0.70 costs the diffusion mutants between 0.5% and 1.5%, and buys
0.058 of headroom over clean.

The same re-measurement corrects three standing claims in that file, each marked in place: em is
now the sensitive column rather than the weak one (a 21% diffusion error moved its EOC from
0.956 / 0.945 to 0.339 / 0.308); the documented 5%-10% detection floor is false, since k = 1.05 and
lambda = 1.05 both now fail on the m order while passing on u; and the finest-level relative error
in m is 0.735%, not the 6.6% the module docstring used to argue that this study cannot resolve a
model-side perturbation of zeta. That conclusion survives the correction; the figure did not.
