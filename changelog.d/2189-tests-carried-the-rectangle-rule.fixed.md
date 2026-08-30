Two integration tests built their expectation with the measure the library had stopped using, and
went red on `main` after #2145 and #1887.

`test_jacobian_transpose_converges` reconstructed the initial density as `m0 /= np.sum(m0) * dx`
and compared at `atol=1e-15`. It now asserts that `result.M[0]` comes back exactly as handed in,
with no normalisation at all -- which is what #1887 promises and is strictly stronger than the
constant it used to pin.

`test_coupled_2d_no_flux_converges_at_first_order` asserted `0.8 <= EOC(m) <= 1.3`. Removing the
rectangle-rule error cut the m error ~10x (9.876e-03 -> 9.187e-04 at Nx=11), and what remains is
first order approached from below: the order rises 0.758 -> 0.882 -> 0.916 over Nx = 11/21/31/41.
The bound is now 0.70, and the diffusion mutants were re-measured to show the widening costs no
kill -- the clean run reads 0.758 and the nearest mutant 0.495. The same re-measurement corrects
the docstring: before #2145 a 21% diffusion error left EOC(m) at 0.956 / 0.945 and killed nothing;
it now reads 0.339 / 0.308, so em became a working discriminator rather than the weak column.
