`HJBWENOSolver` carried the dimensional solve three times, and one copy did not run (#2021).

`_solve_hjb_system_2d`, `_3d` and `_nd` were the same backward loop written out three times,
differing in slice notation (`[:, :]` / `[:, :, :]` / `[...]`, and the last covers the others), in
logging, and in **three different sources for one array shape** — 2d read `self.num_grid_points_*`,
3d read `M_density.shape[1:]`, nd read `U_terminal.shape`, with nothing checking they agree.

**The 3-D copy raised on its first statement.** It called `self._get_logger()`, which exists nowhere
in the MRO, so every 3-D WENO solve failed with `AttributeError` and always had. Twenty-three test
files mention WENO and none is 3-D — which is what a duplicated path with no oracle looks like from
the outside: green. 3-D now runs.

Six rivals deleted: the two extra solves, the two one-line `_step_*_split` delegators, and the two
extra CFL steps. Implementation count for those three families drops 8 → 3 (`_1d` and `_nd` for the
solve and the CFL step, one `_step_nd_split`). The 1-D path stays separate on purpose: it is not a
dimensional split and is the only one carrying `source_term`.

**The surviving CFL step takes the deleted 3-D copy's semantics, which were the corrected ones.**
Zero gradient gets an explicit branch rather than an epsilon, and the `max(dt_stable, 1e-10)` floor
is gone. That floor said "ensure positive time step", but positivity is not the property required:
when the diffusion-limited step is genuinely smaller it returns a step **above** the stability bound
— measured, 1e-10 against a true limit of 3.906250e-15 at `sigma = 1e7` — and the solve runs
unstably, where without it `_advance_full_interval` reaches its `max_substeps` guard and fails loud.

Not oversold: that floor needs `sigma > h * 5e4` to fire, i.e. `sigma > 5000` at `h = 0.1`, which no
grid an explicit scheme can afford will reach. It is removed because it converts a loud failure into
a silent one, not because it was firing. `_compute_dt_stable_1d` still carries it, and a test records
that rather than letting the change widen silently into a path this consolidation does not own.

Pinned against the **pre-consolidation** output, captured before the rivals were deleted — comparing
the paths against each other afterwards would be tautological, since they are now the same code. The
2-D solve reproduces `sum = -4.825099898418e+02` and `u0_sum = -1.298054011318e+02` exactly; only
`dt_stable` moves, in the 11th digit (1.238178469094e-02 → 1.238178469155e-02), which is the removed
epsilon.
