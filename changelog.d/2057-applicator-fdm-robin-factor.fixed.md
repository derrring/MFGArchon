Three corrections in `applicator_fdm.py`'s module docstring, all tails of #2057.

**The Robin block carried the stale `beta/(2*dx)` factor**, four lines below the Neumann block
#2057 corrected for the same factor. The branch at the bottom of the same file delegates to
`ghost_cell_robin`, which uses `beta/dx`. Measured (`alpha=1, beta=0.3, g=0.7, dx=0.1, u_i=0.5`):
the docstring's form gives **0.600000** with a residual of **+0.15** against the Robin condition;
the live path gives **0.557143** with residual **0**. It was also internally inconsistent — the
value term `(u_g + u_i)/2` commits to the face-midpoint geometry, where the separation is `dx`.

The only text in the repo naming that factor as stale was a `test_robin` docstring calling it *"the
stale pre-fix factor (Refs #1237)"*, and **#2057 deleted it** along with the orphaned method it
covered. The correction and its only signpost were removed in the same change; this restores the
correction.

**A sentence #2057 added to the Neumann block claimed both centrings** inside a section whose header
scopes it *"(cell-centered grid, boundary at cell face)"* with `x_b = (x_g + x_i)/2`. Under that
geometry a vertex-centred pair at `-dx` and `0` puts the boundary at `-dx/2`, not on the vertex.
`ghost_cell_neumann` *is* centring-free, but that is a property of the function, not of the layout
this header declares.

**The zero-gradient branch's comment wrote `du/dn = (u_interior - u_ghost)/dx`**, which is `-du/dn`.
Interior → ghost is the outward direction at either wall, which is exactly what makes the header's
formula sign-free. Inert at zero flux, and contradicting the header twenty lines up.
