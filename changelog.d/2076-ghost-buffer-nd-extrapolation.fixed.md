- **`GhostBuffer.update()` crashed in 2D/3D on extrapolation BCs, and disagreed with
  `pad_array_with_ghosts` at `ghost_depth > 1`** (Issue #2059 follow-up). The interior stencil was
  taken with an INTEGER index, which drops the axis; the dropped-axis layer then broadcast against
  the kept-axis `interior_value` along the wrong axis, so on a (7, 7) buffer with `axis=1` a (7, 1)
  interior and a (7,) layer broadcast to (7, 7) and raised on assignment back into the (7, 1) ghost
  slot. 1D could not see it and 1D was the only shape tested. Separately, at `g > 1` the mirror
  slices carried `g` interior layers, so the calculator produced `g` different ghosts while
  `pad_array_with_ghosts` evaluates the wall-anchored stencil once and writes that value to every
  layer — measured disagreement 2.200e+01 at g=2, 2D (7, 6), linear. Both paths now agree across
  1D/2D/3D and `g` in {1, 2}. The constant ghost REGION that agreement reproduces is not fixed
  here; it stays tracked in #1966.
