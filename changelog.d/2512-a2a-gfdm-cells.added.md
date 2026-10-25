- **HJB-GFDM has exact-solution oracles on its NEUMANN and PERIODIC cells in 2-D and 1-D, and its ROBIN and
  positive_qp_fan NEUMANN cells are counted by routing** (#2512 (a), stage A2a). The manifest gains one cell,
  `hjb_gfdm_positive_qp_fan/neumann_2d` (#2512 comments 6333097202 and 6333790265), and its oracle line goes from
  49/67 to 54/68.
  - New, in `tests/unit/test_alg/test_gfdm_bc_oracles_2512.py`, on the default configuration (`none` with Newton):
    - a NEUMANN datum that differs on each face, and one shared by every face, on exp17's Sobol cloud with the four box
      corners added, at second order;
    - a periodic solution that is not even about the seam, on the problem's own periodic grid, 1-D and 2-D, at second
      order.
  - Each asserts the order between levels, each level's error, and the error on each face's nodes and the corners, or
    on the seam nodes, against its recorded value. Rivals are measured and recorded there: the row along the tangent,
    the normal's sign, the datum dropped, a face read for another, the corner's normal averaged, and, for the seam, no
    ghost points, mirror ghosts and a period too long.
  - ROBIN(0, 1, g) on each face solves as NEUMANN(g) there, in 2-D, to the bit at t = 0; positive_qp_fan solves
    NEUMANN(0) as NO_FLUX on its reflecting cell's box, to the bit.
  - The PERIODIC declaration's comment in `hjb_gfdm.py` no longer calls #1841 open.
