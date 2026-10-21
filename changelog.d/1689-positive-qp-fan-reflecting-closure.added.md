- **`monotonicity_scheme="positive_qp_fan"` closes reflecting faces** (#1689 PR 4). A no-flux or Neumann face with
  the datum 0 of an axis-aligned box, whole faces in any combination with Dirichlet faces, is closed by two
  rows, each admitted by its certificate:
  - **a mirror row** on every interior node whose nearest reflecting face lies within `MIRROR_REACH` = 3/5 of its
    pool's radius: the positive P2 row over its 21 nearest nodes and their mirror images across that face, under the
    paper's obliqueness certificate (chap_01 eq:bc_gamma_main), `2 delta_i sum_j L_ghost_ij s_ij >= MIRROR_GAMMA0` =
    0.5, `s_ij` a source's distance to the face and `delta_i` the pool's radius, with each ghost's weight folded onto
    its source, so the row stays Metzler with zero row sum;
  - **a value-copy row** `u_b - u_j = 0` on each sampled node of the face, `j` its nearest interior node.
  On exp17's Sobol clouds (seed 42), on exp17's exact reflected reference and on an asymmetric one, the error is
  below the no-mirror operator's at every level from N_int 900 to 2500: 1.8 to 3.4 times with every face no-flux, 1.9
  to 5.0 times with x_max Dirichlet. That is a measurement on those clouds, not a guarantee: on other scrambles of the
  same Sobol sequence the no-mirror operator is the more accurate in a few cases. Two differences from the paper are stated in the module: copied boundary nodes are stencil sources, and a
  mirror row goes on every node within reach, where the paper places one only where the plain row fails. Refused by
  name: an obstacle beside a reflecting face, a door (a face part Dirichlet, part reflecting), a boundary read from
  `domain_sdf`, a no-flux or Neumann datum other than the constant 0, a Robin face, a periodic face, a node within reach that no mirror row
  places or that has no complete physical fan, and a boundary condition adopted after construction that reflects on
  other faces, or gives a value-copy row at other points, than the operator was built with. Before, every face other than Dirichlet was refused.
