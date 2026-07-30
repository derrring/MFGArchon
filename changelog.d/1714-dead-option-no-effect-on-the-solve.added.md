- **The dead-option guards now have a test that measures their premise** (#1714, #1426). A
  `pytest.raises` on "stored but never read" records that the guard fires, not that refusing is
  right. Injecting `_boundary_indices` and `_domain_bounds` after construction, bypassing the
  guard, and comparing full solves: every node marked boundary, a domain fifty times the real one,
  and a domain covering a fifth of it all leave the solution byte-identical under a nonzero drift.
  Scope, stated because it is narrower than "the option is dead": this establishes that no code
  reached by `solve_fp_system` on this configuration branches on either value in a way that
  changes the returned array. A read with no effect on the output — into a log record, say — would
  not be detected, and construction-time consumption is out of reach of post-construction
  injection by design.
