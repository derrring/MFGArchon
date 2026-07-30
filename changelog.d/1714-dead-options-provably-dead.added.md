- **The dead-option guards now have a test that proves the option is dead** (#1714, #1426). A
  `pytest.raises` on "stored but never read" cannot distinguish "there is no reader" from "I could
  not find a reader" — and a grep only ever gives the second. Injecting `_boundary_indices` and
  `_domain_bounds` after construction, so the guard is bypassed, and comparing full solves: marking
  every node a boundary, and a domain fifty times the real one, both leave the solution
  byte-identical. The test asserts the injected attribute exists first, because setting a public
  name where the solver stores a private one creates an unread attribute and byte-identity then
  follows trivially.
