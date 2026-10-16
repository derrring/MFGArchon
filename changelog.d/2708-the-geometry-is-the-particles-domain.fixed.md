- **`FPParticleSolver` keeps its particles in the problem's implicit geometry** (#2708, #2679 stage 2c). It read the
  domain its particles must end in only from the `implicit_domain=` keyword, so a solve on an `ImplicitDomain`
  geometry recorded particles inside its obstacles and outside its curved boundary: up to 335 of 500 inside a disc
  obstacle, and up to 499 of 500 outside a disc.
  - One owner, `FPParticleSolver._obstacle_domain`, reads two channels: the problem's geometry when it is an
    `ImplicitDomain`, and the keyword, the only channel for a grid geometry with a separate obstacle domain.
  - Both naming the same object is accepted. Two different objects that both reach the placement, even two equal
    domains built twice, are refused by name. The message gives both routes: drop `implicit_domain=` when the two are
    one domain, and when `implicit_domain=` carries an obstacle inside the geometry, make that domain the problem's
    geometry, e.g. `DifferenceDomain(geometry, obstacle)`.
  - **Behaviour changes** on an `ImplicitDomain` geometry other than a grid-box `Hyperrectangle`, which is now a
    domain the particles must stay in:
    - a periodic outer condition is refused, where it solved with particles recorded inside an obstacle;
    - initial particles, or `m_initial` mass, outside the domain are refused, where they were recorded there;
    - the GPU path is refused;
    - a keyword that cut a hole out of a disc geometry solved on the keyword alone, and is now refused by the rule
      above.
    A `Hyperrectangle` subclass is such a geometry. A periodic grid with a grid-box `Hyperrectangle` keyword now
    solves, where it was refused.
  - On that route every boundary of the geometry off the grid box's walls reflects, whatever boundary condition was
    declared: a declared condition acts only on the box's walls (#2684).
  - A `Hyperrectangle` on the grid's box, in either channel, reaches no placement: it carries nothing the boundary
    condition does not place. So a periodic box geometry still solves as a torus, and a box geometry beside an
    obstacle keyword leaves the keyword to declare the domain.
