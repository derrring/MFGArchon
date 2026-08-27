- **`MFGProblem` validates the initial density's mass and no longer rescales it** (Issue #1887).
  Normalising is the caller's job: a sub-probability density or one population's share is a
  legitimate initial condition, and silently dividing by the discrete integral both hid a miscoded
  `m_initial` and imposed a convention nothing stated. Three tiers replace the rescale — refuse a
  non-finite, negative or non-positive-mass density; report the measured mass with the NAME of the
  measure that produced it; and warn, silenceably, when it is not 1. `problem.initial_mass` and
  `problem.initial_mass_measure` carry the result. `mass = 1` is not a law of the Fokker-Planck
  equation, which conserves whatever it starts with, so conservation checks now compare against the
  mass handed in rather than against 1.
