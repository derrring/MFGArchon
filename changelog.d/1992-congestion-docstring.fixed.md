`BaseMFGProblem.hamiltonian`'s docstring no longer calls `0.5*|p|^2 + 0.1*m` a congestion
Hamiltonian. It is a separable Hamiltonian with a local coupling: congestion means the density
scales the momentum term, `H = (m^alpha/gamma)(|p|/m^alpha)^gamma` (Gomes-Pimentel-Voskanyan
p. 116), whereas there `m` sits in a term with no `p` in it. The example now points to
`CongestionHamiltonian` for the real form, flips the sign to `-0.1*m` so the default is
Lasry-Lions monotone rather than crowd-attracting, and states that `m` is the scalar density at
`x` and not the measure, so a nonlocal coupling has to go through `source_term_hjb`.
