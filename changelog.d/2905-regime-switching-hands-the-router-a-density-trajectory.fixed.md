- **RegimeSwitchingIterator hands the FP drift router each regime's density trajectory from its first sweep** (#2905),
  the cold start's iterate (#2991); its first sweep handed the router the single initial density.
  - On FP-FDM, a `SeparableHamiltonian` subclass with its own `optimal_control` was refused there by name ("not a
    density trajectory", #2902), and now takes the velocity route, as the library H's control does.
  - A `CongestionHamiltonian`, whose velocity route reads the density, solved, and with the regimes decoupled (Q = 0)
    its first sweep differed from one FixedPoint sweep: by 9.2e-02 in regime 0's density on a 21-node grid with an
    exit. It now equals it.
  - An L1 control cost is unchanged.
