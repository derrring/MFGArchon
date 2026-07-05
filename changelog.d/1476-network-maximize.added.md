- **Network MFG supports `sense=MAXIMIZE`** (Issue #1476). The finite-state network Hamiltonian now
  handles reward-to-go / uphill control as well as cost-to-go / downhill, closing a network-vs-continuum
  gap (the continuum `SeparableHamiltonian` already respected `sense`). The MINIMIZE↔MAXIMIZE mirror is
  single-sourced through one orientation sign `NetworkHamiltonian.sense_sign` (+1 / −1): the **control**
  terms flip with `s` — control cost, `optimal_control`, `dp`, the RK45 backward-integration control
  term, and the policy-iteration control cost — while the **source** (node potential + congestion) is
  sense-independent (a running payoff accumulates identically whether you minimise or maximise). Pass
  `sense` to `NetworkMFGProblem(...)`; the previous fail-loud on MAXIMIZE is removed. MINIMIZE is
  byte-identical (`s=+1`). Validated: MAXIMIZE control consistency (uphill, α≥0, envelope, method==object),
  reward-to-go physicality for both a terminal reward and a running-reward potential (value peaks at /
  increases toward the reward), and policy-iteration agrees with RK45 for both senses.
