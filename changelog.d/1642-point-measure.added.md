- **What a coupling cost binds against is a protocol: `PointMeasure`** (#1642 stage 1b-2). It is points in the
  problem's frame, one quadrature weight per point, and the measure of the domain the points discretize, from
  `mfgarchon.operators.interaction`.
  - The library's costs bind on one through `bind_on(measure)`, the protocol `BindsOnPointMeasure`.
    `bind_on_measure(cost, measure)` is the one place a measure is put to use. It refuses, by name, points not of shape
    `(n, d)`, weights not of shape `(n,)`, a weight that is not positive, weights that do not sum to the domain's
    measure, a domain measure that is not one positive number, a cost with no `bind_on`, and a `bind_on` that returns
    no energy on the measure's points. `InteractionCost(use_fft=True)` on a measure that is not a grid's is refused.
  - A problem's grid is one such measure, `GridPointMeasure`: its nodes and the problem's measure, on the box the
    nodes span, which is what that measure integrates over (#2769). `InteractionCost(K).bind(problem)` and
    `PotentialCost(V).bind(problem)` bind through it, and so do the library parts of a sum. Every bound array is
    bit-identical to before, and the interaction keeps its FFT path on a uniform grid. It refuses subclassing, and
    its points and weights are read-only, because the interaction's FFT path reads the problem's grid and measure
    and would ignore a change to either.
  - A finite-element space's dofs with their weights, and a collocation cloud's points with theirs, are the
    measures stage 1b-2 and #2761 add.
