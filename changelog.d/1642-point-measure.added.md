- **What a coupling cost binds against is a protocol: `PointMeasure`** (#1642 stage 1b-2). It is points in the
  problem's frame, one quadrature weight per point, and the measure of the domain the points discretize, from
  `mfgarchon.operators.interaction`.
  - The library's costs bind on one through `bind_on(measure)`, the protocol `BindsOnPointMeasure`.
    `bind_on_measure(cost, measure)` is the one place a measure is put to use. It refuses, by name, a measure whose
    weights are negative or do not sum to the domain's measure, and a cost with no `bind_on`.
  - A problem's grid is one such measure, `GridPointMeasure`: its nodes and the problem's measure.
    `InteractionCost(K).bind(problem)`, `PotentialCost(V).bind(problem)` and their sum bind through it. Every bound
    array is bit-identical to before, and the interaction keeps its FFT path on a uniform grid.
  - A finite-element space's dofs with their weights, and a collocation cloud's points with theirs, are the
    measures stage 1b-2 and #2761 add.
