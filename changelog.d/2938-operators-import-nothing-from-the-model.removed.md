- **The operators layer imports nothing from the model layer** (#2938, stage 2d). Each name below keeps its name and
  moves to its owner, with no alias:
  - `InteractionCost`, `PotentialCost` and `CombinedCost`, the values of `Model.coupling_cost`: from
    `mfgarchon.core.coupling_costs`, not `mfgarchon.operators.interaction` or
    `mfgarchon.operators.interaction.coupling_costs`, which is gone;
  - `GridPointMeasure`: from `mfgarchon.core.coupling_costs`, not `mfgarchon.operators.interaction` or
    `mfgarchon.operators.interaction.point_measure`;
  - `bound_energy_functional`: from `mfgarchon.core.coupling_costs`, not
    `mfgarchon.operators.interaction.energy_functionals`;
  - `diffusion_from_volatility`, `volatility_from_diffusion`, `validate_symmetric_psd` and
    `diffusion_from_volatility_torch`: from `mfgarchon.utils.numerical.volatility`, not
    `mfgarchon.utils.pde_coefficients`.
- **`ConvolutionCouplingOperator.from_problem` is removed.** `InteractionCost(kernel).bind(problem)` builds the same
  convolution on a problem's grid, and its energy applies it; a convolution on given points or a regular grid is
  `ConvolutionCouplingOperator(kernel, ...)`, as before. Every input it refused is still refused, through
  `InteractionCost(kernel).bind(problem)`: a periodic axis and a `CartesianGrid` that is not a `TensorProductGrid` by
  `InteractionCost`, and a geometry that is not a `CartesianGrid` by `GridPointMeasure`, which reads the problem first.
