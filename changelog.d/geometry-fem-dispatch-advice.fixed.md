- **`get_applicator_for_geometry(..., DiscretizationType.FEM)` now names an import that exists.** Its
  `NotImplementedError` advised `from mfgarchon.geometry.boundary.bc_adapter import apply_fem_bc`, and
  neither that module nor that function exists, so following the advice raised `ModuleNotFoundError`. It
  now names `from mfgarchon.alg.numerical.fem.bc_adapter import apply_bc_to_fem_system`. A test executes
  the advice and checks the function it imports.
