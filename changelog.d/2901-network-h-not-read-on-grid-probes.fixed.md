- **On a geometry that declares itself a network, construction no longer reads the Hamiltonian at a grid probe** (#2901,
  stage 1c-iii). Construction's checks read the Hamiltonian's `__call__`, `dm`, `dp` and doors at a grid probe point: x
  a point's coordinates, p of the coordinate dimension and m one number. The network solvers hand x a node, and p and m
  one entry per node, and none of them hands that form. Where the geometry's `geometry_type` is `NETWORK`, each of these
  was refused, and now builds:
  - a `hamiltonian_func` that raised `TypeError`, returned `None` or returned NaN there;
  - a `hamiltonian_dm_func` that raised `TypeError` or returned `None` there;
  - a `node_interaction_func` that raised `TypeError`, returned `None` or returned NaN at a density of one entry, on a
    network where the probe's node has a single neighbour;
  - a `NetworkHamiltonian` subclass handed to `MFGProblem` through a model or its components, whose `dp` or `dm`
    raised `TypeError` there, whether a method, a static or class method, or set on the instance;
  - any other Hamiltonian handed to `MFGProblem` on a network, through a model, its components or `network=`, such as a
    `SeparableHamiltonian` whose potential returns NaN;
  - a `state_penalty` that returns one value for a batch, on a network.

  No network solver accepts a problem built through `MFGProblem` on a network. The geometry's declared type decides,
  not the Hamiltonian's type, and these are now refused:
  - a `NetworkHamiltonian` on any geometry that does not declare itself a network: the library's own on a grid, an
    implicit domain, a point cloud, a mesh or a maze built, and failed at solve (with `IndexError` on a
    `TensorProductGrid`);
  - a `NetworkMFGProblem` on a network geometry that declares another `geometry_type`, which `NetworkHJBSolver` solved
    (#2972);
  - `validate_custom_functions` called directly with a geometry object that has no `geometry_type` now raises
    `AttributeError`.

  A network Hamiltonian's sources are compared nowhere: with the built-in control `__call__`, the rates and
  `source_term` come from one formula, and with a custom `hamiltonian_func` both readers of the rates, `FPNetworkSolver`
  and `NetworkPolicyIterationHJBSolver`, refuse it (#1545). Those two refusals are now pinned.
