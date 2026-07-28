- **18 modules with no inbound imports, and four subtrees past their own removal
  deadline** (Issue #1711) — `backends/solver_wrapper.py`, `types/problem_protocols.py`,
  `geometry/boundary/handler_protocol.py` (whose symbols already live in
  `protocols.py`), `utils/notebooks/pydantic_integration.py`,
  `alg/reinforcement/environments/multi_population_env.py`, the empty
  `alg/reinforcement/approaches/` scaffold, three never-populated placeholder packages,
  and the `geometry/operators/`, `alg/iterative/`, `utils/numerical/mesh_distances.py`
  shims (declared for removal in v0.19.0–v0.21.0; the version is 0.22.0.dev0). Product
  code drops 2,153 lines.
