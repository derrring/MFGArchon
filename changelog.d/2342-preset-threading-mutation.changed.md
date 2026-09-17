- **The discrimination kill matrix carries a mutation that drops a threaded argument** (Issue #2342).
  - `constructed_preset_is_accepted_and_ignored` makes `HJBFDMSolver.__init__` store `"engquist_osher"` instead of the preset it was handed. The validation still runs, so a bad preset still raises and the constructor *looks* like it honoured the request; the preset is simply dropped at all eight read sites — the refusal check, `solve_hjb_system`, `_compute_gradients_nd`, `_build_advection_matrix_1d`, and twice each in the 1-D and nD linearised-operator assemblies, which read the momentum and its derivatives separately. It is the matrix's first mutation that removes an **argument** rather than altering arithmetic, which is the defect class #2340's blockers were about.
  - Killed by **10** tests: 8 introduced by #2340 (`5d232f62`) and 2 by #2344 (`09eaaa28`). The union over single-site mutations is only 4 distinct tests, so six of the ten fire only on a *total* drop — the row records which tests notice the preset being ignored wholesale, not which pin covers which site.
  - **The re-record also absorbs nine improvements that the weekly tier never recorded.** `--check-baseline` gates in the weekly tier only (`scripts/local_ci.sh`), so #2340, #2344 and #2341 merged without re-recording, and main's baseline is still stamped `a70d9a06` at 4057 collected against this sweep's 4090. Measured against it, nine pre-existing mutations gained 29 killer-edges and none lost any:

    | mutation | was | now |
    |---|---|---|
    | `drift_coefficient_2x` | 29 | 37 |
    | `engquist_osher_magnitude_is_the_larger_branch` | 33 | 41 |
    | `engquist_osher_derivative_drops_the_forward_branch` | 54 | 58 |
    | `hjb_marches_forward_in_time` | 25 | 27 |
    | `rouy_tourin_branch_swap` | 33 | 35 |
    | `upwind_minimum_takes_a_one_sided_difference` | 29 | 31 |
    | `optimal_control_sign` | 41 | 42 |
    | `periodic_endpoint_convention_inverted` | 23 | 24 |
    | `periodic_wrap_endpoint_inverted` | 49 | 50 |

    The script's own docstring requires improvements to be recorded in the same change, "otherwise the next baseline encodes the gain as if it had always held". They are recorded here rather than absorbed silently.
