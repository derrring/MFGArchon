- **A rebuilt problem carries every input it was built from** (#2667).
  - `MFGProblem.solve(Nt=)`, `with_model`, `with_domain` and `with_conditions` rebuild a problem made in the
    model/domain/conditions form. Each restated model, domain, conditions, constraints and Nt, and dropped everything
    else the constructor took: `source_term_hjb`, `source_term_fp`, `nonlocal_operator`, `state_penalty`,
    `state_penalty_scale`, `obstacles`, `lambda_`, `coupling_coefficient`, `suppress_warnings`. A problem with a
    source solved at another Nt as if it had none. All four now read one owner, `MFGProblem._rebuilt`.
  - `MFGProblem(boundary_conditions=...)` is refused, in every form: nothing read it, so the problem solved on its
    geometry's boundary conditions whatever it said. Declare them on the geometry, or as
    `MFGComponents(boundary_conditions=...)` in the components form.
  - `drift=` is refused in the model/domain/conditions form: no solver reads the problem's drift field (#2678).
  - A rebuild now meets the new object with the old inputs, and refuses where the constructor would: a problem built
    with `state_penalty` and rebuilt by `with_model` onto a `DualHamiltonian` raises, where the penalty used to be
    dropped silently.
