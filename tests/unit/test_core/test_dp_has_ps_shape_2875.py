"""What `dp` returns has p's shape, wherever the library reads it (#2875, stage 3, second slice).

`HamiltonianBase.dp` documents (d,) for a point and (N, d) for a batch, and `ControlCostBase.dp` "same shape as p".
Every library read of what `dp` or `evaluate_dp` returns passes it through the owner of the first slice,
`checked_control`, which refuses another shape and names the reader, the source and its method, the shapes of p and m
handed, and the shape that came back. Before it (main f5e6ffc2), each read decided alone: HJB-FDM, HJB-SL, the weak-form
FEM Newton and HJB-GFDM's `joint_socp` Newton ran (N,) for p (N, 1), a leading unit axis or a 1-D transpose to the
library's result by accident; HJB-FDM's n-D linearised operator took a return of another size silently, and HJB-SL a
d = 2 transpose; the construction check `validate_hamiltonian_consistency` read the first entries of a point return
of another shape.

- **Each read refuses another shape** (a mutation-verified pin). Per read, a return of p's size in another shape and a
  return of another size; HJB-SL's in 1-D and at d = 2. The message is pinned up to the shape that came back. A form
  that applies to every batch also reaches the construction check of batch against point reads
  (`refuse_a_hamiltonian_whose_reads_disagree`), which passes it; a form it refuses (a doubled return, a d = 2
  transpose) is applied past its batch of 3, so the row is about the read. HJB-SL's reads are called directly: on each
  step the foot velocity reads `dp` before the value update does.
- **Each read takes the library's dp.** Through the same fixture, bitwise the shipped H's result, so a refusal is the
  form's and not the fixture's. Mutation-verified at four reads (HJB-FDM's and HJB-GFDM's Newton, the weak-form FEM
  Newton, `jacobian_fd`): a read that hands the owner a p other than the one it handed `dp` fails here.
- **At construction, a point return of another shape is refused** (a mutation-verified pin): (1, d) for p (d,) at d = 1
  and 2, and () for p (1,), with batches right, so a solver that reads no point, such as HJB-FDM, is refused too.
  `validate_hamiltonian_consistency` turns a failure to evaluate into a warning (#1642, C1: it gates only on an
  exhibited witness), and a return of the refused shape at its probe is one, so it is an error. The control: a dp that
  raises at the probe still leaves the result valid, with C1's warning.
- **The 0-d admission and the message per method.** *Not yet met (#2875): the form of p.* The library's dp answers a
  0-d p with (1,), as its control does; the owner admits it under `dp` too. The refusal's last sentence names what the
  method returns.

HJB-GFDM Howard's read (`alpha_star`) keeps the owner's call and has no row: Howard refuses at construction every
Hamiltonian that is not a shipped `SeparableHamiltonian` or `QuadraticMFGHamiltonian`, and each of those returns p's
shape, so no public input reaches it with another shape.
"""

from __future__ import annotations

import logging
import re
import warnings

import pytest

import numpy as np

from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian, checked_control
from mfgarchon.core.mfg_components import MFGComponents
from mfgarchon.core.mfg_problem import MFGProblem
from mfgarchon.geometry import Mesh1D, TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc
from mfgarchon.utils.validation import ValidationError, validate_custom_functions

T, NT, SIGMA = 0.5, 5, 0.3

#: The forms a test dp can take on the library's own return ``g``.
FORMS = {
    "library": lambda g: g,
    "squeezed": lambda g: g[:, 0],  # (N,) for p (N, 1)
    "lead": lambda g: g[None],  # a leading unit axis
    "transposed": lambda g: g.T,
    "double": lambda g: np.concatenate([g.ravel(), g.ravel()]),  # another size
    "scalar": lambda g: g.reshape(()),  # () for a point p (1,)
}


#: Where a form applies, on (p, t).
def _batch(p, t):
    return np.ndim(p) == 2


def _batch_past_the_probe(p, t):  # the construction check reads a batch of 3
    return np.ndim(p) == 2 and np.shape(p)[0] != 3


def _point(p, t):
    return np.ndim(p) == 1


def _point_after_0(p, t):  # construction reads points at t = 0
    return np.ndim(p) == 1 and t > 0


def _separable(form, where):
    """A SeparableHamiltonian whose dp is the library's in another form where ``where(p, t)`` holds."""

    class Formed(SeparableHamiltonian):
        def dp(self, t, x, p, m):
            g = super().dp(t=t, x=x, p=p, m=m)
            return FORMS[form](np.asarray(g)) if where(p, t) else g

    return Formed(control_cost=QuadraticControlCost(control_cost=1.0))


def _shipped():
    return SeparableHamiltonian(control_cost=QuadraticControlCost(control_cost=1.0))


def _quiet(fn):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        logging.disable(logging.CRITICAL)
        try:
            return fn()
        finally:
            logging.disable(logging.NOTSET)


def _components(h):
    return MFGComponents(
        m_initial=lambda x: np.ones_like(np.asarray(x[0], dtype=float)),
        u_terminal=lambda x: (np.asarray(x[0], dtype=float) - 0.3) ** 2,
        hamiltonian=h,
    )


def _grid_problem(h, dim=1, n=11):
    grid = TensorProductGrid(
        bounds=[(0.0, 1.0)] * dim, Nx_points=[n] * dim, boundary_conditions=no_flux_bc(dimension=dim)
    )
    return MFGProblem(geometry=grid, T=T, Nt=NT, volatility=SIGMA, components=_components(h))


X11 = np.linspace(0.0, 1.0, 11)


def _fdm(h):
    from mfgarchon.alg.numerical.hjb_solvers import HJBFDMSolver

    solver = HJBFDMSolver(_grid_problem(h))
    return solver.solve_hjb_system(np.ones((NT + 1, 11)), (X11 - 0.3) ** 2, np.zeros((NT + 1, 11)))


def _linearised(dim):
    def run(h):
        from mfgarchon.alg.numerical.hjb_solvers import HJBFDMSolver

        n = 11 if dim == 1 else 6
        pts = np.meshgrid(*[np.linspace(0.0, 1.0, n)] * dim, indexing="ij")
        U = sum((c - 0.3) ** 2 * (1 + k) for k, c in enumerate(pts))
        operator = HJBFDMSolver(_grid_problem(h, dim, n)).build_linearized_operator(U, np.ones_like(U), time=0.1)
        return operator.toarray()

    return run


def _gfdm(**options):
    def run(h):
        from mfgarchon.alg.numerical.hjb_solvers import HJBGFDMSolver

        solver = HJBGFDMSolver(_grid_problem(h), X11[:, None], delta=0.25, **options)
        return solver.solve_hjb_system(np.ones((NT + 1, 11)), (X11 - 0.3) ** 2, np.zeros((NT + 1, 11)))

    return run


def _semi_lagrangian(read, dim):
    def run(h):
        from mfgarchon.alg.numerical.hjb_solvers import HJBSemiLagrangianSolver

        solver = HJBSemiLagrangianSolver(_grid_problem(h, dim, 11 if dim == 1 else 6))
        n = 7
        x = np.column_stack([np.linspace(0.1, 0.9, n)] * dim)
        p = np.column_stack([np.linspace(-1.0, 1.4, n) * (1 + 0.5 * k) for k in range(dim)])
        if read == "foot":
            return solver._characteristic_foot_velocity(x, np.ones(n), p, 0.1)
        return solver._sl_value_update(np.linspace(0.0, 1.0, n), x, np.ones(n), p, 0.1, 0.05)

    return run


def _fem_newton(h):
    from mfgarchon.alg.numerical.fem.hjb_fem_solver import HJBFEMSolver

    mesh = Mesh1D(bounds=(0.0, 1.0), num_elements=10)
    mesh.generate_mesh()
    mesh.boundary_conditions = no_flux_bc(dimension=1)
    problem = MFGProblem(
        geometry=mesh, T=T, Nt=NT, volatility=SIGMA, coupling_coefficient=0.0, components=_components(h)
    )
    solver = HJBFEMSolver(problem, order=1)
    x = np.asarray(solver._disc.dof_coordinates, dtype=float).ravel()
    n = len(x)
    return solver.solve_hjb_system(
        M_density=np.ones((NT + 1, n)),
        U_terminal=(x - 0.3) ** 2,
        U_coupling_prev=np.zeros((NT + 1, n)),
        use_newton=True,
    )


def _built_2d(h):
    _grid_problem(h, 2, 6)
    return np.zeros(1)


def _jacobian_fd(h):
    jac = h.jacobian_fd(t=0.1, x=(0.5,), p=np.array([0.7]), m=1.0, dx=0.1)
    return np.array([jac.diagonal, jac.lower, jac.upper], dtype=float)


#: read -> (the reader the refusal names, the method it names, how it is run with an H, the m clause it names, and
#: (form, where it applies, shape of p handed, shape that came back) per row). At f5e6ffc2 each row ran (silently wrong,
#: or right by accident), refused with a message of its own, or raised a numpy error; the PR's revert run lists which.
READS = {
    "HJB-FDM Newton": (
        "compute_hjb_jacobian",
        "evaluate_dp",
        _fdm,
        "m of shape (11,)",
        [("squeezed", _batch, (11, 1), (11,)), ("double", _batch_past_the_probe, (11, 1), (22,))],
    ),
    "HJB-GFDM Newton": (
        "assemble_hjb_jacobian_diag",
        "evaluate_dp",
        _gfdm(),
        "m of shape (11,)",
        [("squeezed", _batch, (11, 1), (11,)), ("double", _batch_past_the_probe, (11, 1), (22,))],
    ),
    "HJB-FDM linearised operator 1-D": (
        "HJBFDMSolver._build_linearized_operator_1d",
        "evaluate_dp",
        _linearised(1),
        "m of shape (11,)",
        [("squeezed", _batch, (11, 1), (11,)), ("double", _batch_past_the_probe, (11, 1), (22,))],
    ),
    "HJB-FDM linearised operator 2-D": (
        "HJBFDMSolver._build_linearized_operator_nd",
        "evaluate_dp",
        _linearised(2),
        "m of shape (36,)",
        [("transposed", _batch_past_the_probe, (36, 2), (2, 36)), ("double", _batch_past_the_probe, (36, 2), (144,))],
    ),
    "HJB-GFDM joint_socp Newton": (
        "HJBGFDMSolver._compute_hjb_jacobian_sparse",
        "dp",
        _gfdm(monotonicity_scheme="joint_socp", monotonicity_application="precompute"),
        "m of shape ()",
        [("lead", _point_after_0, (1,), (1, 1)), ("double", _point_after_0, (1,), (2,))],
    ),
    "HJB-SL foot velocity 1-D": (
        "HJBSemiLagrangianSolver._characteristic_foot_velocity",
        "evaluate_dp",
        _semi_lagrangian("foot", 1),
        "m of shape (7,)",
        [("squeezed", _batch, (7, 1), (7,))],
    ),
    "HJB-SL foot velocity 2-D": (
        "HJBSemiLagrangianSolver._characteristic_foot_velocity",
        "evaluate_dp",
        _semi_lagrangian("foot", 2),
        "m of shape (7,)",
        [("transposed", _batch_past_the_probe, (7, 2), (2, 7)), ("double", _batch_past_the_probe, (7, 2), (28,))],
    ),
    "HJB-SL value update 1-D": (
        "HJBSemiLagrangianSolver._sl_value_update",
        "evaluate_dp",
        _semi_lagrangian("value", 1),
        "m of shape (7,)",
        [("squeezed", _batch, (7, 1), (7,))],
    ),
    "HJB-SL value update 2-D": (
        "HJBSemiLagrangianSolver._sl_value_update",
        "evaluate_dp",
        _semi_lagrangian("value", 2),
        "m of shape (7,)",
        [("transposed", _batch_past_the_probe, (7, 2), (2, 7)), ("double", _batch_past_the_probe, (7, 2), (28,))],
    ),
    "weak-form FEM Newton": (
        "WeakFormHJBSolver._solve_timestep_newton",
        "dp",
        _fem_newton,
        "m of shape (11,)",
        [("squeezed", _batch, (11, 1), (11,)), ("double", _batch_past_the_probe, (11, 1), (22,))],
    ),
    "construction, a solver that reads no point": (
        "validate_hamiltonian_consistency",
        "dp",
        _fdm,
        "m of shape ()",
        [("lead", _point, (1,), (1, 1)), ("scalar", _point, (1,), ())],
    ),
    "construction 2-D": (
        "validate_hamiltonian_consistency",
        "dp",
        _built_2d,
        "m of shape ()",
        [("lead", _point, (2,), (1, 2))],
    ),
    "jacobian_fd": (
        "HamiltonianBase.jacobian_fd",
        "dp",
        _jacobian_fd,
        "m of shape ()",
        [("lead", _point, (1,), (1, 1)), ("double", _point, (1,), (2,))],
    ),
}
REFUSALS = [(read, form) for read, row in READS.items() for form, _, _, _ in row[4]]


@pytest.mark.parametrize(("read", "form"), REFUSALS)
def test_each_read_refuses_a_dp_of_another_shape(read, form):
    reader, method, run, m_clause, cases = READS[read]
    where, p_shape, got_shape = next((w, p, got) for f, w, p, got in cases if f == form)
    at_construction = reader == "validate_hamiltonian_consistency"
    message = (
        ("^" if not at_construction else r"^\[ERROR\] \(hamiltonian\) ")
        + rf"{re.escape(reader)}: Formed\.{method} was handed p of shape {re.escape(str(p_shape))} and "
        rf"{re.escape(m_clause)}, and returned shape {re.escape(str(got_shape))}\."
    )
    with pytest.raises(ValidationError if at_construction else ValueError, match=message):
        _quiet(lambda: run(_separable(form, where)))


@pytest.mark.parametrize("read", [read for read in READS if not read.startswith("construction")])
def test_each_read_takes_the_librarys_dp(read):
    """The dp every reader must accept, through the same fixture, gives the shipped H's result bitwise: the refusal
    above is the form's, not the fixture's."""
    run = READS[read][2]
    formed = _quiet(lambda: run(_separable("library", _batch)))
    shipped = _quiet(lambda: run(_shipped()))
    formed, shipped = (np.asarray(r[0] if isinstance(r, tuple) else r) for r in (formed, shipped))
    assert formed.shape == shipped.shape
    assert np.array_equal(formed, shipped)


def test_a_dp_that_fails_to_evaluate_at_the_probe_still_warns():
    """C1 (#1642) is intact: only the owner's refusal moved to the error channel. A dp that raises at the consistency
    probe is a failure to evaluate, which warns and leaves the result valid. Called as MFGProblem calls it: through a
    problem, the batch-against-point check reads the same point dp first."""

    class Raising(SeparableHamiltonian):
        def dp(self, t, x, p, m):
            if np.ndim(p) == 1:
                raise ValueError("this dp cannot be evaluated at a point")
            return super().dp(t=t, x=x, p=p, m=m)

    h = Raising(control_cost=QuadraticControlCost(control_cost=1.0))
    grid = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[6], boundary_conditions=no_flux_bc(dimension=1))
    result = validate_custom_functions(hamiltonian=h, dH_dm=h.dm, dH_dp=h.dp, geometry=grid)
    assert result.is_valid
    warned = [str(issue.message) for issue in result.issues]
    assert any(
        w.startswith("Could not verify Hamiltonian consistency: this dp cannot be evaluated at a point") for w in warned
    ), warned


def test_a_0d_p_may_come_back_as_one_value_from_dp():
    """*Not yet met (#2875):* the library's dp answers a 0-d p with (1,), since it hands ``np.atleast_1d(p)`` to the
    cost. Admitted under ``dp`` as under ``optimal_control``, and returned 0-d, until stage 3 sets the form of p."""
    library = _shipped()
    returned = library.dp(t=0.0, x=(0.5,), p=np.float64(2.0), m=1.0)
    assert np.shape(returned) == (1,)
    got = checked_control(returned, np.float64(2.0), reader="here", source=library, method="dp")
    assert got.shape == ()
    assert got == 2.0


@pytest.mark.parametrize(
    ("method", "contract"),
    [
        (
            "optimal_control",
            "a control of p's shape, as HamiltonianBase.optimal_control and ControlCostBase.optimal_control",
        ),
        (
            "evaluate_optimal_control",
            "a control of p's shape, as HamiltonianBase.optimal_control and ControlCostBase.optimal_control",
        ),
        ("dp", "dH/dp of p's shape, as HamiltonianBase.dp and ControlCostBase.dp"),
        ("evaluate_dp", "dH/dp of p's shape, as HamiltonianBase.dp and ControlCostBase.dp"),
    ],
)
def test_the_refusal_says_what_the_method_returns(method, contract):
    message = (
        rf"^here: object\.{method} was handed p of shape \(3,\) and m None, and returned shape \(2,\)\. "
        rf"The library reads {re.escape(contract)} document "
        r'\(docs/user/CONVENTIONS\.md, "What optimal_control and dp return"; #2875\)\.$'
    )
    with pytest.raises(ValueError, match=message):
        checked_control(np.zeros(2), np.zeros(3), reader="here", source=object(), method=method)
