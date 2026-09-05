"""A network problem's mass is the node sum, not 1/N (Issue #2177).

`_measure_initial_density` gated its network branch on ``self.dimension``, but
``NetworkMFGProblem`` sets ``dimension = "network"`` *after* ``super().__init__()`` and the
measurement runs inside it. The guard therefore read ``dimension == 2`` — a network problem
describing itself as two-dimensional during its own construction — and fell through the
ladder to ``point-average``, ``sum(m) / num_spatial_points``.

Three things were wrong at once and the third caused the others:

1. ``problem.initial_mass`` reported ``1/N`` (0.04 on a 5x5 grid network) under the name
   "initial density mass", while the density already summed to 1.
2. The #1887 warning fired as a false positive, with a remedy — divide by the integral —
   that could not work, because dividing a density that already sums to 1 changes nothing.
3. The ``node-sum`` branch written for exactly this case never executed, and
   ``NetworkMFGProblem`` is the only network problem class, so it was dead as written.

The gate now reads ``self.is_network``, which derives from ``geometry.geometry_type`` and is
already correct at measure time. Same lesson as #2157: gate on the thing you are about to
use. Note this does NOT depend on the constructor ordering being changed — ``topology.py``
carries a comment explaining why ``dimension`` is set late, and that ordering is untouched.

Retirement condition: these trip if the network branch is ever gated on something not
available at measure time again, or if ``point-average`` starts claiming a network.
"""

from __future__ import annotations

import pytest

import numpy as np

from mfgarchon.extensions.topology import NetworkMFGProblem
from mfgarchon.geometry.graph.network_geometry import GridNetwork


@pytest.fixture
def network_problem():
    net = GridNetwork(width=5, height=5)
    net.create_network()
    return NetworkMFGProblem(geometry=net, T=1.0, Nt=20)


class TestNetworkMassIsMeasuredOnTheNodes:
    def test_the_reported_mass_is_the_node_sum_and_not_one_over_n(self, network_problem):
        """25 nodes, so ``point-average`` gives 0.04 and ``node-sum`` gives 1.0 — a 25x gap.

        Asserting against ``sum(m)`` rather than the literal 1.0 keeps this a statement about
        the *measure* rather than about this fixture's normalisation.
        """
        m = np.asarray(network_problem.m_initial)
        assert network_problem.initial_mass == pytest.approx(float(np.sum(m)))
        assert network_problem.initial_mass != pytest.approx(1.0 / len(m)), (
            "the point-average branch is still measuring this network (#2177)"
        )

    def test_the_measure_names_itself_node_sum(self, network_problem):
        """#1887's rule: a number without the name of the measure that produced it is the
        invisible convention. The name is the half a reader acts on."""
        assert network_problem.initial_mass_measure == "node-sum"

    def test_no_false_positive_mass_warning_and_the_capture_works(self):
        """The warning's own remedy was inert, which is what made it a defect and not noise:
        the density already summed to 1, so dividing by its integral is a no-op.

        The absence half is paired with a PRESENCE half in the same test and through the same
        capture. An absence assertion alone is satisfied by a broken capture, a filtered
        warning, or a typo in the match string exactly as it is by a warning that did not fire
        — so the second block builds a grid problem whose mass really is not 1 and requires
        the identical machinery to see it.
        """
        import warnings

        net = GridNetwork(width=5, height=5)
        net.create_network()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            NetworkMFGProblem(geometry=net, T=1.0, Nt=20)
        absent = [w for w in caught if "mass" in str(w.message).lower()]
        assert not absent, f"the #1887 mass warning still fires on a network: {absent}"

        # Presence control: same capture, a density whose mass genuinely is not 1.
        from mfgarchon import Conditions, MFGProblem, Model
        from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
        from mfgarchon.geometry import TensorProductGrid
        from mfgarchon.geometry.boundary import no_flux_bc

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            MFGProblem(
                model=Model(
                    hamiltonian=SeparableHamiltonian(
                        control_cost=QuadraticControlCost(control_cost=1.0), coupling=lambda m: m
                    ),
                    sigma=0.3,
                ),
                domain=TensorProductGrid(
                    bounds=[(0.0, 1.0)], Nx_points=[11], boundary_conditions=no_flux_bc(dimension=1)
                ),
                conditions=Conditions(
                    u_terminal=lambda x: np.squeeze(0.0 * np.asarray(x)),
                    m_initial=lambda x: 0.25,  # mass 0.25 on [0, 1]: genuinely not 1
                    T=0.1,
                ),
                Nt=4,
            )
        present = [w for w in caught if "mass" in str(w.message).lower()]
        assert present, "the capture sees no mass warning even where one is due -- the absence above proves nothing"

    def test_a_grid_problem_still_uses_the_grid_measure(self):
        """The control. The gate changed for every problem, not only networks, so a grid
        problem must still reach ``integrate`` — if this ever returned ``node-sum`` the new
        predicate would be claiming every geometry."""
        from mfgarchon import Conditions, MFGProblem, Model
        from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
        from mfgarchon.geometry import TensorProductGrid
        from mfgarchon.geometry.boundary import no_flux_bc

        problem = MFGProblem(
            model=Model(
                hamiltonian=SeparableHamiltonian(
                    control_cost=QuadraticControlCost(control_cost=1.0), coupling=lambda m: m
                ),
                sigma=0.3,
            ),
            domain=TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[11], boundary_conditions=no_flux_bc(dimension=1)),
            conditions=Conditions(u_terminal=lambda x: np.squeeze(0.0 * np.asarray(x)), m_initial=lambda x: 1.0, T=0.1),
            Nt=4,
        )
        assert problem.initial_mass_measure == "grid"
        assert problem.is_network is False
