"""
Multi-population MFG problem container.

Issue #910 Phase 2: Holds K single-population MFGProblems and defines
cross-population coupling. NOT an MFGProblem subclass — it is a
container that coordinates K independent problems.

Each population k has:
- Its own HamiltonianBase H_k (may depend on all densities m_1,...,m_K)
- Its own (u_k, m_k) solution pair
- Its own HJB and FP solvers

Cross-population coupling enters through H_k's coupling function,
which receives the full density state (all K populations).
"""

from __future__ import annotations

from dataclasses import KW_ONLY, dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mfgarchon.core.mfg_problem import MFGProblem


@dataclass
class MultiPopulationProblem:
    """Container for K-population MFG problem.

    Each population is a standard MFGProblem. Cross-coupling is defined
    by how each H_k evaluates its coupling function with the full
    density state.

    Parameters
    ----------
    populations : list[MFGProblem]
        K individual MFG problems, one per population.
    population_names : list[str] or None
        Optional names for each population (for logging/plotting).

    Examples
    --------
    >>> from mfgarchon import MFGProblem, Model
    >>> prob_A = MFGProblem(model=Model(hamiltonian=H_A), ...)
    >>> prob_B = MFGProblem(model=Model(hamiltonian=H_B), ...)
    >>> multi = MultiPopulationProblem(
    ...     populations=[prob_A, prob_B],
    ...     population_names=["commuters", "residents"],
    ... )
    >>> multi.K  # number of populations
    2
    """

    populations: list[MFGProblem]
    population_names: list[str] | None = None
    _: KW_ONLY  # the fields below are keyword-only (#2648)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if len(self.populations) < 1:
            raise ValueError("At least one population required.")
        if self.population_names is None:
            self.population_names = [f"pop_{k}" for k in range(len(self.populations))]
        if len(self.population_names) != len(self.populations):
            raise ValueError(
                f"population_names length ({len(self.population_names)}) "
                f"must match populations length ({len(self.populations)})."
            )

    @property
    def K(self) -> int:
        """Number of populations."""
        return len(self.populations)

    @property
    def T(self) -> float:
        """Terminal time (from first population)."""
        return self.populations[0].T

    @property
    def Nt(self) -> int:
        """Number of time steps (from first population)."""
        return self.populations[0].Nt

    def get_population(self, k: int) -> MFGProblem:
        """Get the k-th population's MFGProblem."""
        return self.populations[k]
