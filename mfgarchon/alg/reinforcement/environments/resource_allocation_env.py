r"""
Resource Allocation Mean Field Game Environment.

Portfolio optimization environment where agents allocate capital across assets
while accounting for congestion effects and market impact. Demonstrates:
- Constrained optimization (simplex constraints)
- Portfolio dynamics with stochastic returns
- Congestion in popular assets
- Strategic asset allocation

Mathematical Formulation:
- State: $(w, v) \in \Delta^n \times \mathbb{R}_+^n$ (allocation + asset values)
- Action: $\Delta w \in \mathbb{R}^n$ with $\sum \Delta w_i = 0$ (rebalancing)
- Dynamics: $v' = v \odot (1 + \mu + \sigma \epsilon)$, $w' = \Pi_\Delta(w + \Delta w)$
- Individual Cost: -returns + risk penalty + transaction cost
- Mean Field Cost: congestion in overcrowded assets (reduced returns)
- Goal: Maximize risk-adjusted returns while avoiding crowded trades

Use Cases:
- Portfolio optimization and asset allocation
- Resource competition modeling
- Congestion games in finance
- Multi-agent trading strategies

Author: MFGarchon Team
Date: October 2025
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

try:
    from mfgarchon.alg.reinforcement.environments.continuous_mfg_env_base import ContinuousMFGEnvBase

    BASE_AVAILABLE = True
except ImportError:
    BASE_AVAILABLE = False
    ContinuousMFGEnvBase = object  # type: ignore


class ResourceAllocationEnv(ContinuousMFGEnvBase):
    r"""
    Resource Allocation Mean Field Game Environment.

    Portfolio optimization with congestion effects in popular assets.

    State Space:
    - Allocation: $w = (w_1, \ldots, w_n) \in \Delta^n$ (simplex constraint: $\sum w_i = 1$, $w_i \geq 0$)
    - Asset values: $v = (v_1, \ldots, v_n) \in \mathbb{R}_+^n$

    Action Space:
    - Rebalancing: $\Delta w \in [-\Delta_{\max}, \Delta_{\max}]^n$ with $\sum \Delta w_i = 0$

    Dynamics:
    - Asset values: $v_{t+1} = v_t \odot (1 + \mu_t + \sigma \sqrt{dt} \epsilon_t)$
    - Allocation: $w_{t+1} = \Pi_\Delta(w_t + \Delta w_t)$ (project to simplex)
    - Expected returns: $\mu_t = \mu_{\text{base}} - \alpha \cdot m_t$ (congestion penalty)

    Reward:
    - Portfolio return: $w^T (v_{t+1} - v_t) / v_t$
    - Risk penalty: $-\lambda w^T \Sigma w$ (quadratic in allocation)
    - Transaction cost: $-\kappa \|\Delta w\|_1$ (proportional to rebalancing)
    - Congestion penalty: based on mean field distribution

    Mean Field Coupling:
    - Congestion in popular assets reduces expected returns
    - Market impact proportional to population concentration
    - Penalty: $\alpha \cdot \text{density}_i$ for asset $i$

    Termination:
    - Early: allocation becomes invalid (numerical issues)
    - Truncation: max_steps reached
    """

    def __init__(
        self,
        num_agents: int = 100,
        num_assets: int = 3,
        delta_max: float = 0.2,
        base_return: float = 0.05,
        return_volatility: float = 0.1,
        risk_penalty: float = 0.5,
        transaction_cost: float = 0.01,
        congestion_penalty: float = 0.3,
        dt: float = 0.1,
        max_steps: int = 200,
        noise_std: float = 0.05,
        population_bins: int = 50,
    ):
        """
        Initialize Resource Allocation environment.

        Args:
            num_agents: Number of agents in population
            num_assets: Number of assets to allocate across
            delta_max: Maximum allocation change per asset
            base_return: Base expected return for assets (μ_base)
            return_volatility: Volatility of asset returns (σ)
            risk_penalty: Weight for portfolio risk penalty (λ)
            transaction_cost: Cost per unit of rebalancing (κ)
            congestion_penalty: Penalty for congestion in popular assets (α)
            dt: Time step size
            max_steps: Maximum episode length
            noise_std: Standard deviation of dynamics noise
            population_bins: Number of bins for population histogram
        """
        self.num_assets = num_assets
        self.delta_max = delta_max
        self.base_return = base_return
        self.return_volatility = return_volatility
        self.risk_penalty = risk_penalty
        self.transaction_cost = transaction_cost
        self.congestion_penalty = congestion_penalty

        # Covariance matrix for returns (simplified: diagonal)
        self.covariance = np.eye(num_assets, dtype=np.float32) * (return_volatility**2)

        # Initialize base class
        # State: (w_1, ..., w_n, v_1, ..., v_n)
        super().__init__(
            num_agents=num_agents,
            state_dim=2 * num_assets,
            action_dim=num_assets,
            action_bounds=(-delta_max, delta_max),
            population_bins=population_bins,
            dt=dt,
            max_steps=max_steps,
            noise_std=noise_std,
        )

    def _get_state_bounds(self) -> tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]]:
        """
        Get state space bounds.

        Returns:
            (low, high) arrays with bounds for (w_1, ..., w_n, v_1, ..., v_n)
        """
        low = np.concatenate(
            [
                np.zeros(self.num_assets, dtype=np.float32),  # allocations >= 0
                np.zeros(self.num_assets, dtype=np.float32),  # asset values >= 0
            ]
        )
        high = np.concatenate(
            [
                np.ones(self.num_assets, dtype=np.float32),  # allocations <= 1
                np.full(self.num_assets, 10.0, dtype=np.float32),  # asset values (arbitrary upper bound)
            ]
        )
        return low, high

    def _sample_initial_states(self) -> NDArray[np.floating[Any]]:
        """
        Sample initial states for all agents.

        Initial distribution:
        - Allocations: uniform on simplex (equal weights with noise)
        - Asset values: all start at 1.0

        Returns:
            Array of shape (num_agents, 2*num_assets) with initial states
        """
        # Start with equal allocation + small noise
        allocations = np.ones((self.num_agents, self.num_assets), dtype=np.float32) / self.num_assets
        noise = self.rng.normal(0, 0.05, size=(self.num_agents, self.num_assets)).astype(np.float32)
        allocations = allocations + noise

        # Project to simplex
        allocations = np.maximum(allocations, 0)
        allocations = allocations / allocations.sum(axis=1, keepdims=True)

        # All assets start at value 1.0
        values = np.ones((self.num_agents, self.num_assets), dtype=np.float32)

        # Stack into (num_agents, 2*num_assets) array
        states = np.concatenate([allocations, values], axis=1)
        return states

    def _project_to_simplex(self, weights: NDArray[np.floating[Any]]) -> NDArray[np.floating[Any]]:
        """
        Project weights onto probability simplex.

        Uses efficient sorting algorithm for simplex projection.

        Args:
            weights: Array of weights (may violate simplex constraints)

        Returns:
            Projected weights satisfying sum=1 and all >= 0
        """
        # Clip negative values
        w = np.maximum(weights, 0)

        # Normalize to sum to 1
        total = w.sum()
        if total > 0:
            w = w / total
        else:
            # If all zero, use uniform distribution
            w = np.ones_like(weights) / len(weights)

        return w.astype(np.float32)

    def _drift(
        self,
        state: NDArray[np.floating[Any]],
        action: NDArray[np.floating[Any]],
        population: NDArray[np.floating[Any]],
    ) -> NDArray[np.floating[Any]]:
        """
        Compute drift term in dynamics.

        Portfolio dynamics:
        - Asset values evolve with expected return (adjusted for congestion)
        - Allocations rebalance toward action target

        Args:
            state: Current state (w_1, ..., w_n, v_1, ..., v_n)
            action: Current action (Δw_1, ..., Δw_n)
            population: Population histogram (used for congestion)

        Returns:
            Drift vector (dw_1/dt, ..., dw_n/dt, dv_1/dt, ..., dv_n/dt)
        """
        # Extract components
        values = state[self.num_assets :]

        # Compute congestion-adjusted returns
        # Higher population concentration → lower returns
        congestion = self._compute_congestion_from_population(population)
        adjusted_returns = np.maximum(self.base_return - self.congestion_penalty * congestion, -0.5)  # Floor at -50%

        # Asset value dynamics (drift only, noise added by base class)
        dv_dt = values * adjusted_returns

        # Allocation dynamics (action is rebalancing Δw)
        # Enforce sum(Δw) = 0 constraint
        delta_w = action - action.mean()  # Zero-sum constraint
        dw_dt = delta_w / self.dt  # Rate of change

        # Combine
        drift = np.concatenate([dw_dt, dv_dt])
        return drift.astype(np.float32)

    def _compute_congestion_from_population(self, population: NDArray[np.floating[Any]]) -> NDArray[np.floating[Any]]:
        """
        Compute congestion level for each asset from population distribution.

        Simplified: assume uniform distribution over assets, use population bin as proxy.

        Args:
            population: Population histogram

        Returns:
            Congestion array of shape (num_assets,)
        """
        # Simplified: map population bins to asset congestion
        # In full implementation, would compute histogram over allocation space
        # For now, use first num_assets bins as proxy
        if len(population) >= self.num_assets:
            congestion = population[: self.num_assets]
        else:
            congestion = np.zeros(self.num_assets, dtype=np.float32)

        return congestion

    def _individual_reward(
        self,
        state: NDArray[np.floating[Any]],
        action: NDArray[np.floating[Any]],
        next_state: NDArray[np.floating[Any]],
    ) -> float:
        r"""
        Compute individual reward (negative cost).

        Components:
        1. Portfolio return: $w^T \cdot \text{returns}$
        2. Risk penalty: $-\lambda w^T \Sigma w$
        3. Transaction cost: $-\kappa \|\Delta w\|_1$

        Args:
            state: Current state
            action: Current action
            next_state: Next state (used for return calculation)

        Returns:
            Individual reward (negative cost + returns)
        """
        # Extract components
        weights = state[: self.num_assets]
        values = state[self.num_assets :]
        next_values = next_state[self.num_assets :]

        # Portfolio returns
        returns = (next_values - values) / np.maximum(values, 1e-6)  # Avoid division by zero
        portfolio_return = float(np.dot(weights, returns))

        # Risk penalty (quadratic in allocation)
        risk_cost = self.risk_penalty * float(weights @ self.covariance @ weights)

        # Transaction cost (L1 norm of action)
        transaction_penalty = self.transaction_cost * float(np.abs(action).sum())

        # Total reward
        return portfolio_return - risk_cost - transaction_penalty

    def compute_mean_field_coupling(
        self, state: NDArray[np.floating[Any]], population: NDArray[np.floating[Any]]
    ) -> float:
        r"""
        Compute mean field interaction term.

        Congestion penalty for allocating to popular assets:
        - Higher population concentration → stronger penalty
        - Penalty: $\alpha \sum_i w_i \cdot \text{density}_i$

        Args:
            state: Current state (w_1, ..., w_n, v_1, ..., v_n)
            population: Population histogram (binned by allocation)

        Returns:
            Mean field coupling term (negative congestion penalty)
        """
        # Extract allocation
        weights = state[: self.num_assets]

        # Compute congestion
        congestion = self._compute_congestion_from_population(population)

        # Congestion penalty (weighted by allocation)
        congestion_cost = self.congestion_penalty * float(np.dot(weights, congestion))

        return -congestion_cost

    def get_population_state(self) -> NDArray[np.floating[Any]]:
        """
        Get population distribution as histogram over allocation space.

        Bins population by first asset allocation only (simplified).

        Returns:
            Flattened histogram of shape (population_bins,)
            Represents binned density over first asset weight
        """
        if self.agent_states is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        # Extract first asset allocation from all agent states
        first_asset_weights = self.agent_states[:, 0]  # (num_agents,)

        # Create histogram
        histogram, _ = np.histogram(first_asset_weights, bins=self.population_bins, range=(0, 1))

        # Normalize to probability distribution
        histogram = histogram.astype(np.float32)
        total = histogram.sum()
        if total > 0:
            histogram = histogram / total
        else:
            histogram = np.ones(self.population_bins, dtype=np.float32) / self.population_bins

        return histogram

    def _postprocess_next_states(self, next_states: NDArray[np.floating[Any]]) -> NDArray[np.floating[Any]]:
        """Enforce the portfolio constraints on EVERY advanced agent: project each agent's allocation
        block to the probability simplex and clamp asset values >= 0.

        Overrides the base identity hook rather than the whole ``step()`` (Issue #1600). The base
        ``_advance_population`` stays the single Euler-Maruyama driver, so the population-frozen and
        constant-dim-noise fixes reach the resource env too -- the prior ``step()`` override was a
        full copy of the base step that advanced only agent 0 and re-drew noise on every dim.
        """
        out = np.array(next_states, dtype=float, copy=True)
        for i in range(out.shape[0]):
            out[i, : self.num_assets] = self._project_to_simplex(out[i, : self.num_assets])
        out[:, self.num_assets :] = np.maximum(out[:, self.num_assets :], 0.0)
        return out

    def _compute_reward(
        self,
        state: NDArray[np.floating[Any]],
        action: NDArray[np.floating[Any]],
        next_state: NDArray[np.floating[Any]],
        population: NDArray[np.floating[Any]],
    ) -> float:
        """
        Compute total reward (individual + mean field).

        Args:
            state: Current state
            action: Current action
            next_state: Next state
            population: Population histogram

        Returns:
            Total reward
        """
        individual = self._individual_reward(state, action, next_state)
        mean_field = self.compute_mean_field_coupling(next_state, population)
        return individual + mean_field

    def _is_terminated(self, state: NDArray[np.floating[Any]]) -> bool:
        """
        Check if episode should terminate early.

        Terminates if allocation becomes invalid (numerical issues).

        Args:
            state: Current state (w_1, ..., w_n, v_1, ..., v_n)

        Returns:
            True if allocation invalid, False otherwise
        """
        weights = state[: self.num_assets]

        # Check for invalid allocations (NaN or Inf)
        # Don't terminate on simplex violation - projection handles it
        return bool(np.any(np.isnan(weights)) or np.any(np.isinf(weights)))
