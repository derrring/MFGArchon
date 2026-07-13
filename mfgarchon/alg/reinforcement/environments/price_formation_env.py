r"""
Price Formation Mean Field Game Environment.

Market making environment where agents set bid/ask spreads to manage
inventory while accounting for aggregate market liquidity. Demonstrates:
- Financial market dynamics
- Inventory risk management
- Price formation via mean field coupling
- Order flow aggregation

Mathematical Formulation:
- State: $(q, p, \dot{p}, m_{\text{depth}}) \in \mathbb{R}^4$
- Action: $(\delta_{\text{bid}}, \delta_{\text{ask}}) \in [0, \delta_{\max}]^2$ (spread controls)
- Dynamics: $p' = p + \sigma \epsilon + \lambda \cdot \text{imbalance}$, $q' = q + \Delta q$
- Individual Cost: -PnL + inventory penalty + spread cost
- Mean Field Cost: liquidity depletion at crowded price levels
- Goal: Maximize profit while managing inventory risk

Use Cases:
- Market making and high-frequency trading
- Optimal execution strategies
- Liquidity provision modeling
- Multi-agent market simulation

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


class PriceFormationEnv(ContinuousMFGEnvBase):
    r"""
    Price Formation Mean Field Game Environment.

    Market making with bid/ask spread control and inventory management.

    State Space:
    - Inventory: $q \in [-Q_{\max}, Q_{\max}]$
    - Mid-price: $p \in [p_{\min}, p_{\max}]$
    - Price velocity: $\dot{p}$ (recent price change)
    - Market depth: $m_{\text{depth}}$ (aggregated liquidity measure)

    Action Space:
    - Bid spread: $\delta_{\text{bid}} \in [0, \delta_{\max}]$
    - Ask spread: $\delta_{\text{ask}} \in [0, \delta_{\max}]$

    Dynamics:
    - Price: $p_{t+1} = p_t + \sigma \sqrt{dt} \epsilon + \lambda \cdot \text{imbalance}$
    - Inventory: $q_{t+1} = q_t + \Delta q_t$ (from filled orders)
    - Velocity: $\dot{p}_{t+1} = (p_{t+1} - p_t) / dt$

    Reward:
    - PnL: profit from trades at spread
    - Inventory penalty: $-\gamma q^2$ (quadratic risk)
    - Spread cost: wider spreads reduce fill probability
    - Market depth penalty: based on mean field liquidity

    Mean Field Coupling:
    - Liquidity depletion at popular price levels
    - Price impact from aggregate order flow
    - Reduced fill rates in crowded regions

    Termination:
    - Early: inventory exceeds limits (bankruptcy)
    - Truncation: max_steps reached
    """

    def __init__(
        self,
        num_agents: int = 100,
        q_max: float = 10.0,
        p_min: float = 90.0,
        p_max: float = 110.0,
        delta_max: float = 1.0,
        price_volatility: float = 0.5,
        price_impact: float = 0.1,
        inventory_penalty: float = 0.5,
        spread_cost: float = 0.1,
        liquidity_penalty: float = 0.3,
        fill_probability_base: float = 0.8,
        dt: float = 0.1,
        max_steps: int = 200,
        noise_std: float = 0.1,
        population_bins: int = 50,
    ):
        """
        Initialize Price Formation environment.

        Args:
            num_agents: Number of agents in population
            q_max: Maximum inventory (absolute value)
            p_min: Minimum price level
            p_max: Maximum price level
            delta_max: Maximum spread width
            price_volatility: Standard deviation of price noise (σ)
            price_impact: Impact coefficient for order imbalance (λ)
            inventory_penalty: Weight for inventory risk penalty (γ)
            spread_cost: Cost for wide spreads (reduces fill probability)
            liquidity_penalty: Penalty for trading at crowded price levels
            fill_probability_base: Base probability of order fill
            dt: Time step size
            max_steps: Maximum episode length
            noise_std: Standard deviation of dynamics noise
            population_bins: Number of bins for population histogram
        """
        self.q_max = q_max
        self.p_min = p_min
        self.p_max = p_max
        self.delta_max = delta_max
        self.price_volatility = price_volatility
        self.price_impact = price_impact
        self.inventory_penalty = inventory_penalty
        self.spread_cost = spread_cost
        self.liquidity_penalty = liquidity_penalty
        self.fill_probability_base = fill_probability_base

        # Initial mid-price (set in reset)
        self.initial_price = (p_min + p_max) / 2

        # Initialize base class
        # State: (q, p, dp/dt, market_depth)
        super().__init__(
            num_agents=num_agents,
            state_dim=4,
            action_dim=2,
            action_bounds=(0.0, delta_max),
            population_bins=population_bins,
            dt=dt,
            max_steps=max_steps,
            noise_std=noise_std,
        )

    def _get_state_bounds(self) -> tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]]:
        """
        Get state space bounds.

        Returns:
            (low, high) arrays with bounds for (q, p, dp/dt, market_depth)
        """
        low = np.array(
            [
                -self.q_max,  # inventory
                self.p_min,  # price
                -2.0,  # price velocity (bounded)
                0.0,  # market depth (always non-negative)
            ],
            dtype=np.float32,
        )
        high = np.array(
            [
                self.q_max,  # inventory
                self.p_max,  # price
                2.0,  # price velocity
                1.0,  # market depth (normalized)
            ],
            dtype=np.float32,
        )
        return low, high

    def _sample_initial_states(self) -> NDArray[np.floating[Any]]:
        """
        Sample initial states for all agents.

        Initial distribution:
        - Inventory: zero (start flat)
        - Price: initial_price with small noise
        - Price velocity: zero
        - Market depth: uniform initial liquidity

        Returns:
            Array of shape (num_agents, 4) with initial states
        """
        # Start with zero inventory
        inventories = np.zeros((self.num_agents, 1), dtype=np.float32)

        # Price near initial with small noise
        prices = self.initial_price + self.rng.normal(0, 0.1, size=(self.num_agents, 1)).astype(np.float32)

        # Zero initial velocity
        velocities = np.zeros((self.num_agents, 1), dtype=np.float32)

        # Uniform initial market depth
        market_depths = np.ones((self.num_agents, 1), dtype=np.float32) * 0.5

        # Stack into (num_agents, 4) array
        states = np.concatenate([inventories, prices, velocities, market_depths], axis=1)
        return states

    def _drift(
        self,
        state: NDArray[np.floating[Any]],
        action: NDArray[np.floating[Any]],
        population: NDArray[np.floating[Any]],
    ) -> NDArray[np.floating[Any]]:
        """
        Compute drift term in dynamics.

        Price formation dynamics:
        - Price evolves with noise + order imbalance impact
        - Inventory changes from filled orders
        - Velocity tracks price changes

        Args:
            state: Current state (q, p, dp/dt, market_depth)
            action: Current action (delta_bid, delta_ask)
            population: Population histogram (used for order flow)

        Returns:
            Drift vector (dq/dt, dp/dt, d(dp/dt)/dt, 0)
        """
        q = state[0]
        delta_bid, delta_ask = action[0], action[1]

        # Order fill probability depends on spread width
        # Tighter spreads → higher fill probability
        fill_prob_bid = self.fill_probability_base * np.exp(-self.spread_cost * delta_bid)
        fill_prob_ask = self.fill_probability_base * np.exp(-self.spread_cost * delta_ask)

        # Order flow: buy if inventory low, sell if high (simplified mean-reversion)
        target_inventory = 0.0
        inventory_signal = target_inventory - q

        # Filled orders (stochastic, but use expected value for drift)
        dq_dt = fill_prob_bid * np.sign(inventory_signal) * np.maximum(0, inventory_signal) - fill_prob_ask * np.sign(
            -inventory_signal
        ) * np.maximum(0, -inventory_signal)

        # Price impact from order imbalance (aggregate across population)
        # Simplified: assume uniform distribution for now
        order_imbalance = dq_dt  # Simplified: own contribution
        dp_dt = self.price_impact * order_imbalance

        # Price velocity acceleration (mean reversion)
        price_velocity = state[2]
        d_price_vel_dt = -0.5 * price_velocity  # Damping

        # Market depth doesn't drift (updated from population)
        dmarket_depth_dt = 0.0

        return np.array([dq_dt, dp_dt, d_price_vel_dt, dmarket_depth_dt], dtype=np.float32)

    def _noise_mask(self) -> NDArray[np.floating[Any]]:
        """Market depth (dim 3) has zero drift -- it is updated from the population, not diffused --
        so it must NOT receive Brownian noise (Issue #1600), which would inject a spurious random
        walk into a quantity the dynamics treat as externally set."""
        mask = np.ones(self.state_dim, dtype=np.float64)
        mask[3] = 0.0
        return mask

    def _individual_reward(
        self,
        state: NDArray[np.floating[Any]],
        action: NDArray[np.floating[Any]],
        next_state: NDArray[np.floating[Any]],
    ) -> float:
        r"""
        Compute individual reward (negative cost).

        Components:
        1. PnL: profit from spread (if orders filled)
        2. Inventory penalty: $-\gamma q^2$
        3. Spread cost: penalty for wide spreads

        Args:
            state: Current state
            action: Current action
            next_state: Next state (used for PnL calculation)

        Returns:
            Individual reward (negative cost + profits)
        """
        # Extract components
        q = next_state[0]
        delta_bid, delta_ask = action[0], action[1]

        # PnL from spread (simplified: assume some fills)
        fill_prob_bid = self.fill_probability_base * np.exp(-self.spread_cost * delta_bid)
        fill_prob_ask = self.fill_probability_base * np.exp(-self.spread_cost * delta_ask)

        # Average spread earned
        avg_spread = (delta_bid + delta_ask) / 2
        pnl = (fill_prob_bid + fill_prob_ask) / 2 * avg_spread

        # Inventory penalty (quadratic risk)
        inventory_cost = self.inventory_penalty * q**2

        # Spread cost (opportunity cost of wide spreads)
        spread_penalty = self.spread_cost * (delta_bid + delta_ask) / 2

        # Total reward
        return float(pnl - inventory_cost - spread_penalty)

    def compute_mean_field_coupling(
        self, state: NDArray[np.floating[Any]], population: NDArray[np.floating[Any]]
    ) -> float:
        r"""
        Compute mean field interaction term.

        Liquidity depletion at crowded price levels:
        - Higher population density at price level → reduced liquidity
        - Penalty for trading at crowded prices

        Args:
            state: Current state (q, p, dp/dt, market_depth)
            population: Population histogram (binned by price)

        Returns:
            Mean field coupling term (negative liquidity penalty)
        """
        # Extract price
        p = state[1]

        # Map price to population bin
        price_range = self.p_max - self.p_min
        bin_idx = int((p - self.p_min) / price_range * self.population_bins)
        bin_idx = np.clip(bin_idx, 0, self.population_bins - 1)

        # Local density at this price level
        local_density = population[bin_idx]

        # Liquidity penalty (higher density → less liquidity)
        liquidity_cost = self.liquidity_penalty * local_density

        return float(-liquidity_cost)

    def get_population_state(self) -> NDArray[np.floating[Any]]:
        """
        Get population distribution as histogram over price space.

        Bins population by price coordinate only (ignores inventory and velocity).

        Returns:
            Flattened histogram of shape (population_bins,)
            Represents binned density over price domain
        """
        if self.agent_states is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        # Extract prices from all agent states
        prices = self.agent_states[:, 1]  # (num_agents,)

        # Create histogram over price space
        histogram, _ = np.histogram(prices, bins=self.population_bins, range=(self.p_min, self.p_max))

        # Normalize to probability distribution
        histogram = histogram.astype(np.float32)
        total = histogram.sum()
        if total > 0:
            histogram = histogram / total
        else:
            histogram = np.ones(self.population_bins, dtype=np.float32) / self.population_bins

        return histogram

    def _is_terminated(self, state: NDArray[np.floating[Any]]) -> bool:
        """
        Check if episode should terminate early.

        Terminates when inventory exceeds limits (bankruptcy risk).

        Args:
            state: Current state (q, p, dp/dt, market_depth)

        Returns:
            True if inventory limit exceeded, False otherwise
        """
        q = state[0]
        return bool(np.abs(q) >= self.q_max * 0.95)  # 95% of limit
