# Hooks Pattern Implementation Guide

**Status**: 🔧 **TECHNICAL SPECIFICATION**
**Date**: 2025-09-23
**Purpose**: Concrete implementation details for the hooks pattern

---

## 📋 **Core Hooks Architecture**

### **Base Hook System**
```python
# mfgarchon/hooks/base.py
from typing import Optional, Any, Protocol
from abc import ABC
from ..types import SpatialTemporalState, MFGResult

class SolverHooks(ABC):
    """
    Base class for solver customization hooks.

    Override any method to customize solver behavior. All methods
    are optional - only implement what you need.
    """

    def on_solve_start(self, initial_state: SpatialTemporalState) -> None:
        """
        Called once at the beginning of the solve process.

        Use this for:
        - Initialization of custom data structures
        - Starting timers or logging
        - Saving initial conditions

        Args:
            initial_state: Initial solution state
        """
        pass

    def on_iteration_start(self, state: SpatialTemporalState) -> None:
        """
        Called at the start of each solver iteration.

        Use this for:
        - Pre-iteration logging
        - Adaptive parameter updates
        - Custom preprocessing

        Args:
            state: Current solution state
        """
        pass

    def on_iteration_end(self, state: SpatialTemporalState) -> Optional[str]:
        """
        Called after each solver iteration completes.

        Use this for:
        - Progress monitoring and visualization
        - Custom convergence checks
        - Intermediate result saving
        - Algorithm state inspection

        Args:
            state: Updated solution state after iteration

        Returns:
            Control string:
            - None: Continue normally
            - "stop": Stop iteration early
            - "restart": Restart with modified conditions
        """
        pass

    def on_convergence_check(self, state: SpatialTemporalState) -> Optional[bool]:
        """
        Called during convergence checking.

        Use this for:
        - Custom convergence criteria
        - Override default convergence logic

        Args:
            state: Current solution state

        Returns:
            - None: Use default convergence check
            - True: Force convergence (stop iteration)
            - False: Force non-convergence (continue iteration)
        """
        pass

    def on_solve_end(self, result: MFGResult) -> MFGResult:
        """
        Called just before returning the final result.

        Use this for:
        - Post-processing of results
        - Adding custom metadata
        - Final visualization
        - Result validation

        Args:
            result: Final solver result

        Returns:
            Potentially modified result object
        """
        return result
```

### **Hook Composition System**
```python
# mfgarchon/hooks/composition.py
from typing import List, Optional
from .base import SolverHooks
from ..types import SpatialTemporalState, MFGResult

class MultiHook(SolverHooks):
    """Compose multiple hooks into one."""

    def __init__(self, *hooks: SolverHooks):
        self.hooks = list(hooks)

    def add_hook(self, hook: SolverHooks):
        """Add a hook to the composition."""
        self.hooks.append(hook)

    def on_solve_start(self, initial_state: SpatialTemporalState) -> None:
        for hook in self.hooks:
            hook.on_solve_start(initial_state)

    def on_iteration_start(self, state: SpatialTemporalState) -> None:
        for hook in self.hooks:
            hook.on_iteration_start(state)

    def on_iteration_end(self, state: SpatialTemporalState) -> Optional[str]:
        for hook in self.hooks:
            result = hook.on_iteration_end(state)
            if result:  # First hook to request control flow wins
                return result
        return None

    def on_convergence_check(self, state: SpatialTemporalState) -> Optional[bool]:
        for hook in self.hooks:
            result = hook.on_convergence_check(state)
            if result is not None:  # First definitive answer wins
                return result
        return None

    def on_solve_end(self, result: MFGResult) -> MFGResult:
        for hook in self.hooks:
            result = hook.on_solve_end(result)
        return result

class ConditionalHook(SolverHooks):
    """Execute hook only when condition is met."""

    def __init__(self, hook: SolverHooks, condition: callable):
        self.hook = hook
        self.condition = condition

    def on_iteration_end(self, state: SpatialTemporalState) -> Optional[str]:
        if self.condition(state):
            return self.hook.on_iteration_end(state)
        return None

# Usage examples:
# combined = MultiHook(DebugHook(), PlottingHook(), PerformanceHook())
# every_10th = ConditionalHook(PlottingHook(), lambda state: state.iteration % 10 == 0)
```

---

## 🎨 **Pre-built Hook Collections**

### **Debugging and Monitoring Hooks**
```python
# mfgarchon/hooks/debugging.py
import time
import logging
from typing import Optional, Dict, Any
from .base import SolverHooks
from ..types import SpatialTemporalState, MFGResult

class DebugHook(SolverHooks):
    """Comprehensive debugging information."""

    def __init__(self, log_level: str = "INFO"):
        self.logger = logging.getLogger("mfgarchon.solver")
        self.logger.setLevel(getattr(logging, log_level.upper()))
        self.start_time = None
        self.iteration_times = []

    def on_solve_start(self, initial_state: SpatialTemporalState) -> None:
        self.start_time = time.time()
        self.logger.info(f"Starting solve with initial residual: {initial_state.residual:.2e}")

    def on_iteration_end(self, state: SpatialTemporalState) -> Optional[str]:
        iteration_time = time.time() - (self.start_time + sum(self.iteration_times))
        self.iteration_times.append(iteration_time)

        self.logger.info(
            f"Iteration {state.iteration:3d}: "
            f"residual={state.residual:.2e}, "
            f"time={iteration_time:.3f}s"
        )

        # Check for numerical issues
        if not np.isfinite(state.residual):
            self.logger.error("Non-finite residual detected!")
            return "stop"

        return None

    def on_solve_end(self, result: MFGResult) -> MFGResult:
        total_time = time.time() - self.start_time
        avg_iteration_time = total_time / len(self.iteration_times) if self.iteration_times else 0

        self.logger.info(f"Solve completed in {total_time:.2f}s")
        self.logger.info(f"Average iteration time: {avg_iteration_time:.3f}s")
        self.logger.info(f"Converged: {result.converged}")

        return result

class PerformanceHook(SolverHooks):
    """Monitor memory and CPU usage."""

    def __init__(self):
        import psutil

        self.process = psutil.Process()
        self.memory_usage = []
        self.cpu_times = []

    def on_iteration_end(self, state: SpatialTemporalState) -> Optional[str]:
        memory_mb = self.process.memory_info().rss / 1024 / 1024
        cpu_percent = self.process.cpu_percent()

        self.memory_usage.append(memory_mb)
        self.cpu_times.append(cpu_percent)

        if state.iteration % 10 == 0:
            print(f"Memory: {memory_mb:.1f}MB, CPU: {cpu_percent:.1f}%")

        return None
```

### **Visualization Hooks**
```python
# mfgarchon/hooks/visualization.py
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, Tuple
from .base import SolverHooks
from ..types import SpatialTemporalState

class PlottingHook(SolverHooks):
    """Real-time visualization during solving."""

    def __init__(self, plot_every: int = 10, figsize: Tuple[int, int] = (12, 8)):
        self.plot_every = plot_every
        self.figsize = figsize
        self.residual_history = []

    def on_iteration_end(self, state: SpatialTemporalState) -> Optional[str]:
        self.residual_history.append(state.residual)

        if state.iteration % self.plot_every == 0:
            self._create_plots(state)

        return None

    def _create_plots(self, state: SpatialTemporalState):
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)

        # Value function
        im1 = axes[0, 0].imshow(state.u, aspect='auto', cmap='viridis')
        axes[0, 0].set_title(f'Value Function u(t,x) - Iter {state.iteration}')
        axes[0, 0].set_xlabel('Space')
        axes[0, 0].set_ylabel('Time')
        plt.colorbar(im1, ax=axes[0, 0])

        # Density function
        im2 = axes[0, 1].imshow(state.m, aspect='auto', cmap='plasma')
        axes[0, 1].set_title(f'Density m(t,x) - Iter {state.iteration}')
        axes[0, 1].set_xlabel('Space')
        axes[0, 1].set_ylabel('Time')
        plt.colorbar(im2, ax=axes[0, 1])

        # Convergence history
        axes[1, 0].semilogy(self.residual_history)
        axes[1, 0].set_title('Convergence History')
        axes[1, 0].set_xlabel('Iteration')
        axes[1, 0].set_ylabel('Residual')
        axes[1, 0].grid(True)

        # Final time profiles
        axes[1, 1].plot(state.u[-1, :], label='Value u(T,x)', alpha=0.7)
        axes[1, 1].plot(state.m[-1, :], label='Density m(T,x)', alpha=0.7)
        axes[1, 1].set_title('Final Time Profiles')
        axes[1, 1].set_xlabel('Space')
        axes[1, 1].legend()

        plt.tight_layout()
        plt.show()

class ConvergenceMonitorHook(SolverHooks):
    """Advanced convergence monitoring and early stopping."""

    def __init__(self,
                 patience: int = 20,
                 min_improvement: float = 1e-8,
                 smoothing_window: int = 5):
        self.patience = patience
        self.min_improvement = min_improvement
        self.smoothing_window = smoothing_window
        self.residual_history = []
        self.best_residual = float('inf')
        self.patience_counter = 0

    def on_iteration_end(self, state: SpatialTemporalState) -> Optional[str]:
        self.residual_history.append(state.residual)

        # Use smoothed residual for early stopping decisions
        if len(self.residual_history) >= self.smoothing_window:
            smoothed_residual = np.mean(self.residual_history[-self.smoothing_window:])

            if smoothed_residual < self.best_residual - self.min_improvement:
                self.best_residual = smoothed_residual
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            if self.patience_counter >= self.patience:
                print(f"Early stopping: no improvement for {self.patience} iterations")
                return "stop"

        return None
```

---

## 🔧 **Solver Integration**

### **Modified Solver Base Class**
```python
# mfgarchon/solvers/base.py
from abc import ABC, abstractmethod
from typing import Optional
from ..hooks import SolverHooks
from ..types import MFGProblem, MFGResult, SpatialTemporalState

class BaseSolver(ABC):
    """Base class for all MFG solvers with hooks support."""

    def __init__(self, config=None):
        self.config = config or {}
        self.max_iterations = config.get('max_iterations', 100) if config else 100
        self.tolerance = config.get('tolerance', 1e-6) if config else 1e-6

    def solve(self,
              problem: MFGProblem,
              hooks: Optional[SolverHooks] = None) -> MFGResult:
        """
        Solve the MFG problem with optional hooks.

        Args:
            problem: MFG problem instance
            hooks: Optional hooks for customization

        Returns:
            Solution result
        """
        # Initialize state
        state = self._initialize_state(problem)

        # Call solve start hook
        if hooks:
            hooks.on_solve_start(state)

        # Main iteration loop
        for iteration in range(self.max_iterations):
            state.iteration = iteration

            # Pre-iteration hook
            if hooks:
                hooks.on_iteration_start(state)

            # Perform one iteration step
            state = self._iteration_step(state, problem)

            # Post-iteration hook
            control_signal = None
            if hooks:
                control_signal = hooks.on_iteration_end(state)

            # Handle control signals
            if control_signal == "stop":
                break
            elif control_signal == "restart":
                state = self._initialize_state(problem)
                continue

            # Check convergence
            converged = self._check_convergence(state)
            if hooks:
                hook_convergence = hooks.on_convergence_check(state)
                if hook_convergence is not None:
                    converged = hook_convergence

            if converged:
                break

        # Create result
        result = self._create_result(state, problem)

        # Final hook
        if hooks:
            result = hooks.on_solve_end(result)

        return result

    @abstractmethod
    def _initialize_state(self, problem: MFGProblem) -> SpatialTemporalState:
        """Initialize solver state."""
        pass

    @abstractmethod
    def _iteration_step(self, state: SpatialTemporalState, problem: MFGProblem) -> SpatialTemporalState:
        """Perform one iteration step."""
        pass

    @abstractmethod
    def _check_convergence(self, state: SpatialTemporalState) -> bool:
        """Check if solver has converged."""
        pass

    @abstractmethod
    def _create_result(self, state: SpatialTemporalState, problem: MFGProblem) -> MFGResult:
        """Create final result object."""
        pass
```

### **Example: Fixed Point Solver with Hooks**
```python
# mfgarchon/solvers/fixed_point.py
import numpy as np
from .base import BaseSolver
from ..types import SpatialTemporalState, MFGResult

class FixedPointSolver(BaseSolver):
    """Fixed-point iteration solver with hooks support."""

    def _initialize_state(self, problem) -> SpatialTemporalState:
        # Initialize u and m from problem
        u_init = problem.get_initial_value_function()
        m_init = problem.get_initial_density()

        return SpatialTemporalState(
            u=u_init,
            m=m_init,
            iteration=0,
            residual=float('inf'),
            metadata={}
        )

    def _iteration_step(self, state, problem) -> SpatialTemporalState:
        # Solve HJB equation
        u_new = self._solve_hjb_step(state.u, state.m, problem)

        # Solve Fokker-Planck equation
        m_new = self._solve_fp_step(u_new, state.m, problem)

        # Compute residual
        residual = np.linalg.norm(u_new - state.u) + np.linalg.norm(m_new - state.m)

        return SpatialTemporalState(
            u=u_new,
            m=m_new,
            iteration=state.iteration + 1,
            residual=residual,
            metadata=state.metadata
        )

    def _check_convergence(self, state) -> bool:
        return state.residual < self.tolerance

    def _create_result(self, state, problem) -> MFGResult:
        return MFGResult(
            u=state.u,
            m=state.m,
            converged=state.residual < self.tolerance,
            iterations=state.iteration,
            final_residual=state.residual,
            problem=problem
        )
```

---

## 📚 **Usage Examples**

### **Basic Usage**
```python
from mfgarchon.solvers import FixedPointSolver
from mfgarchon.hooks import DebugHook, PlottingHook, MultiHook

# Create solver
solver = FixedPointSolver(config={'tolerance': 1e-8})

# Simple debugging
debug_hook = DebugHook(log_level="INFO")
result = solver.solve(problem, hooks=debug_hook)

# Combine multiple hooks
combined_hooks = MultiHook(
    DebugHook(),
    PlottingHook(plot_every=5),
    PerformanceHook()
)
result = solver.solve(problem, hooks=combined_hooks)
```

### **Custom Hook Development**
```python
class ResearchHook(SolverHooks):
    """Custom hook for research-specific analysis."""

    def __init__(self, save_dir: str):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)

    def on_iteration_end(self, state) -> Optional[str]:
        # Save intermediate results
        if state.iteration % 20 == 0:
            np.save(self.save_dir / f"u_iter_{state.iteration}.npy", state.u)
            np.save(self.save_dir / f"m_iter_{state.iteration}.npy", state.m)

        # Custom convergence criterion
        if state.iteration > 10:
            u_change = np.max(np.abs(state.u - self.prev_u))
            if u_change < 1e-10:
                print("Custom convergence achieved!")
                return "stop"

        self.prev_u = state.u.copy()
        return None

# Usage
research_hook = ResearchHook("./results/experiment_1/")
result = solver.solve(problem, hooks=research_hook)
```

This implementation provides a complete, extensible hooks system that balances simplicity for basic users with powerful customization for advanced research applications.
