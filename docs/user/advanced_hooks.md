# Advanced Hooks Guide - Expert API

**For algorithm researchers and advanced users who need full control**

The hooks system provides unprecedented access to internal solver algorithms, allowing you to customize, monitor, and extend every aspect of the solution process.

## Hooks System Overview

Hooks are callback functions that execute at specific points during solving:

```python
┌─────────────────────────────────────────────────────────────┐
│                    Solver Execution                        │
├─────────────────────────────────────────────────────────────┤
│  on_solve_start()                                          │
│  ├─ on_iteration_start()                                   │
│  │  ├─ on_hjb_start() → on_hjb_step() → on_hjb_end()     │
│  │  ├─ on_fp_start() → on_fp_step() → on_fp_end()       │
│  │  ├─ on_coupling_update()                              │
│  │  └─ on_iteration_end()                                │
│  ├─ on_convergence_check()                                │
│  └─ on_solve_end()                                        │
└─────────────────────────────────────────────────────────────┘
```

## Built-in Hook Classes

### Monitoring and Debugging

```python
from mfgarchon.hooks import (
    DebugHook, ProgressHook, PerformanceHook,
    ConvergenceAnalysisHook, VisualizationHook
)

# Real-time debugging information
debug_hook = DebugHook(
    log_level="INFO",
    save_intermediate=True,
    output_dir="debug_output/"
)

# Progress monitoring with custom formatting
progress_hook = ProgressHook(
    update_frequency=10,
    show_eta=True,
    custom_format="Iter {iteration:4d}: {residual:.2e} | ETA: {eta}"
)

# Performance profiling
perf_hook = PerformanceHook(
    profile_memory=True,
    profile_compute=True,
    save_profile="profile.json"
)

# Advanced convergence analysis
convergence_hook = ConvergenceAnalysisHook(
    save_history=True,
    detect_stagnation=True,
    analyze_spectral_radius=True
)

# Live visualization during solving
viz_hook = VisualizationHook(
    update_frequency=20,
    plot_density=True,
    plot_value=True,
    save_animation="evolution.gif"
)
```

### Multiple Hook Usage

```python
from mfgarchon.hooks import HookCollection

# Combine multiple hooks
hooks = HookCollection([
    DebugHook(log_level="INFO"),
    ProgressHook(update_frequency=5),
    PerformanceHook(profile_memory=True),
    ConvergenceAnalysisHook(detect_stagnation=True)
])

result = solver.solve(problem, hooks=hooks)
```

## Custom Hook Development

### Basic Custom Hook

```python
from mfgarchon.hooks import SolverHooks

class CustomMonitoringHook(SolverHooks):
    def __init__(self):
        self.iteration_times = []
        self.residual_history = []
        self.mass_conservation = []

    def on_solve_start(self, initial_state):
        print(f"🚀 Starting solve with {initial_state.grid_size} grid points")
        self.start_time = time.time()

    def on_iteration_start(self, state):
        self.iter_start_time = time.time()

    def on_iteration_end(self, state):
        iter_time = time.time() - self.iter_start_time
        self.iteration_times.append(iter_time)
        self.residual_history.append(state.residual)

        # Check mass conservation
        mass = np.trapz(state.density, state.x_grid)
        self.mass_conservation.append(abs(mass - 1.0))

        if state.iteration % 10 == 0:
            print(f"Iter {state.iteration:3d}: "
                  f"residual={state.residual:.2e}, "
                  f"mass_error={self.mass_conservation[-1]:.2e}, "
                  f"time={iter_time:.3f}s")

    def on_solve_end(self, result):
        total_time = time.time() - self.start_time
        avg_iter_time = sum(self.iteration_times) / len(self.iteration_times)

        print(f"✅ Solve completed in {total_time:.2f}s")
        print(f"📊 Average iteration time: {avg_iter_time:.3f}s")
        print(f"📈 Final mass conservation error: {self.mass_conservation[-1]:.2e}")

        # Add analysis to result
        result.iteration_times = self.iteration_times
        result.mass_conservation_history = self.mass_conservation
        return result

# Use custom hook
hook = CustomMonitoringHook()
result = solver.solve(problem, hooks=hook)

# Access custom analysis
import matplotlib.pyplot as plt
plt.plot(result.iteration_times)
plt.title("Iteration Times")
plt.show()
```

### Algorithm Modification Hooks

```python
class AdaptiveDampingHook(SolverHooks):
    """Dynamically adjust damping based on convergence behavior."""

    def __init__(self, initial_damping=0.8, min_damping=0.1, max_damping=0.95):
        self.damping = initial_damping
        self.min_damping = min_damping
        self.max_damping = max_damping
        self.residual_history = []
        self.damping_history = []

    def on_iteration_end(self, state):
        self.residual_history.append(state.residual)

        if len(self.residual_history) >= 3:
            # Check convergence trend
            recent_residuals = self.residual_history[-3:]

            if recent_residuals[-1] > recent_residuals[-2]:
                # Residual increased - reduce damping for stability
                self.damping = max(self.min_damping, self.damping * 0.9)
                print(f"Reducing damping to {self.damping:.3f}")

            elif all(r1 > r2 for r1, r2 in zip(recent_residuals[:-1], recent_residuals[1:])):
                # Consistent improvement - can increase damping for speed
                self.damping = min(self.max_damping, self.damping * 1.05)

        self.damping_history.append(self.damping)

        # Update solver damping (if supported)
        if hasattr(state.solver, 'set_damping'):
            state.solver.set_damping(self.damping)

        return state
```

### Custom Stopping Criteria

```python
class EarlyStoppingHook(SolverHooks):
    """Stop early based on custom criteria."""

    def __init__(self, patience=50, min_improvement=1e-8):
        self.patience = patience
        self.min_improvement = min_improvement
        self.best_residual = float('inf')
        self.wait_count = 0

    def on_convergence_check(self, state):
        if state.residual < self.best_residual - self.min_improvement:
            self.best_residual = state.residual
            self.wait_count = 0
        else:
            self.wait_count += 1

        if self.wait_count >= self.patience:
            print(f"Early stopping: no improvement for {self.patience} iterations")
            state.should_stop = True

        return state
```

### Data Collection and Analysis

```python
class DataCollectionHook(SolverHooks):
    """Collect detailed data for post-hoc analysis."""

    def __init__(self, collect_frequency=10):
        self.collect_frequency = collect_frequency
        self.snapshots = []

    def on_iteration_end(self, state):
        if state.iteration % self.collect_frequency == 0:
            snapshot = {
                'iteration': state.iteration,
                'residual': state.residual,
                'density': state.density.copy(),
                'value_function': state.value_function.copy(),
                'velocity_field': self._compute_velocity(state),
                'energy': self._compute_energy(state),
                'entropy': self._compute_entropy(state)
            }
            self.snapshots.append(snapshot)

    def _compute_velocity(self, state):
        """Compute velocity field from value function."""
        return -np.gradient(state.value_function, state.dx)

    def _compute_energy(self, state):
        """Compute total kinetic energy."""
        velocity = self._compute_velocity(state)
        return 0.5 * np.trapz(state.density * velocity**2, state.x_grid)

    def _compute_entropy(self, state):
        """Compute entropy of density distribution."""
        density_safe = np.maximum(state.density, 1e-12)
        return -np.trapz(state.density * np.log(density_safe), state.x_grid)

    def export_data(self, filename):
        """Export collected data to file."""
        import pickle
        with open(filename, 'wb') as f:
            pickle.dump(self.snapshots, f)

    def create_analysis(self):
        """Create analysis plots from collected data."""
        iterations = [s['iteration'] for s in self.snapshots]
        energies = [s['energy'] for s in self.snapshots]
        entropies = [s['entropy'] for s in self.snapshots]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        ax1.plot(iterations, energies)
        ax1.set_title("Energy Evolution")
        ax1.set_ylabel("Total Energy")

        ax2.plot(iterations, entropies)
        ax2.set_title("Entropy Evolution")
        ax2.set_ylabel("Entropy")
        ax2.set_xlabel("Iteration")

        return fig
```

## Algorithm-Specific Hooks

### HJB Equation Hooks

```python
class HJBCustomizationHook(SolverHooks):
    """Customize HJB equation solving."""

    def on_hjb_start(self, state):
        print(f"Solving HJB equation at iteration {state.iteration}")

    def on_hjb_step(self, state, x_point, current_value):
        # Custom processing at each spatial point
        if self._needs_special_treatment(x_point):
            # Apply custom boundary conditions or modifications
            return self._apply_custom_logic(x_point, current_value, state)
        return current_value

    def on_hjb_end(self, state):
        # Post-process HJB solution
        state.value_function = self._smooth_solution(state.value_function)

    def _needs_special_treatment(self, x_point):
        # Example: special treatment near boundaries
        return x_point < 0.1 or x_point > 0.9

    def _apply_custom_logic(self, x_point, value, state):
        # Custom boundary behavior
        if x_point < 0.1:
            return value * 0.5  # Reduce value near left boundary
        return value

    def _smooth_solution(self, solution):
        # Apply smoothing filter
        from scipy.ndimage import gaussian_filter1d
        return gaussian_filter1d(solution, sigma=1.0)
```

### Fokker-Planck Equation Hooks

```python
class FPCustomizationHook(SolverHooks):
    """Customize Fokker-Planck equation solving."""

    def on_fp_start(self, state):
        self.initial_mass = np.trapz(state.density, state.x_grid)

    def on_fp_step(self, state, density_update):
        # Apply conservation constraints during FP steps
        corrected_update = self._enforce_mass_conservation(
            density_update, state.density, state.x_grid
        )
        return corrected_update

    def on_fp_end(self, state):
        # Ensure non-negativity and mass conservation
        state.density = np.maximum(state.density, 0.0)
        current_mass = np.trapz(state.density, state.x_grid)
        state.density *= self.initial_mass / current_mass

    def _enforce_mass_conservation(self, update, current_density, x_grid):
        projected_density = current_density + update
        projected_mass = np.trapz(projected_density, x_grid)

        # Project back to preserve mass
        if abs(projected_mass - self.initial_mass) > 1e-10:
            correction = (self.initial_mass - projected_mass) / len(x_grid)
            update += correction

        return update
```

## Advanced Hook Patterns

### Hook Composition

```python
class CompositeHook(SolverHooks):
    """Compose multiple hooks with priority ordering."""

    def __init__(self, hooks, priorities=None):
        self.hooks = hooks
        self.priorities = priorities or [1] * len(hooks)

    def _execute_hooks(self, method_name, *args, **kwargs):
        """Execute hooks in priority order."""
        hook_methods = [(p, getattr(hook, method_name, None))
                       for p, hook in zip(self.priorities, self.hooks)]

        # Sort by priority (higher priority first)
        hook_methods.sort(key=lambda x: x[0], reverse=True)

        result = args[0] if args else None  # Usually the state

        for priority, method in hook_methods:
            if method is not None:
                result = method(*args, **kwargs) or result

        return result

    def on_iteration_end(self, state):
        return self._execute_hooks('on_iteration_end', state)

    def on_solve_end(self, result):
        return self._execute_hooks('on_solve_end', result)
```

### Conditional Hooks

```python
class ConditionalHook(SolverHooks):
    """Execute hooks based on conditions."""

    def __init__(self, hook, condition):
        self.hook = hook
        self.condition = condition

    def on_iteration_end(self, state):
        if self.condition(state):
            return self.hook.on_iteration_end(state)
        return state

# Usage examples
debug_hook = DebugHook()

# Only debug if convergence is slow
slow_convergence = lambda state: state.iteration > 100 and state.residual > 1e-5
conditional_debug = ConditionalHook(debug_hook, slow_convergence)

# Only visualize every 50 iterations
periodic_viz = ConditionalHook(
    VisualizationHook(),
    lambda state: state.iteration % 50 == 0
)
```

### Stateful Hook Networks

```python
class NetworkHook(SolverHooks):
    """Hook that can communicate with external systems."""

    def __init__(self, network_config):
        self.config = network_config
        self.session = self._create_session()

    def on_iteration_end(self, state):
        # Send data to monitoring system
        self._send_metrics({
            'iteration': state.iteration,
            'residual': state.residual,
            'timestamp': time.time()
        })

        # Check for external commands
        command = self._check_external_commands()
        if command == 'pause':
            input("Solver paused. Press Enter to continue...")
        elif command == 'save':
            state.save_checkpoint('external_checkpoint.pkl')

    def _send_metrics(self, data):
        # Send to monitoring dashboard
        requests.post(f"{self.config['url']}/metrics", json=data)

    def _check_external_commands(self):
        # Check for external control signals
        response = requests.get(f"{self.config['url']}/commands")
        return response.json().get('command')
```

## Performance Optimization Hooks

### Memory Management

```python
class MemoryOptimizationHook(SolverHooks):
    """Optimize memory usage during solving."""

    def __init__(self, memory_limit_gb=8):
        self.memory_limit = memory_limit_gb * 1024**3  # Convert to bytes
        self.gc_frequency = 10

    def on_iteration_end(self, state):
        if state.iteration % self.gc_frequency == 0:
            import gc
            import psutil

            # Check memory usage
            memory_usage = psutil.Process().memory_info().rss
            if memory_usage > self.memory_limit * 0.8:
                print(f"High memory usage: {memory_usage / 1024**3:.1f}GB")

                # Force garbage collection
                gc.collect()

                # Clear intermediate data if available
                if hasattr(state, 'clear_intermediates'):
                    state.clear_intermediates()

        return state
```

### Adaptive Precision

```python
class AdaptivePrecisionHook(SolverHooks):
    """Adapt numerical precision based on convergence."""

    def __init__(self, high_precision_threshold=1e-6):
        self.threshold = high_precision_threshold
        self.using_high_precision = False

    def on_convergence_check(self, state):
        if (state.residual < self.threshold and
            not self.using_high_precision):

            print("Switching to high precision arithmetic")
            # Switch to higher precision (if supported)
            if hasattr(state.solver, 'set_precision'):
                state.solver.set_precision('double')
                self.using_high_precision = True

        return state
```

## Research and Development Hooks

### Experimental Algorithm Testing

```python
class ExperimentalMethodHook(SolverHooks):
    """Test experimental methods alongside standard ones."""

    def __init__(self):
        self.experimental_results = []
        self.standard_results = []

    def on_hjb_end(self, state):
        # Save standard result
        standard_solution = state.value_function.copy()
        self.standard_results.append(standard_solution)

        # Test experimental method
        experimental_solution = self._experimental_hjb_solver(state)
        self.experimental_results.append(experimental_solution)

        # Compare and potentially switch
        if self._experimental_is_better(standard_solution, experimental_solution):
            print(f"Experimental method outperformed at iteration {state.iteration}")
            state.value_function = experimental_solution

        return state

    def _experimental_hjb_solver(self, state):
        # Implement experimental HJB solving method
        return self._novel_algorithm(state.value_function, state.density)

    def _experimental_is_better(self, standard, experimental):
        # Define criteria for "better"
        standard_energy = self._compute_energy(standard)
        experimental_energy = self._compute_energy(experimental)
        return experimental_energy < standard_energy * 0.99
```

## Integration with External Tools

### Machine Learning Integration

```python
class MLAcceleratedHook(SolverHooks):
    """Use ML to accelerate convergence."""

    def __init__(self, model_path):
        import torch
        self.model = torch.load(model_path)
        self.prediction_history = []

    def on_iteration_start(self, state):
        # Use ML model to predict next iteration
        if state.iteration > 5:  # Need history for prediction
            features = self._extract_features(state)
            prediction = self.model(features)

            # Use prediction as initialization
            state.value_function = 0.7 * state.value_function + 0.3 * prediction
            self.prediction_history.append(prediction)

    def _extract_features(self, state):
        # Convert state to ML model features
        import torch
        return torch.tensor([
            state.residual,
            state.iteration,
            *state.value_function,
            *state.density
        ], dtype=torch.float32)
```

## Best Practices

### Hook Development Guidelines

1. **Keep hooks focused**: Each hook should have a single responsibility
2. **Handle errors gracefully**: Don't crash the solver
3. **Minimize performance impact**: Expensive operations should be optional
4. **Document side effects**: Clearly document what your hook modifies
5. **Provide configuration**: Make behavior customizable

### Example Production Hook

```python
class ProductionMonitoringHook(SolverHooks):
    """Production-ready monitoring hook with robust error handling."""

    def __init__(self, config_file=None):
        self.config = self._load_config(config_file)
        self.logger = self._setup_logging()
        self.metrics_collector = self._setup_metrics()
        self.start_time = None

    def on_solve_start(self, initial_state):
        try:
            self.start_time = time.time()
            self.logger.info(f"Starting MFG solve: {initial_state}")
            self.metrics_collector.increment('solves_started')
        except Exception as e:
            self.logger.error(f"Error in on_solve_start: {e}")

    def on_iteration_end(self, state):
        try:
            if state.iteration % self.config.get('log_frequency', 10) == 0:
                self.logger.info(
                    f"Iteration {state.iteration}: "
                    f"residual={state.residual:.2e}"
                )

            # Send metrics to monitoring system
            self.metrics_collector.gauge('current_residual', state.residual)
            self.metrics_collector.gauge('current_iteration', state.iteration)

        except Exception as e:
            self.logger.error(f"Error in on_iteration_end: {e}")

        return state

    def on_solve_end(self, result):
        try:
            solve_time = time.time() - self.start_time
            success = result.converged

            self.logger.info(
                f"Solve completed: success={success}, "
                f"time={solve_time:.2f}s, "
                f"iterations={result.iterations}"
            )

            self.metrics_collector.histogram('solve_time', solve_time)
            self.metrics_collector.increment(
                'solves_successful' if success else 'solves_failed'
            )

        except Exception as e:
            self.logger.error(f"Error in on_solve_end: {e}")

        return result

    def _load_config(self, config_file):
        """Load configuration with sensible defaults."""
        default_config = {'log_frequency': 10, 'log_level': 'INFO'}

        if config_file:
            import json
            with open(config_file) as f:
                user_config = json.load(f)
            default_config.update(user_config)

        return default_config

    def _setup_logging(self):
        """Setup structured logging."""
        import logging
        logger = logging.getLogger('mfgarchon.hooks')
        logger.setLevel(self.config['log_level'])
        return logger

    def _setup_metrics(self):
        """Setup metrics collection."""
        # Return appropriate metrics collector
        # (StatsD, Prometheus, etc.)
        return MockMetricsCollector()
```

The hooks system provides unlimited extensibility while maintaining clean interfaces. Use it to implement custom algorithms, gather research data, optimize performance, or integrate with external systems.

## What's Next?

- **Need examples?** → Check [Advanced Examples](../../examples/advanced/)
- **Building custom solvers?** → See [HJB Solver Selection Guide](HJB_SOLVER_SELECTION_GUIDE.md)
- **Performance optimization?** → Read [Multidimensional MFG Guide](multidimensional_mfg_guide.md)
- **Understanding hooks?** → See [Hooks Guide](guides/hooks.md)
