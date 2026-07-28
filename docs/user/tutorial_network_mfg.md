# Mean Field Games on Networks: Complete Implementation Guide

## Overview

This guide provides comprehensive documentation for the Network MFG implementation in MFGArchon, which extends Mean Field Games from continuous domains to discrete network/graph structures.

## Table of Contents

1. [Mathematical Framework](#mathematical-framework)
2. [Architecture Overview](#architecture-overview)  
3. [Network Geometry System](#network-geometry-system)
4. [Network MFG Problems](#network-mfg-problems)
5. [Network Solvers](#network-solvers)
6. [Visualization Tools](#visualization-tools)
7. [Usage Examples](#usage-examples)
8. [Performance Considerations](#performance-considerations)
9. [API Reference](#api-reference)

## Mathematical Framework

### Continuous vs Network MFG

**Continuous MFG:**
- State space: $\Omega \subset \mathbb{R}^d$ (continuous domain)
- HJB equation: $\frac{\partial u}{\partial t} + H(x, m, \nabla u, t) = 0$
- FP equation: $\frac{\partial m}{\partial t} - \text{div}(m \nabla H_p) - \frac{\sigma^2}{2}\Delta m = 0$

**Network MFG:**
- State space: $G = (V, E)$ (discrete graph with $N$ nodes)
- HJB equation: $\frac{\partial u_i}{\partial t} + H_i(m, \nabla_G u, t) = 0$ for node $i$
- FP equation: $\frac{\partial m_i}{\partial t} - \text{div}_G(m \nabla_G H_p) - \frac{\sigma^2}{2}\Delta_G m = 0$

### Network Operators

**Graph Gradient:** For function $u: V \to \mathbb{R}$:
$$(\nabla_G u)_{ij} = u_j - u_i \quad \text{for edge } (i,j) \in E$$

**Graph Divergence:** For edge flow $f: E \to \mathbb{R}$:
$$(\text{div}_G f)_i = \sum_{j \sim i} w_{ij}(f_{ji} - f_{ij})$$

**Graph Laplacian:** For adjacency matrix $A$ and degree matrix $D$:
$$\Delta_G = D - A$$

### Network Hamiltonian

The network Hamiltonian at node $i$ typically takes the form:
$$H_i(m, p, t) = \sum_{j \sim i} c_{ij}(t) + V_i(t) + F_i(m_i, t)$$

where:
- $c_{ij}(t)$: cost of moving from node $i$ to $j$
- $V_i(t)$: potential at node $i$  
- $F_i(m_i, t)$: congestion/coupling function

## Architecture Overview

```
mfgarchon/
├── geometry/network_geometry.py      # Network structures and topologies
├── core/network_mfg_problem.py       # Network MFG problem formulation
├── alg/hjb_solvers/hjb_network.py    # HJB solvers for networks
├── alg/fp_solvers/fp_network.py      # FP solvers for networks  
├── alg/mfg_solvers/network_mfg_solver.py  # Complete network MFG solvers
├── visualization/network_plots.py     # Network visualization tools
└── examples/
    ├── basic/network_mfg_example.py   # Basic usage example
    └── advanced/network_mfg_comparison.py  # Advanced comparison
```

## Network Geometry System

### NetworkData Class

Core data structure for network information:

```python
@dataclass
class NetworkData:
    adjacency_matrix: csr_matrix      # (N, N) sparse adjacency matrix
    num_nodes: int                    # Number of nodes
    num_edges: int                    # Number of edges
    network_type: NetworkType         # Type of network structure
    node_positions: np.ndarray        # (N, d) coordinates [optional]
    edge_weights: np.ndarray          # Edge weights [optional]
    laplacian_matrix: csr_matrix      # Graph Laplacian
    degree_matrix: csr_matrix         # Node degree matrix
    incidence_matrix: csr_matrix      # Node-edge incidence matrix
```

### Supported Network Types

1. **Grid Networks** (`GridNetwork`)
   - Regular 2D lattices with optional periodic boundaries
   - Suitable for spatial diffusion problems
   - Well-defined geometric structure

2. **Random Networks** (`RandomNetwork`)
   - Erdős–Rényi random graphs
   - Uniform connection probability
   - Good for homogeneous populations

3. **Scale-Free Networks** (`ScaleFreeNetwork`)
   - Barabási–Albert preferential attachment
   - Power-law degree distribution
   - Models social networks, transportation hubs

### Creating Networks

```python
from mfgarchon.geometry.graph.network import create_network

# Grid network
grid_network = create_network("grid", num_nodes=25, width=5, height=5)

# Random network
random_network = create_network("random", num_nodes=50, connection_prob=0.1)

# Scale-free network
scale_free_network = create_network("scale_free", num_nodes=50, num_edges_per_node=3)
```

## Network MFG Problems

### NetworkMFGProblem Class

Extends the base MFG framework to network structures:

```python
class NetworkMFGProblem(MFGProblem):
    def __init__(self,
                 network_geometry: BaseNetworkGeometry,
                 T: float = 1.0,
                 Nt: int = 100,
                 components: Optional[NetworkMFGComponents] = None)
```

### NetworkMFGComponents

Defines network-specific MFG components:

```python
@dataclass
class NetworkMFGComponents:
    # Network-specific functions
    hamiltonian_func: Optional[Callable] = None              # H(node, neighbors, m, p, t)
    node_potential_func: Optional[Callable] = None           # V(node, t)
    edge_cost_func: Optional[Callable] = None                # Cost of edge traversal
    
    # Initial and terminal conditions
    initial_node_density_func: Optional[Callable] = None     # m_0(node)
    terminal_node_value_func: Optional[Callable] = None      # u_T(node)
    
    # Coupling and interaction
    node_interaction_func: Optional[Callable] = None         # Local node interactions
    congestion_func: Optional[Callable] = None               # Congestion effects
    
    # Parameters
    diffusion_coefficient: float = 1.0
    drift_coefficient: float = 1.0
```

### Factory Functions

Quick problem creation:

```python
# Grid MFG problem
problem = create_grid_mfg_problem(
    width=5, height=5, T=2.0, Nt=50,
    terminal_node_value_func=my_terminal_func,
    node_interaction_func=my_congestion_func
)

# Random MFG problem
problem = create_random_mfg_problem(
    num_nodes=50, connection_prob=0.15, T=1.0, Nt=40,
    initial_node_density_func=my_initial_func
)
```

## Network Solvers

### HJB Network Solvers

**NetworkHJBSolver** supports multiple time discretization schemes:

```python
from mfgarchon.alg.hjb_solvers.hjb_network import create_network_hjb_solver

# Explicit scheme
hjb_solver = create_network_hjb_solver(problem, "explicit", cfl_factor=0.5)

# Implicit scheme  
hjb_solver = create_network_hjb_solver(problem, "implicit", max_iterations=100)

# Policy iteration
hjb_solver = create_network_hjb_solver(problem, "policy_iteration")
```

**Key Features:**
- Discrete HJB equation handling
- Network-specific stability constraints
- Policy iteration for optimal control
- Boundary condition management

### FP Network Solvers

**NetworkFPSolver** handles density evolution on networks:

```python
from mfgarchon.alg.fp_solvers.fp_network import create_network_fp_solver

# Explicit scheme
fp_solver = create_network_fp_solver(problem, "explicit")

# Flow-based solver (mass conservative)
fp_solver = create_network_fp_solver(problem, "flow")

# Upwind scheme
fp_solver = create_network_fp_solver(problem, "upwind")
```

**Key Features:**
- Mass drift reported rather than imposed (Issue #1683)
- Flow-based density evolution
- Network diffusion operators
- Stability-preserving schemes

### Complete Network MFG Solvers

**NetworkFixedPointIterator** combines HJB and FP solvers:

```python
from mfgarchon.alg.mfg_solvers.network_mfg_solver import create_network_mfg_solver

# Standard fixed point iteration
solver = create_network_mfg_solver(
    problem,
    solver_type="fixed_point",
    hjb_solver_type="explicit",
    fp_solver_type="explicit",
    damping_factor=0.6
)

# Flow-based solver
solver = create_network_mfg_solver(
    problem,
    solver_type="flow",
    hjb_solver_type="implicit",
    fp_solver_type="flow"
)

# Solve the problem
U, M, convergence_info = solver.solve(
    max_iterations=50,
    tolerance=1e-5,
    verbose=True
)
```

## Visualization Tools

### NetworkMFGVisualizer

Comprehensive visualization for network MFG:

```python
from mfgarchon.visualization.network_plots import create_network_visualizer

# Create visualizer
visualizer = create_network_visualizer(problem=problem)

# Plot network topology
fig1 = visualizer.plot_network_topology(
    node_values=initial_density,
    title="Network with Initial Density"
)

# Plot density evolution
fig2 = visualizer.plot_density_evolution(
    M, times, selected_nodes=[0, 5, 10],
    title="Density Evolution at Key Nodes"
)

# Create flow animation
anim = visualizer.create_flow_animation(U, M, times)

# Analysis dashboard
dashboard = visualizer.plot_network_statistics_dashboard(convergence_info)
```

**Visualization Features:**
- Interactive network topology plots
- Density and value function evolution
- Flow animations and analysis
- Convergence monitoring dashboards
- Network statistics visualization

## Usage Examples

### Basic Congestion Game

```python
import numpy as np
from mfgarchon import create_grid_mfg_problem
from mfgarchon.alg.mfg_solvers.network_mfg_solver import create_network_mfg_solver

# Create 5x5 grid problem
def terminal_reward(node):
    i, j = divmod(node, 5)
    # Reward corner nodes
    if (i, j) in [(0,0), (0,4), (4,0), (4,4)]:
        return -5.0  # High reward (negative cost)
    return 0.0

def congestion_cost(node, m, t):
    return 2.0 * m[node]**2  # Quadratic congestion

problem = create_grid_mfg_problem(
    width=5, height=5, T=2.0, Nt=50,
    terminal_node_value_func=terminal_reward,
    node_interaction_func=congestion_cost
)

# Solve
solver = create_network_mfg_solver(problem, solver_type="fixed_point")
U, M, info = solver.solve(max_iterations=30, tolerance=1e-4)

print(f"Converged: {info['converged']}")
print(f"Final error: {info['final_error']:.2e}")
```

### Network Comparison Study

```python
from mfgarchon.geometry.graph.network import create_network
from mfgarchon.core.network_mfg_problem import NetworkMFGProblem

# Compare different network topologies
networks = {
    'grid': create_network("grid", 25, width=5),
    'random': create_network("random", 25, connection_prob=0.15),
    'scale_free': create_network("scale_free", 25, num_edges_per_node=3)
}

results = {}
for name, network in networks.items():
    # Create identical problems on different networks
    problem = NetworkMFGProblem(network, T=1.0, Nt=40)
    solver = create_network_mfg_solver(problem)
    U, M, info = solver.solve(verbose=False)
    
    results[name] = {
        'convergence': info['converged'],
        'time': info['execution_time'],
        'final_density_variance': np.var(M[-1, :])
    }

print("Network Comparison Results:")
for name, result in results.items():
    print(f"{name}: converged={result['convergence']}, "
          f"time={result['time']:.2f}s, "
          f"density_var={result['final_density_variance']:.3f}")
```

## Performance Considerations

### Computational Complexity

**Network Size Scaling:**
- Node operations: $O(N)$ per time step
- Edge operations: $O(E)$ per time step  
- Matrix operations: $O(N^2)$ for dense, $O(N + E)$ for sparse

**Time Complexity:**
- Explicit schemes: $O(T \cdot N_t \cdot (N + E))$
- Implicit schemes: $O(T \cdot N_t \cdot N^{1.5})$ (sparse solve)
- Policy iteration: $O(T \cdot N_t \cdot K \cdot N^2)$ where $K$ is policy iterations

### Memory Usage

**Storage Requirements:**
- Adjacency matrix: $O(E)$ (sparse) or $O(N^2)$ (dense)
- Solution arrays: $O(N_t \cdot N)$ for U and M
- Network operators: $O(N + E)$

### Optimization Tips

1. **Use Sparse Matrices:** Always use `scipy.sparse` for large networks
2. **Choose Appropriate Solvers:** 
   - Explicit for small networks or loose tolerances
   - Implicit for stiff problems or tight tolerances
   - Flow-based for mass conservation requirements
3. **Time Step Selection:** Follow CFL conditions for stability
4. **Network Preprocessing:** Precompute network operators when possible

## API Reference

### Core Classes

```python
# Network geometry
class NetworkData: ...
class GridNetwork(BaseNetworkGeometry): ...
class RandomNetwork(BaseNetworkGeometry): ...
class ScaleFreeNetwork(BaseNetworkGeometry): ...

# MFG problems
class NetworkMFGProblem(MFGProblem): ...
class NetworkMFGComponents: ...

# Solvers  
class NetworkHJBSolver(BaseHJBSolver): ...
class NetworkFPSolver(BaseFPSolver): ...
class NetworkFixedPointIterator(MFGSolver): ...

# Visualization
class NetworkMFGVisualizer: ...
```

### Factory Functions

```python
# Network creation
create_network(network_type, num_nodes, **kwargs) -> BaseNetworkGeometry

# Problem creation
create_grid_mfg_problem(width, height, **kwargs) -> NetworkMFGProblem
create_random_mfg_problem(num_nodes, connection_prob, **kwargs) -> NetworkMFGProblem
create_scale_free_mfg_problem(num_nodes, num_edges_per_node, **kwargs) -> NetworkMFGProblem

# Solver creation
create_network_hjb_solver(problem, solver_type, **kwargs) -> NetworkHJBSolver
create_network_fp_solver(problem, solver_type, **kwargs) -> NetworkFPSolver  
create_network_mfg_solver(problem, solver_type, **kwargs) -> NetworkFixedPointIterator

# Visualization
create_network_visualizer(problem=None, network_data=None) -> NetworkMFGVisualizer
```

### Key Parameters

**Network Creation:**
- `num_nodes`: Number of nodes in network
- `connection_prob`: Edge probability for random networks  
- `num_edges_per_node`: Edges per node for scale-free networks
- `width`, `height`: Grid dimensions
- `periodic`: Periodic boundary conditions for grids

**Solver Configuration:**
- `max_iterations`: Maximum Picard iterations
- `tolerance`: Convergence tolerance
- `damping_factor`: Picard damping parameter (0.5-0.8 recommended)
- `cfl_factor`: CFL stability factor for explicit schemes
- `diffusion_coefficient`: Network diffusion strength

**Visualization Options:**
- `interactive`: Use Plotly for interactive plots
- `node_size_scale`: Scale factor for node sizes
- `selected_nodes`: Specific nodes to plot in evolution graphs
- `save_path`: Path to save visualizations

## Advanced Topics

### Custom Network Types

To implement custom network types, extend `BaseNetworkGeometry`:

```python
class CustomNetwork(BaseNetworkGeometry):
    def create_network(self, **kwargs) -> NetworkData:
        # Implement custom network generation
        # Return NetworkData object
        pass
    
    def compute_distance_matrix(self) -> np.ndarray:
        # Implement distance computation
        pass
```

### Custom Hamiltonians

Define problem-specific Hamiltonians:

```python
def custom_hamiltonian(node, neighbors, m, p, t):
    # Implement custom Hamiltonian logic
    # Example: distance-based costs
    total_cost = 0.0
    for neighbor in neighbors:
        distance = compute_distance(node, neighbor)  # Custom distance
        control_cost = 0.5 * distance * (p[neighbor] - p[node])**2
        total_cost += control_cost
    
    # Add congestion and potential terms
    total_cost += node_potential(node, t) + congestion_function(node, m, t)
    return total_cost

# Use in problem definition
components = NetworkMFGComponents(hamiltonian_func=custom_hamiltonian)
```

### Advanced Solver Customization

Create custom solver configurations:

```python
# Custom HJB solver with specific parameters
hjb_solver = NetworkHJBSolver(
    problem, 
    scheme="semi_implicit",
    cfl_factor=0.3,
    max_iterations=100,
    tolerance=1e-8
)

# Custom FP solver with flow tracking
fp_solver = NetworkFlowFPSolver(
    problem,
    diffusion_coefficient=0.1,
)

# Combine in custom MFG solver
mfg_solver = NetworkFixedPointIterator(
    problem,
    hjb_solver_type="custom",  # Would need implementation
    fp_solver_type="custom",   # Would need implementation
    damping_factor=0.7
)
```

This completes the comprehensive Network MFG implementation guide. The system provides a full framework for solving Mean Field Games on discrete network structures with extensive customization options and visualization capabilities.
