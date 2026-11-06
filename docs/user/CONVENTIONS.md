# Conventions

MFGArchon commits to one convention per question, and this file is where each one is stated. Code,
docstrings and tests cite it **by section name**, never by line number, and do not restate the rule
they cite. `AGENTS.md` points here rather than repeating it.

Each rule keeps at most a one-line reason, where the rule would otherwise read as arbitrary. The
history behind a rule — what was refuted, which incident produced it — lives in the issue that
settled it, not here.

Where the code does not yet meet a convention, the section says so and points at the issue that
tracks it — **#2429** lists them, linking each gap that has an issue of its own — rather than stating
a rule the code breaks.

---

## 1. The library minimises

Every cost functional in MFGArchon is **minimised**. There is no optimisation-direction parameter,
no `sense` keyword, and no maximisation mode.

A maximisation problem is negated before it reaches the library:

```python
# maximise E[ reward(X_t) ]  is stated to MFGArchon as
def reward(x):
    return -sum((xi - 0.5) ** 2 for xi in x)  # what the agents would maximise


H = SeparableHamiltonian(
    control_cost=QuadraticControlCost(lambda_=1.0),
    potential=lambda t, x: -reward(x),
)
```

Stating the problem as a minimisation is the user's step, and deliberately theirs: every sign inside
the library is then fixed.

---

## 2. The HJB equation, and the sign of every term in it

### The canonical equation

$$
-\,\partial_t u \;+\; H(t, x, \nabla u, m) \;-\; \operatorname{tr}\!\bigl(A\,D^{2}u\bigr) \;=\; 0,
\qquad u(T, x) = g(x)
$$

$u(t,x)$ is the value function (cost-to-go), $H$ is the **Hamiltonian** and not the running cost,
$g$ is the terminal cost, and $A$ is § *Volatility and diffusion*'s second-order coefficient. In one
dimension with scalar volatility the third term is $\tfrac{\sigma^{2}}{2}\,\partial_{xx}u$. This is
the viscosity-solution form, and solver residuals are assembled in exactly this arrangement.

### One rule fixes every sign

> **A positive value of any term makes that state more expensive.**

$$
H(t, x, p, m) \;=\; H_{\text{control}}(t, x, p) \;-\; V(t, x) \;-\; f(m)
$$

so, moving the data to the right-hand side,

$$
-\,\partial_t u \;+\; H_{\text{control}} \;-\; \operatorname{tr}\!\bigl(A\,D^{2}u\bigr)
\;=\; V \;+\; f \;+\; S .
$$

A large potential $V$, a crowded coupling $f$ and a positive source $S$ each raise $u$. They are
costs, in the same sense as the terminal condition $g$.

- **$V$ is a cost.** Agents avoid large $V$; an attractive well is a *negative* potential.
- **$f(m)$ is a congestion cost.** Agents avoid crowds where $f$ increases in $m$, and
  $\partial H / \partial m = -f'(m)$.
- **`source_term_hjb` keeps its sign** and is the same sign as everything beside it.

**Three cost channels enter $H$.** `potential` is $V$; `state_penalty` is a cost-signed soft wall
composed into $V$ (`V + state_penalty_scale * state_penalty`); `coupling` is $f$. `state_penalty` and
`state_penalty_scale` are read **once, at construction** — assigning either afterwards does nothing.

The library's `SeparableHamiltonian` takes the coupling as $f(m)$, a function of the density alone.
An $x$-dependent interaction goes through $V$ or through `source_term_hjb`.

The FP drift does not move with these signs: $\alpha^* = -D_pH$ depends only on the control part.

### The Hamiltonian is where the sign lives

`SeparableHamiltonian.__call__` returns the full combination above (`HamiltonianBase.__call__` is
abstract). No solver keeps its own sign for a part of it: a constant $c$ added to $V$ moves $u(0)$ by
$+cT$ in every solver family that reads the problem's $H$.

How the solvers use it differs, and the difference is part of the convention:

- The Newton and explicit assemblies — FDM, GFDM, WENO, the weak-form family and `NetworkHJBSolver` —
  add $H$ whole.
- HJB semi-Lagrangian's update assembles the running cost at the optimal control from the same object, as
  $p \cdot \partial_p H - H$. `diffusion_method='canonical_cs'` hand-codes the quadratic running cost in its objective,
  so it takes only a `SeparableHamiltonian` with a `QuadraticControlCost`, and refuses any other Hamiltonian when built.
- The policy-iteration assemblies assemble control cost $+\,V + f$ as a running cost:
  `HJBGFDMSolver(inner_solver="howard")` takes each part from the same Hamiltonian object;
  `NetworkPolicyIterationHJBSolver` recomputes the built-in quadratic control cost itself and refuses a
  custom Hamiltonian.
- `HJBHowardSolver` used directly takes its running cost and feedback law from the caller and never
  reads the problem's $H$.
- On a network, a custom `hamiltonian_func` is the whole cost-signed $H$; passing
  `node_potential_func` or `node_interaction_func` alongside it raises.

### Lasry–Lions monotonicity is a condition on $f$, never on $H$

$$
\int \bigl(f(x, m_1) - f(x, m_2)\bigr)\,\mathrm{d}(m_1 - m_2)(x) \;\ge\; 0
$$

State it on the coupling function. Because $f$ sits inside $H$ with a minus sign, the same condition
read off $H$ has the opposite sign, and reading it off $H$ is how a well-posedness claim gets
inverted.

**Aggregating (crowd-seeking) models are legitimate.** They violate this condition, uniqueness is
not claimed for them, and the library does not check it either way.

### The source term

- **`source_term_hjb(t, x, v, m)` receives the density *slice* $m(t,\cdot)$**, not the full
  $(N_t+1, N_x)$ trajectory. A callback that slices internally slices twice.
- **Every HJB solver accepts a source and adds it, except three, and those raise
  `NotImplementedError` rather than drop it.** The two network HJB solvers do not name `source_term`.
  `HJBHowardSolver` names it and refuses it: a source enters Howard through its constructor's
  `running_cost`, and `HJBGFDMSolver(inner_solver="howard")` does that conversion for a composed
  source.
- **A $p$-dependent coupling must not be routed through the source.** The FP drift is derived from
  $D_pH$, and a $p$-dependence the Hamiltonian does not see cannot appear in it.
- For a repulsive interaction energy $F[m]$, with its strength inside $F$, the source is
  $S = +\,\delta F/\delta m$, cost-signed like everything else here. `create_lions_source(F, ...)`
  returns exactly this.
- **`Model(coupling_cost=...)` is that source, bound by the library** (#1642). The value is a
  specification, `InteractionCost(kernel)`, `PotentialCost(V)` or their sum. `MFGProblem` binds it
  on its grid and measure (`problem.coupling_energy`), and the HJB carries
  $\delta F/\delta m$ at each density slice, summed with the Hamiltonian's own $f$ and with
  `source_term_hjb`. On a `TensorProductGrid` the problem binds it. HJB-GFDM, which evaluates its
  source at its collocation points, reads $\delta F/\delta m$ there: $V(x)$, or the grid's quadrature
  applied at $x$. A weak-form HJB solver refuses a problem bound on a grid. Off a grid the HJB solver
  binds it on its own points and weights: HJB-FEM P1 on its dofs with the consistent mass matrix's
  row sums, on a triangle or tetrahedron mesh or a generated `Mesh1D` (#1642 stage 1b-2). HJB-FEM
  refuses it at construction at a higher order, and on a mesh with no element volumes, such as a quad
  or hexahedron mesh; the meshless Galerkin solver refuses it at construction. A solver or loop that
  binds none refuses it by name at solve time. On a network it is refused when the problem is built.
- **`Model(terminal_coupling=...)` is its terminal counterpart G** (#1642). It takes the same
  specifications. The terminal condition is then $u(T, x) = g(x) + \delta G/\delta m$ at $m(T, \cdot)$,
  the density of the HJB solve it is the terminal of, cost-signed as $g$ is;
  `problem.terminal_value(m_T, binding=...)` is its one owner. On a `TensorProductGrid` the problem
  binds it (`problem.terminal_energy`). Off a grid, as $F$ is, it is held by the problem
  (`problem.declared_terminal_coupling`) and bound by the HJB solver beside $F$, on the same points
  and weights: HJB-FEM P1 on its dofs (#1642, stage 1b's G line). HJB-FEM P2 and the meshless
  Galerkin solver refuse it at construction, and a solver or loop that binds none refuses it by name
  where it reads $u(T)$. It is refused by name on a network and in `MultiPopulationIterator`.

### Worked example — 1-D LQ Riccati

Validation probes use this, and the ansatz normalisation is load-bearing. Terminal cost
$u(T,x) = \tfrac12 (x - x_c)^2$, pure LQ ($H = \tfrac12|\nabla u|^2$, $\sigma = 0$, no coupling, no
potential, no source). Substituting $u(t,x) = P(t)(x-x_c)^2$ — **not** $\tfrac12 P(t)(x-x_c)^2$,
which gives a different Riccati equation — into the canonical equation:

$$
P'(t) = 2P(t)^2, \qquad P(T) = 0.5,
\qquad\text{so}\qquad P(t) = \frac{1}{4 - 2t} \ \text{ at } T = 1 .
$$

| quantity | value |
|---|---|
| $P(T)$ | 0.5 |
| $P(0)$ | 0.25 |
| $u(0)/u(T)$ at an off-centre probe | **0.5** |

**Probe off-centre.** At $x = x_c$ the factor $(x-x_c)^2$ vanishes and $u \equiv 0$ for all $t$, so a
centred probe cannot distinguish a correct solver from a broken one.

### Backward-sweep direction — and its hypothesis

**With $V = f = S = 0$ and $\sigma = 0$**, the residual gives $\partial_t u = H_{\text{control}} \ge 0$,
so $u_n \le u_{n+1}$, strictly where $|\nabla u| > 0$: the cost-to-go shrinks backward in time as
control acts. That is the worked example above.

**Any of those hypotheses failing reverses it.** A positive $V$, $f$ or $S$ adds to $\partial_t u$
with the opposite sign and $u$ can increase backward in time; so can diffusion against convex
terminal data. The sign of $g$ is irrelevant — it does not enter $\partial_t u$ at all. Do not use
this direction as an acceptance criterion outside the stated hypotheses.

### The density the HJB step reads

**The HJB step at $t_n$ reads the density at the same time level, $m^n$**: #1423's ruling, stated here
on 2026-10-08.
Reading $m^{n+1}$ instead is an $O(\Delta t)$ change in the coupling alone, so two schemes solving the
same problem must read the same level. HJB-FDM (1-D, through `base_hjb`, and n-D) reads `M_density[n]`.
*Not yet met (#2429, #2557): HJB-SL reads $m^{n+1}$ (`m_idx = min(n + 1, …)` in
`hjb_semi_lagrangian.py`), and no single owner yet gives "the density for HJB step $n$".*

---

## 3. Hamiltonian and Lagrangian

### Duality

$$
H(t, x, p, m) \;=\; \sup_{\alpha}\,\bigl\{\, -\,p \cdot \alpha \;-\; L(t, x, \alpha, m) \,\bigr\}
$$

with the **minus** sign on $p \cdot \alpha$. This is the form under which

$$
\alpha^{*}(t, x, p, m) \;=\; -\,D_p H(t, x, p, m)
$$

holds for **every** $L$, and it is the identity the feedback law is computed from throughout the
library. `legendre_transform` on a Hamiltonian or a Lagrangian means this formula.
`LagrangianBase.conjugate_argmax` returns $\partial H/\partial p$ — the **negated** maximiser — and
`optimal_control` returns the maximiser $\alpha^*$ itself.

Minimisation alone does not fix this form. $H = H_{\text{control}} - V - f$ takes minimisation, the sup above and
cost-signed $V$ and $f$ (§ 2) together. The same minimisation written as $\inf_\alpha\{p\cdot\alpha + L\}$ gives
$V + f - H_{\text{control}}$, and reading $V$ and $f$ as rewards, as mechanics reads $L = T - V$, gives
$H_{\text{control}} + V + f$.

The plain convex conjugate $L^{*}(p) = \sup_\alpha\{p\cdot\alpha - L(\alpha)\}$ is a different
function: $H(p) = L^{*}(-p)$. The two agree exactly when $L$ is even in $\alpha$ — every quadratic,
$\ell^1$ and symmetric bounded-control cost here — and disagree as soon as $L$ carries an odd term
such as a drift penalty $b\cdot\alpha$.

### The separable case

For $L(t,x,\alpha,m) = \ell(\alpha) + V(t,x) + f(m)$ with $\ell$ convex,

$$
H(t,x,p,m) \;=\; \ell^{*}(-p) \;-\; V(t,x) \;-\; f(m),
$$

and for $\ell(\alpha) = \tfrac{\lambda}{2}|\alpha|^{2}$ this is $H_{\text{control}} =
|p|^{2}/(2\lambda)$ with $\alpha^{*} = -p/\lambda$.

### One drift, one owner

The Fokker–Planck drift has one owner, the Hamiltonian's control cost, and no solver keeps its own
copy. Solvers that advect with $-c\,\nabla u$ read $c$ from `fp_drift_coefficient(problem)`; FVM, FEM,
meshless-Galerkin and, for a separable $H$, the particle solver take $\alpha^*$ from `H.optimal_control`
on their own basis.

`fp_drift_coefficient(problem)` resolves in this order:

1. A `SeparableHamiltonian` that serves an optimal control of its own → **raises**, naming what solves
   it (#2900). It serves its own unless the method it serves is `SeparableHamiltonian.optimal_control`
   bound to it and, for a `QuadraticControlCost`, the cost serves that class's own
   (`drift_is_the_control_costs`).
2. A quadratic `SeparableHamiltonian` → $1/\lambda$, taken from its control cost. **This is the path a
   well-formed problem takes.**
3. A `SeparableHamiltonian` that is *not* quadratic → **raises**. The scalar form $-c\,\nabla u$ does
   not represent that optimal control.
4. Any Hamiltonian that is not a `SeparableHamiltonian`, or none at all → the problem's
   `coupling_coefficient`, a legacy scalar.
5. Neither of the above → **raises**.

`coupling_coefficient` is **not** an alternative spelling of the control cost and must never be set
to stand in for one.

### What the Fokker–Planck solver receives

Two channels, and each FP solver declares which one it reads (`_drift_convention`):

- **`potential_field` carries the value function $u$.** On the default path — a smooth separable
  $H$ — the coupling layer passes `potential_field=U` to every FP solver that has that parameter, and
  the solver forms the drift itself. FP-FDM and the semi-Lagrangian solver form $-c\,\nabla u$, the
  control cost's drift, and say so by `_forms_its_drift_from_u`. FVM, the particle solver and the
  weak-form family read `optimal_control`. *Not yet met (#2429): `MultiPopulationIterator` passes the
  value function to network populations as `drift_field=`, the deprecated alias.*
- **`drift_field` carries a velocity $\alpha^*$.** The coupling layer computes it from
  `optimal_control` for a non-smooth or non-separable $H$, and passes it only to a solver whose
  convention is `VELOCITY` (FDM, FVM; FP-GFDM was the third, withdrawn in #2583).
- **A separable $H$ that serves an optimal control of its own** (#2900; `drift_is_the_control_costs`
  is false) is not driven by $-c\,\nabla u$. A solver that reads `optimal_control` gets it as $u$, as
  any separable $H$. A solver that forms $-c\,\nabla u$ gets it as a velocity only where FP-FDM's
  velocity route has been measured equal to its value-function route: one population on a 1-D grid
  with `NO_FLUX` walls, handed a density trajectory, with an advection scheme that reads a velocity.
  Anywhere else, and on the semi-Lagrangian solver, the coupling layer refuses it, naming only what
  the #2900 tests run there. FP-FDM's velocity route off that set is #2904.

A solver whose convention is `VALUE_FUNCTION` — the particle solver, the semi-Lagrangian solver,
the network solver and the weak-form family — cannot represent the drift of a non-smooth or
non-separable $H$, and the coupling layer refuses to route $u$ to one rather than let it advect with
$-c\,\nabla u$. `FPGFDMSolver`, withdrawn in #2583, had no `potential_field`, and the coupling layer refused
to hand it $u$ as a velocity; that refusal still guards any solver that takes only `drift_field`. *Not yet met (#2429): called directly, the semi-Lagrangian pair and the particle solver
accept `potential_field=U` with a non-separable $H$ and advect with $-c\,\nabla u$, $c$ being the legacy
`coupling_coefficient`.*

`drift_field=` is a deprecated alias for `potential_field=` on the semi-Lagrangian pair, the network
solver and the weak-form family. On the particle solver it is a live alias for the same slot, and also
takes a callable drift, or a precomputed velocity with `drift_is_precomputed=True`.

The two halves of the system agree because both read one Hamiltonian object, not because of where
the drift is computed. The weak-form family differentiates $u$ on its own FEM or MLS basis; do not
replace that path with a coupling-layer velocity computed on another basis, which would break the
paired HJB–FP operator relation.

---

## 4. Derivatives

Derivatives of order $p$ in dimension $d$ are **tensors of shape** `(d,) * p`, carried by
`DerivativeTensors`:

| order | name | shape | access | meaning |
|---|---|---|---|---|
| 0 | value | scalar | `derivs.value` | $u$ |
| 1 | gradient | `(d,)` | `derivs.grad[i]` | $\partial u / \partial x_i$ |
| 2 | Hessian | `(d, d)` | `derivs.hess[i, j]` | $\partial^2 u / \partial x_i \partial x_j$ |
| $p$ | $p$-th order | `(d,) * p` | `derivs[p][i_1, ..., i_p]` | $\partial^p u / \partial x_{i_1} \cdots \partial x_{i_p}$ |

In one dimension the shapes are `(1,)` and `(1, 1)`, not scalars. Derived quantities are properties
of the same object — `derivs.laplacian` is $\operatorname{tr}(\nabla^2 u)$, `derivs.grad_norm_squared`
is $|\nabla u|^2$. `DerivativeTensors.from_arrays`, `from_gradient` and `zeros` construct one;
`from_multi_index_dict` and `to_multi_index_dict` convert to and from the legacy format below.

**What a Hamiltonian receives is the momentum $p$, not a `DerivativeTensors`.**
`HamiltonianBase.__call__(t, x, p, m)` never sees one. `DerivativeTensors` is one of two argument
types of the per-point adapter `problem.H(x_idx, m_at_x, derivs=...)`; the other is the legacy
multi-index dict `{(1, 0): u_x, ...}`, which is still live. The adapter reduces either to $p$.

Who builds one:

- **HJB-WENO** builds one per point per Hamiltonian evaluation on its default path.
- **HJB-GFDM** builds them only with `monotonicity_scheme="joint_socp"`; its
  default batch path passes gradient arrays of shape `(N, d)`.
- **HJB-FDM** never builds one. It evaluates $H$ in one batch on gradient arrays: in every dimension on
  the NumPy backend, and in $d \ge 2$ on any backend. Its 1-D residual and Jacobian call `problem.H()`
  per point, with the legacy dict `{(0,): u, (1,): p}`, only on another backend or with
  `analytic_jacobian=False` (#1884). A Hamiltonian evaluated in batch must give, at every node, what it
  gives at that node alone; `0.5*p[0]**2` does not, since in batch `p[0]` is node 0's momentum. A problem
  checks this when it takes the Hamiltonian (§ *Callable signatures*) and refuses such a Hamiltonian. Write it row-wise,
  as in `0.5*np.sum(p**2, axis=-1)`.
- **HJB semi-Lagrangian** passes arrays and does not use them. No FP solver uses them.

---

## 5. Volatility and diffusion

### Two quantities, never conflated

From the Itô dynamics $\mathrm{d}X_t = \alpha_t\,\mathrm{d}t + \Sigma\,\mathrm{d}W_t$:

- $\Sigma$ is the **volatility** — the SDE's noise matrix, and the primitive the user supplies.
- $A = \tfrac12 \Sigma\Sigma^{\mathsf{T}}$ is the **diffusion** — the PDE's second-order coefficient,
  appearing as $\operatorname{tr}(A\,D^2u)$.

In one dimension this is $D = \sigma^2/2$. That is not a separate rule; it is the scalar case.

The Fokker-Planck equation of those dynamics, the adjoint of the HJB term $\operatorname{tr}(A\,D^2u)$, is in Itô
form: $\partial_t m + \nabla\cdot(\alpha m) = \partial_i \partial_j (A_{ij}\, m)$, with the total flux
$J_i = \alpha_i m - \partial_j(A_{ij}\, m)$ (user ruling 2026-10-30, #2841). Since
$\partial_i \partial_j (A_{ij}\, m) = \nabla\cdot(A \nabla m) + \nabla\cdot(m\, \nabla\cdot A)$, with
$(\nabla\cdot A)_i = \partial_j A_{ij}$, it is $\nabla\cdot(A \nabla m)$, and $J = \alpha m - A \nabla m$, on a
region where every row of $A$ is divergence-free, as for a constant $A$. For a field, $A = D(x)\, I$, it is
$\Delta(D m)$: neither $D\,\Delta m$ nor $\nabla\cdot(D \nabla m)$, which differ from it wherever $D$ varies, and
under NO_FLUX with no drift its stationary density is $m \propto 1/D$. FP-FDM reads every varying $A$ through one
owner, `ito_diffusion`, and the particle FP's Euler-Maruyama step solves the same equation.

$\Sigma\Sigma^{\mathsf{T}}$, **not** $\Sigma^{\mathsf{T}}\Sigma$. For a square $\Sigma$ the two
coincide exactly when $\Sigma$ is normal ($\Sigma\Sigma^{\mathsf{T}} = \Sigma^{\mathsf{T}}\Sigma$) —
every symmetric $\Sigma$ is — so a non-normal $\Sigma$ gives two different matrices.

`problem.volatility` holds $\Sigma$ **as supplied** — a scalar, an array with its kind, or a
callable. `problem.diffusion` is $A$, derived on access. There is no scalar view of a non-scalar
volatility: a consumer that needs one calls `scalar_volatility(volatility, consumer=...)`, which
refuses a field, a tensor or a callable by name rather than averaging it.

### Where the volatility is supplied

In the model/domain/conditions form the volatility belongs to the model: `Model(volatility=..., volatility_kind=...)`.
The `Model` has no `diffusion` field.

The legacy constructor (§ *Component ownership*) takes `volatility=` or `diffusion=`, which are
mutually exclusive: supplying both raises, and the library neither guesses which was meant nor checks
them against each other.

A per-solve override is `volatility=` with the same `volatility_kind=` rule, on `solve_hjb_system`,
`solve_fp_system` and the coupling loops. One function, `resolve_volatility_override`, reads every
override.

### An array volatility declares its kind

A **scalar** volatility is $\sigma$ everywhere, isotropic: $A = \tfrac{\sigma^2}{2} I$. It takes no
kind, and declaring one raises. An **array** does not carry its own meaning, so it declares
`volatility_kind=`:

| `volatility_kind=` | what the array is | shape | diffusion |
|---|---|---|---|
| `"field"` | one $\sigma$ per grid point: isotropic noise whose strength varies in space, $\Sigma(x) = \sigma(x)\,I$ | the grid's shape, or with a leading time axis | $A(x) = \tfrac12\sigma(x)^2 I$ |
| `"tensor"` | the noise matrix $\Sigma$: anisotropic noise, with off-diagonal entries coupling the axes | `(d, d)` constant, or `(*grid_shape, d, d)` per point | $A = \tfrac12\Sigma\Sigma^{\mathsf{T}}$ |

The same numbers are two different problems. On a $2 \times 2$ grid in two dimensions,
`[[0.3, 0.1], [0.1, 0.2]]` read as a tensor is $A = \begin{pmatrix} 0.05 & 0.025 \\ 0.025 & 0.025 \end{pmatrix}$
at every point; read as a field it is four points with $\sigma = 0.3, 0.1, 0.1, 0.2$, each isotropic,
$A = 0.045\,I,\ 0.005\,I,\ 0.005\,I,\ 0.02\,I$.

**`volatility_kind` is required for any array and there is no default.** The library does not read
the kind from the shape: a rule that works on a $50 \times 50$ grid and guesses on a $2 \times 2$ one
produces a silently wrong diffusion exactly where fixtures and quick tests live.

- A **field** may vary in time, with the time axis leading. A meshfree solver's per-solve override is
  indexed like its density, by collocation point. *(The leading time-axis length is not validated:
  #2429.)*
- A **tensor** is square. A non-square $(d, k)$ noise matrix is refused: no solver reads one, and its
  symmetric $(d, d)$ square root, `volatility_from_diffusion(A, kind="tensor")`, has the same $A$. Grid
  axes lead and matrix axes trail. A tensor on a 1-D problem is refused; there it is the scalar
  $\sqrt{\sum_k \Sigma_{1k}^2}$.
- A **callable** $\Sigma(t, x, m)$ is read per point, as returning $\sigma$; it declares
  `volatility_kind="tensor"` when it returns the matrix.
- A **per-axis vector** of shape `(d,)` is refused: the volatility it describes is the diagonal
  tensor, `np.diag(v)` with `volatility_kind="tensor"`.

The kinds are one rule wherever a volatility is accepted. A problem's volatility is shape-checked at
construction; an override's shape is checked by the solver that reads it.

**A solver that cannot discretise a kind refuses it by name rather than approximating it:**

| solver | accepts |
|---|---|
| FP-FDM | every kind, except a tensor that is not symmetric |
| FP-particle | a scalar on the grid-drift path; on the callable-drift path also a field, a constant or per-point tensor and a per-point callable, but not a callable tensor |
| HJB-FDM | a scalar, a field, a scalar-valued callable, or a tensor only as a constant diagonal $\Sigma$ (none in 1-D) |
| HJB semi-Lagrangian | a scalar, or a constant $(d, d)$ tensor; a per-solve override only if it equals the problem's |
| FVM | a scalar, or a constant field read as its scalar |
| HJB-GFDM | a scalar, a field, or a space-only callable $\sigma(x)$ — no tensor |
| Howard (`HJBHowardSolver`) | a scalar or a field, at construction only — a per-solve override is refused |
| FP-GFDM, the FP semi-Lagrangian pair, the weak-form family, `FPNetworkSolver` | a scalar only |
| WENO | a scalar only; a per-solve override only if it equals the problem's |
| the network HJB solvers | no non-zero volatility — they have no diffusion term *(a callable raises a bare `TypeError` rather than a refusal by name: #2429)* |

**Common noise is not a column of $\Sigma$.** A shared noise source does not average out in the
mean-field limit and makes $m$ a random measure flow; it enters through `StochasticMFGProblem`'s
`noise_process`, which samples realisations and solves a conditional MFG for each. Every column of
$\Sigma$ is idiosyncratic.

### The reverse direction is the symmetric square root

`volatility_from_diffusion` converts $A$ back to $\Sigma$ with the **symmetric matrix square root**
(via `eigh`), not a Cholesky factor. The grid consumers that assemble a cross-derivative — FP-FDM and
HJB semi-Lagrangian's ADI step — validate a tensor as symmetric positive semi-definite and reject a
lower-triangular one. The particle solver accepts any square root, and the problem does not check. A
user-supplied $(d, d)$ volatility should therefore be the symmetric standard-deviation matrix.

### One converter

Every solver derives $A$ through `diffusion_from_volatility`. The inline forms that remain are the
CFL diagnostics, the code generator's emitted flux, the GBM noise process's own Itô correction (that
process's $\sigma$, not the agents'), and `diffusion_from_volatility_torch`, the second owner for the
torch substrate. No assembled operator computes its own $\tfrac12\Sigma\Sigma^{\mathsf{T}}$,
$\operatorname{diag}(\Sigma)^2/2$ or $\sigma^2/2$: an HJB solver dropping a cross term that its FP
partner keeps is one system solving two different problems.

The converter's scalar and field paths can differ in the last bit for the same $\sigma$ (#2428); do
not rely on a scalar and a constant field being bit-identical.

---

## 6. Callable signatures

Callables order their arguments

$$
(\,t,\; x,\; u,\; Du,\; D^{2}u,\; m,\; \ldots\,)
$$

— time, then space, then the $u$-derivative family in ascending order, then the measure, then any
further parameters. This holds for every callable the library accepts from a user, and for every
method of the operator family (`HamiltonianBase`, `LagrangianBase`, `ControlCostBase`).

**Slots that do not apply are omitted; the surviving ones never change order.** A Lagrangian's
control $\alpha$ occupies the $Du$ slot, being its conjugate variable.

| callable | signature |
|---|---|
| `HamiltonianBase.__call__`, `.dp`, `.dm`, `.dx`, `.optimal_control` | `(t, x, p, m)` |
| `LagrangianBase.__call__` (and `.dm` where a subclass defines it) | `(t, x, alpha, m)` |
| `u_terminal`, `m_initial` | `(t, x)` |
| `source_term_hjb`, `source_term_fp` | `(t, x, v, m)` |
| a solver's `source_term` | `(t, x)` |
| `potential` | `(t, x)` |
| `PotentialCost`'s and `PotentialEnergy`'s `V` | `(x)` |
| `coupling`, `coupling_dm`, `congestion_factor`, `congestion_factor_dm` | `(m)` |
| a graph coupling's `coupling_hjb`, `coupling_fp` | `(t, i, j, v_j, m_j)` |
| `ControlCostBase.optimal_control` / `.lagrangian` | `(p)` / `(alpha)` |
| `HJBHowardSolver`'s `control_lagrangian` | `(alpha)` |
| a conditional Hamiltonian / a conditional terminal cost | `(t, x, p, m, theta)` / `(x, theta)` |
| a level set's `velocity`, a sampler's `navigable`, Nitsche's `f` / `u_exact` | `(x)` |
| `ImplicitHeatSolver`'s `forcing` | `(t, x)` |
| a variational `lagrangian_func` / `terminal_cost_func` | `(t, x, v, m)` / `(x)` |
| volatility, diffusion, drift and PDE coefficient callables | `(t, x, m)` |
| a measure field's `field_fn` / `gradient_fn` | `(t, x, mu)` |
| a boundary value: `BCSegment.value`, `BoundaryConditions.default_value`, a meshfree `bc_value` | `(t, x)` |
| a region: `sdf_region`, `domain_sdf`, `obstacle_sdf`, a `predicate`, meshless Galerkin's `domain`; `state_penalty`; `Mesh1D`'s `refinement_function` | `(x)` |
| a network condition: `initial_node_density_func`, `terminal_node_value_func` | `(t, node)` |
| a graph condition's `NodeBC.nodes` | `(num_nodes)` |
| a variational `state_constraints` / `velocity_constraints` entry | `(t, x)` / `(t, x, v)` |
| Howard's `alpha_star` / `running_cost` | `(t_idx, x, p, m)` / `(t_idx)` |

The order is chosen so the derivative family can grow: $D^2u$ is already a first-class object one
layer down (§ *Derivatives*), and a signature with $m$ between $x$ and $p$ leaves nowhere to put it.
Time-first also agrees with § *Arrays, grids and time* and with `scipy.integrate.solve_ivp`'s
`(t, y)`.

### A callable is read by the names it declares

**On the operator family, `t` has no default** and is always passed, and methods keep the order above. A callable you
hand the library is read by name instead: each parameter named for a slot is given that slot, by keyword, and a slot it
does not name is not passed. So order does not decide what a parameter reads, and a time-independent callable omits
`t`.

- **A parameter with a default that names no slot keeps its default.** One without a default is refused.
- **It is a function you write**, a `def` or a `lambda`, or a `functools.partial` of one. Any other callable is refused,
  with the advice to wrap it, such as `lambda t: s(t)` for an interpolant `s`: another callable's parameter names were
  not written for the slots, and an interpolant over time data names its argument `x`.
- **`*args`, `**kwargs` and positional-only parameters are refused.** Nothing reaches a starred parameter, since it is
  passed only the slots it names.
- **A partial that freezes a slot by keyword is refused**, since the slot passed by name would override it. A partial
  that freezes a named parameter by position keeps it. One refused for its other parameters is told to write a function
  passing the frozen value by keyword, such as `lambda x: g(x=x, t=0.25)`, because renaming the parameters of
  `partial(g, 0.25)` moves the 0.25 to another parameter.
- **Currying the measure needs a keyword**: `partial(H, m=mu)`, for your own calls.

**A reader with less to pass refuses more.** A boundary value is read where it is, so a reader with no point, such as
the FDM ghost layer, refuses one that declares `x`, even with a default, and one with no time refuses one that declares
`t` (#2512, row B2). `BCSegment.value` is bound, and refused, when the segment is built; `BoundaryConditions.default_value`,
a meshfree `bc_value`, and a `value` reassigned later are bound when first read.

**`u_terminal` and `m_initial` are functions of space.** `lambda x: ...` is read on the grid; `lambda t, x: ...` is read
at $t = T$ for `u_terminal` and $t = 0$ for `m_initial`.

### `x` is a coordinate tuple

`x` holds one entry per coordinate, and each entry is a scalar or an array, all of one shape: `x[0]` is the first
coordinate, of one point or of every point. The library may hand either, so a body reads both the same way:

| reads a batch and a point the same | what it does on a tuple |
|---|---|
| `x[0]`, `a, b = x`, `np.hypot(*x)`, `sum(xi**2 for xi in x)`, `np.linalg.norm(x, axis=0)`, a constant | |
| | `x**2`, `x - 0.5`, `x[:, 0]`, `x[..., 0]`: raises |
| | `np.asarray(x)[:, 0]`, `float(x[0])`, `np.linalg.norm(x)`, `np.sum(x)`, `x[0][0]`, `np.max(x[0])`: refused by the check below |

A door that holds the points states their dimension. It hands them as it holds them: a problem's conditions and
potential in one row, `(N,)`, and a solver's drift, volatility and sources at the grid's shape, `(Nx, Ny)`. A door that
reads one point hands a tuple of scalars, at $d = 1$ too.

A region is read at one point, where the library reads one, or at points of shape `(N, d)`, one row per point. So a
predicate you call yourself, such as `sdf_region(sdf)(points)`, takes `(N, d)`, and refuses a 1-D array by name, since
it does not say whether it is $N$ points on a line or one point in $N$ dimensions: pass `points.reshape(-1, 1)` for
points on a line. Before #2875 such a predicate handed `(N,)` to a body written to it, which at $d = 1$ read it as $N$
points on a line; that read is gone. `sdf_gradient` and an implicit domain's `get_region_mask` read `(N,)` as one point
in $N$ dimensions, as they did before.

**One body you write is not handed a tuple.** A geometry's `signed_distance` is a method of the geometry and not a callable
it was handed, and it takes its points as an array of shape `(d,)` or `(N, d)`. So a subclass you write of `ImplicitGeometry`
reads an array there, as the library's own regions do.

### What the slots beside `x` hold

Each door states it, in `callable_protocols.py`'s `Slots`, and the check below reads the statement.

| slot | at | holds | a point read gets |
|---|---|---|---|
| `m`, `v` | `source_term_hjb`, `source_term_fp`; `m` at a volatility, diffusion, drift or coefficient | a field over the points `x` holds | the field whole, so `np.max(m)` may read it |
| `p`, `m` | a raw Hamiltonian, a conditional Hamiltonian, `alpha_star` | one value per point, in step with `x` | that point's |
| `v`, `m` | a variational Lagrangian | one value per point, in step with `x` | that point's |
| `t`, `t_idx`, `mu`, `theta` | wherever they appear | one value for the call | that value |

### A body must read a batch and a point the same

The library chooses when to loop and when to vectorize, so a body that reads differently comes back as a result and not
as an error. So a callable's batch read is compared with its read of a few of those points, and a difference, or a raise
of `TypeError`, `ValueError`, `IndexError`, `ArithmeticError`, `AttributeError` or `KeyError`, is refused: a difference
names both reads, and a raise names the read that raised. Any other raise reaches you as it is. Two functions make the comparison: `_compare_reads` for a callable at its
door, and `refuse_batch_disagreement` for a Hamiltonian's `evaluate_H` and `evaluate_dp`.

- **Where.** A door with samples when it accepts the callable compares there: `MFGProblem` for its conditions, its
  potential, its `state_penalty` and its Hamiltonian, on its grid. Any other door compares at its first batch, before it
  returns the result: a volatility or a source at a solver's first step, a solver's `source_term`, a drift at a
  particle's first step, a measure field, a manufactured pair. A door that hands one point, or an empty batch, has
  nothing to compare, and the next batch is compared.
- **Once it agrees, and where the mark is kept.** A comparison that agrees, with two or more distinct points among the
  rows it compares, marks the body compared, and it is read with no comparison after that. Where the mark is kept
  depends on the door:
  - **With the body, while it lives, for every door that binds it to the same `Slots`:** a problem's `u_terminal` and
    `m_initial`; a solver's `source_term`; the volatility a solver reads through a
    `CoefficientField` or FP-FDM's tensor path, the particle solver's drift and volatility, `resolve_diffusion_source`
    and `composite_drift`; a region or signed distance, a boundary value, a graph's node value, rule and selector, and a
    network's initial density and terminal value; `PotentialEnergy`'s `V`; Nitsche's `f` and `u_exact`, and a 1-D
    mesh's refinement function; the optimal-transport initial density and running cost, and the variational initial
    guess and constraints; a raw Hamiltonian at the validator. A body accepted at one of these is read with no
    comparison at the next, a new problem's included.
  - **On the object that holds the binding:** a Hamiltonian's potential and its density terms; a measure field; a
    problem's `source_term_hjb` and `source_term_fp`, its `diffusion`, and its `state_penalty`, compared inside the
    potential it is composed into; a stochastic problem's conditional
    Hamiltonian and terminal cost; a variational problem's Lagrangian and terminal cost; a network's node terms; a
    manufactured pair; `ImplicitHeatSolver`'s forcing; Howard's `alpha_star`, running cost and control Lagrangian; a
    graph coupling; FP-FDM's callable drift, whose holder lasts one solve. A body handed to a new holder of these is
    compared once more there.

  A refusal raises and is not kept. One point, a batch whose compared rows are one point, an empty batch, or a read with
  no `x` leaves the body as it was.
- **The samples.** They are spread across the batch at the golden ratio's multiples and are not its ends and middle, where
  a term such as $\sin 2\pi x$ is the same on a uniform grid. A batch of fewer than two points, or of exactly $d$, is
  cycled up to a count that is neither, since a coordinate axis passes for a batch of $d$.
- **The forms a return may take.** At every door a scalar holds at every point, and an array holds one entry per point:
  the batch's shape leading, each entry of one point's own shape. The comparison also reads a per-coordinate result with
  its axis first or last, but each door reads one: FP-FDM's drift the axis first, the particle solver's the axis last
  (#2875's stage 3 settles one). Two more are stated door by door in `Slots`, and are not inferred from the return:
  - At a door with a field, a body may read the field whole, so a point read that returns an array of the batch's shape
    holds its entry there. So `lambda t, x: S` with `S` of the grid's shape reads at `source_term_hjb`.
  - A volatility or a diffusion may return one fixed array, if every point's read returns that array too, and it is read
    under `volatility_kind`.

  Anything else has no entry per point and is refused by name, before it reaches a read by size that would take a
  transposed array in C order. So a body holding an array is refused at a condition, a potential, a solver's `source_term`,
  a region, a measure field, `PotentialCost` and an exact solution's field, and a constant vector is refused at a drift.
- **A Hamiltonian is compared on wide synthetic $p$ and $m$**, $p$ from $-1.3$ to $2.1$, never on a solve's first values,
  where a problem takes it, on `evaluate_H` and `evaluate_dp`. `DualHamiltonian` and `NetworkHamiltonian` read one point
  at a time on every path, and are not compared.
- **A closure the library wrote is not compared.** The user's callable it wraps has its own door.

**What this does not catch.**

- A refusal is not remembered. A later read, by a new holder or the same one, compares again on its own samples, so a
  sweep that catches a refusal and rebuilds its problem can be accepted where the new samples agree. A potential that
  reads a batch differently from a point left of 1.5 is refused on 11 nodes of `[0, 1]`, and accepted on 11 nodes of
  `[1.4, 3.0]`, where the samples sit right of 1.5 and the first node left of it, and that solve reads it on the batch.
- A body accepted at a door that keeps the mark with the body is not compared again, at a new problem either. A
  `u_terminal` that reads a batch 1 higher than a point wherever `x[0] < 0.15` is refused by a problem on 11 nodes of
  `[0, 0.5]`, whose samples include 0.1. Accepted first by a problem on 11 nodes of `[0, 1]`, whose samples are 0.2,
  0.6 and 0.9, it is then read by the `[0, 0.5]` problem with no comparison, 1 higher at its nodes below 0.15.

- A first batch can be narrow. A particle drift's first batch is the initial cloud, so a body that reads its largest
  coordinate past a dead zone reads the same both ways while the cloud is inside it, is accepted, and is not compared
  again, in that solve or a later one.
- At a field door `m[0]` reads node 0 at every point, and the comparison cannot tell it from `np.max(m)`, since a point
  read gets the field whole. A body written to `x` as a field, such as `np.gradient(m, x[0])`, is refused, since `x` is a
  point on a point read.
- The comparison reads a few samples, and it says the reads agreed there.
- A body handed no `x` is not compared: a coupling, a congestion factor and their derivatives, Howard's running cost and
  control Lagrangian, and a network's node terms.
- A closure the library wrote is not compared, and the test is the module that defined it, or a mark the library sets
  on a closure it calls itself, so a body that sets its own `__module__` to the library's, wraps a library function with
  `functools.wraps`, or carries the mark, passes as one.

### The library refuses the old order on a class

A class is not read by name when it is called by its own methods, so a Hamiltonian or Lagrangian subclass, or an
`MFGProblem` subclass's `hamiltonian` / `running_cost`, is refused when the class is created if its methods are out of
order.

*Not yet met (#2429): the per-point adapter `problem.H` / `problem.dH_dm` is not time-first and
defaults its time to 0, and some per-point solver paths call it without one.
`LagrangianBase.proximal(tau, z, *, t=0.0, x=None, m=None)` puts its step arguments first and defaults
its time to 0, and several private helpers put `t` last: the finite-difference ones and
`DualHamiltonian._find_optimal_alpha` take `(x, m, p, t)`, `DualLagrangian._find_optimal_p` takes
`(x, alpha, m, t)`, and `NetworkHamiltonian._default_hamiltonian` takes `(node, m, p, t)`. `p` and a vector return are
next, in #2875's stage 3: `p` is `(N, d)` or `(d,)`, with the coordinate axis last.*

---

## 7. Solver method surface

The two solver entry points, as the base classes declare them:

```text
solve_hjb_system(M_density, U_terminal, U_coupling_prev,
                 volatility=None, source_term=None, volatility_kind=None)

solve_fp_system(M_initial, drift_field=None, volatility=None, show_progress=None,
                progress_callback=None, source_term=None, volatility_kind=None)
```

**Only the leading arrays are the positional contract; pass everything else by keyword.** The
implementations add parameters in different places — a `potential_field` for the value function
(§ *Hamiltonian and Lagrangian*), progress controls, `cross_density` — and on the semi-Lagrangian
pair, the network solver and the weak-form family `potential_field` is the second positional
parameter. `HJBHowardSolver` takes no `U_coupling_prev`; the network HJB solvers take no
`source_term`.

The FP solver's first argument is `M_initial` on every solver. `m_initial_condition=` (GFDM,
network) and `m_initial=` (the weak-form family) survive as deprecated aliases.

*Not yet met (#2429): `WeakFormHJBSolver.solve_hjb_system` still accepts the pre-v0.12 names
`M_density_evolution_from_FP`, `U_final_condition_at_T` and `U_from_prev_picard` for `M_density`,
`U_terminal` and `U_coupling_prev`, with no warning; `base_hjb.solve_hjb_system_backward`
(`M_density_from_prev_picard`, `U_final_condition_at_T`, `U_from_prev_picard`) and HJB-WENO's private
per-dimension solvers still use these parameter names.*

**Solver-method array parameters and results are uppercase** (`U`, `M`, `U_terminal`, `M_density`,
`M_initial`, `SolverResult.U`, `SolverResult.M`); scalars and configuration are lowercase. The channel
names `potential_field` and `drift_field` are lowercase although they carry arrays. A name suffixed
`_coupling_prev` is outer-iteration state.

**Three nested loops.** The **outer** loop is the coupling (Picard / fixed-point) iteration; the
**middle** one is the time step; the **inner** one is Newton — or Howard policy iteration — within a
time step. Howard replaces the per-time-step Newton only, never the coupling loop.

On the coupling-level API — `MFGProblem.solve`, the coupling iterators, `PicardConfig` —
`max_iterations` and `tolerance` are the outer loop's. HJB-FDM and HJB-GFDM qualify theirs:
`max_newton_iterations`, `newton_tolerance`. Elsewhere the bare name does not mean the outer loop:
`NewtonConfig.max_iterations` / `.tolerance`, `HJBSemiLagrangianSolver(tolerance=)`,
`NetworkHJBSolver(tolerance=)` and `HJBHowardSolver(max_iter=, tol=)` are inner-loop settings. What the
outer and the Newton tolerance bound is in § *The measure*, *What a tolerance bounds*.

The fixed-point relaxation parameter is `relaxation` (with `relaxation_M`, `adaptive_relaxation`,
`relaxation_schedule`).

---

## 8. Arrays, grids and time

- **Time is the leading axis.** A solution field is `(Nt + 1, Nx_points)` in one dimension and
  `(Nt + 1, *grid_shape)` in general. `mfgarchon/types/arrays.py` records the layout; its aliases are
  bare `NDArray` and enforce nothing.
- **Spatial axes follow `indexing="ij"`**, so axis 1 is $x$ and axis 2 is $y$. Nothing checks the
  pairing of a tensor row or a grid spacing with its axis, and a mispairing is silent (#1911).
- $u$ is solved **backward** from $u(T,\cdot) = g$ and $m$ **forward** from $m(0,\cdot) = m_0$. Both
  are stored in the same forward-time layout, index 0 being $t = 0$.
- A spatial field carries no time axis. A per-point volatility tensor is `(*grid_shape, d, d)`, grid
  axes leading, matrix axes trailing.

*Not yet met (#2429): vector fields have two layouts — FP-FDM's `drift_field` is
`(Nt+1, d, *grid_shape)` and FP-FVM's is `(Nt+1, *grid_shape, d)`.*

### Counts: intervals versus points

**`Nx` counts intervals; `Nx_points` counts points**, and `Nx_points = Nx + 1` per axis. The spacing
is $L/N_x$. The same split holds in time: `Nt` counts steps, `Nt_points = Nt + 1`, `dt = T / Nt`, and
the time grid is `tSpace`, of length `Nt_points`.

**The two halves live on different objects.** Time is the problem's: `problem.Nt`, `.Nt_points`,
`.dt`, `.tSpace`, and `problem.T`, supplied through `Conditions(T=...)`. Space is the domain's:
`grid.Nx`, `.Nx_points`, `.coordinates` (a tuple of read-only coordinate arrays, one per axis),
`.num_spatial_points` (the product of `Nx_points`) and `.get_spacing(axis)`. There is no
`problem.Nx`, no `problem.xSpace` and no `problem.dx`.

`problem.solve(Nt=...)` solves on a different time grid without changing `problem.Nt`, so a result's
time axis is that solve's `Nt + 1`, not `problem.Nt_points`.

**There is no scalar spatial step.** `get_spacing(axis)` returns a float on a uniform grid; on a
`spacing_type="custom"` grid it returns the array of local spacings,
`np.diff(grid.coordinates[axis])`. There `grid.spacing[axis]` is `None` and `get_grid_spacing()`
raises. A caller may not assume a number.

**A periodic grid is endpoint-inclusive.** With a periodic boundary and `Nx_points=[11]`, `x[0]` and
`x[-1]` are one physical point, and the spacing is $L/N_x$. The convention is carried on the periodic
boundary condition as `PeriodicGridConvention.ENDPOINT_INCLUSIVE`.

A parameter named for one count and holding the other is a defect, not a style choice, and an array
shape is written with the point count. *Not yet met: several internals bind `Nx` to a point
count (#2429), and many docstrings write a shape with `Nx` or `Nx+1` where `Nx_points` is meant (#2236).*

### Spatial parameters are per-axis sequences

A spatial quantity is a sequence with one entry per axis whatever the dimension: `Nx_points=[101]` in
1-D, `Nx_points=[101, 81]` in 2-D, and likewise `bounds`. A scalar `Nx` or `Nx_points` is accepted in
1-D and converted, without a warning; `bounds` has no scalar form, and a flat `(lo, hi)` is read as two
axes and refused. The sequence form is the convention, and library code uses it.

`Nx` and `Nx_points` are mutually exclusive and supplying both raises.

---

## 9. The measure

- **The library does not rescale `m_initial`.** A density whose integral is not 1 is reported and
  used as given; normalising is a modelling decision, not a requirement.
- **The library applies no processing of its own to data** (user ruling, 2026-11-04; `AGENTS.md` § *The library does
  not repair data*). A value is not moved to where the code would like it to be: no clip, floor, shift, regularization or
  renormalization of a density. A callable may do so at its own choice, which changes the equation being solved. The
  census of the sites that do not meet this yet is open under #2485.
  - **FP-FEM and the meshless FP keep a step's negative values.** Nothing is added to the mass. Their size goes in a
    logged warning, one per FP solver in a solve, and a `warnings` filter does not see it, and in
    `result.metadata["fp_negative_density"]`: per time level, `fraction`, the negative part against that level's positive
    part, and `minimum`; then `cumulative`, `largest`, `largest_level`, `lowest`, `lowest_level`, `levels_negative` and
    `levels`. The record is a list on several FP solvers, is in `info` on `NewtonMFGSolver`, and is absent with
    `return_tuple=True`. A solver called alone holds it as `last_negative_density`. On the meshless FP these are MLS
    coefficients, not the density at a point (#2672). A level that is all NaN holds no negative in this record.
  - **A non-finite value function ends a coupled solve at the guard all seven couplers share.** It logs the site, the
    iteration and the minimum of the density that solve was handed, and returns `converged=False`, without raising. `np.log(m)`
    in a `coupling` or a `source_term_hjb` reached it on the six couplers that take FP-FEM, at 81 nodes and sigma 0.3, and on
    `FictitiousPlayIterator` on the meshless FP, at 21 points and sigma 0.1. A direct `H(t, x, p, m)` returns what the body
    returns. A solve can end otherwise:
    `NewtonMFGSolver` past its warm-up raises "a sweep's change must be finite" (#2578), as does a `source_term_fp` that goes
    non-finite on every coupler but `FixedPointIterator`; other HJB solvers on the meshless FP have raised their own errors,
    or stayed finite to `converged=False` with only the warning. A read that stays finite, such as `1/m`, is not caught, and
    only the warning covers it. A density with a negative past the round-off gate has no measure, so
    `ParticleMeasure.from_density`, and `track_measure_field` through it, refuses it.
  - **To clip a step anyway is a change of scheme, and it is the caller's:** a subclass of `FPFEMSolver` whose
    `solve_fp_system` clips what it returns has run, and the coupled solve converged with `np.log(np.maximum(m, 1e-10))`
    on 81 and 289 nodes where the solver alone blows up.
- **The measure has one owner**, `mfgarchon.utils.numerical.quadrature`, exposed on
  `TensorProductGrid` as `integrate`, `measure_weights` and `quadrature_weights`, and correct on
  graded axes. A bare `np.sum(m) * dx` is a *different functional* even on a uniform grid: it gives
  each end node a full cell and over-counts by $\tfrac{dx}{2}(m_0 + m_N)$. No other geometry carries
  an `integrate`; a network is measured on its nodes, and other geometries fall back to a named
  `uniform-cell` or `point-average` measure. `MFGProblem.spatial_measure()` makes that choice, once
  per problem (#2555). It returns the measure as per-node `weights`, and its `integrate` is their
  weighted sum in every branch. A consumer that needs the weights reads them there rather than
  rebuilding them, and a geometry with an `integrate` but no `measure_weights` is refused (#1642).
  *Not yet met (#2429, #2566): live sites outside that fallback still measure with `np.sum(m) * dx`,
  mostly particle/KDE diagnostics and the public convergence helpers #2566 lists.*
- **Mass conservation needs both the boundary condition and a conservative discretisation.** Under
  no-flux and periodic boundaries the divergence-form schemes, FVM and the splatting semi-Lagrangian
  solver conserve mass to rounding; FP-FDM's `gradient_upwind` and `gradient_centered` schemes did not,
  could lose most of it, and were removed in #2007. Under absorbing boundaries the invariant is that the mass lost equals the
  accumulated boundary flux. Conservation and discrete adjointness are separate properties, tested
  separately.
- **Renormalisation is a per-solver property, not a convention.** `FPSLSolver` does not rescale,
  because its splatting conserves; `FPParticleSolver` pins every slice to the caller's mass under `kde_normalization="all"`. Comparing
  $\int m$ across solvers therefore measures the solver as well as the physics.
- **A density converges in a named norm.** At $\sigma = 0$, for a given drift, `FPSLSolver`'s density converges
  in $W_1$, which is what its scheme's theory gives (Carlini–Silva, arXiv:1708.02042, Theorem 4.1: on
  $\mathbb{R}^d$ without a boundary, under its (H)-(Lip) and $\Delta x^2 = o(\Delta t)$, every limit point in
  $C([0,T];\mathcal{P}_1(\mathbb{R}^d))$ solves the equation; at $\sigma = 0$ with a Lipschitz drift that
  solution is unique). It does not converge pointwise where a component of the drift changes sign along its own
  axis, since the deposit's rate departs from the equation's there; in 1-D a compressive stagnation point on a node
  gets twice the equation's rate (#2468, #2645). Where $\sigma = 0$ drives mass into a wall, the exact law has an atom there
  and $L^1$ does not converge either. With $\sigma > 0$, away from walls, the density converges pointwise; where
  mass reaches a no-flux wall, at the default `diffusion_theta`, it need not (#1855). Beyond the semi-discrete
  steady state's wall bias, which is first order in $h$, the split step adds an excess of the same sign of about
  $K(1-\theta)\,|\alpha|\,\Delta t/h$, with $\Delta t$ the step the advection takes and
  $|\alpha|\,\Delta t/h \le 1$. On #1855's 1-D fixture $K$ is between 0.74 and 0.97 at the wall the drift points
  into, and between 0.80 and 1.76 at the other, where it rises with $|\alpha|\,\Delta t/h$. That excess does not
  refine away along $\Delta t \propto h$ at $\theta = 0.5$ and is absent at $\theta = 1$. The library leaves that
  trade-off to the caller; the sub-stepped case and the price are stated at `FPSLSolver`'s `diffusion_theta`. A peak
  value or a plot of an FP-SL density is a pointwise reading.

### What a tolerance bounds

**The outer coupling tolerance bounds a relative change, measured in the discrete $L^2$ norm** whose
measure is the problem's own, the owner above, not a scalar `dx` (ruled 2026-10-08): the change one
sweep makes, the map's output against its input and never the damped step (#1684), relative to the
output's size, for $u$ and for $m$. An iterator over several fields, whether populations, regimes or
graph nodes, takes the max over fields of each field's relative change in its own measure. There is no
absolute criterion unless the caller asks for one with `absolute_tolerance`. A non-finite change is
refused, not judged: the verdict raises, naming the value, and the max over fields raises first, naming
the field (#2578). All six coupling iterators read the owners: `MFGProblem.spatial_measure()` for the
measure, `sweep_change` for the change, and `check_convergence_criteria` for the verdict (#2555); the
three over several fields also read `worst_sweep_change` for the max over fields (#2578).
`tests/unit/test_alg/test_c1_change_has_one_owner_2512.py` holds to the owners everything under
`alg/numerical/coupling/`, and any other module of `alg/` that references `PicardConfig`, its `picard`
attribute, `BaseCouplingIterator`, `check_convergence_criteria`, `sweep_change` or `worst_sweep_change`;
those names approximate "takes the outer tolerance". It fails if one of them measures this change, or
decides on it, itself, through the shapes its docstring lists (#2512). `NewtonMFGSolver` stops on the
same kind of quantity, its Picard residual, the map's output against its input; its map is the Jacobi
one, where `FixedPointIterator`'s is Gauss-Seidel, with the same fixed point. After its warm-up and at
each iterate its Newton loop evaluates, it decides through `sweep_change` and
`check_convergence_criteria` on `solve(tolerance=)` (#2565).
`HomotopyContinuation`'s corrector, with its own `corr_tol`, and `PrimalDualMFGSolver`, which stops on an
optimisation residual and its own density change, are separate algorithms with their own tolerances.
This convention does not govern them (#2573).

**The inner Newton tolerance bounds the grid-scaled residual norm**, and
`base_hjb.hjb_residual_norm` is its owner on every Newton path (ruled 2026-10-08). So a tolerance means
the same thing under refinement and across schemes. *Not yet met (#2429, #1885): the owner's one caller
is the 1-D FDM path's `newton_hjb_step`. The n-D FDM and GFDM Newton paths compare an unscaled
`np.linalg.norm(residual)` against the same `DEFAULT_NEWTON_TOLERANCE`. The weak-form Newton path, which
`HJBFEMSolver` and `MeshlessGalerkinHJBSolver` take with `use_newton=True`, stops on the mass-matrix norm
of the Newton step rather than on a residual, against its own default of `1e-6`. And the owner takes a
scalar `dx`, which can carry a uniform grid's cell volume but not a graded grid's or a scattered point
set's weights.*

### A kernel density bandwidth is a factor

**In kernel density estimation, a `bandwidth` argument is a factor multiplied by the sample standard
deviation**, as in `scipy.stats.gaussian_kde`, and an absolute width goes through a separately named
keyword. `"scott"` and `"silverman"` mean scipy's definitions: $n^{-1/(d+4)}$ and
$\left(n(d+2)/4\right)^{-1/(d+4)}$ (ruled 2026-10-08). The kernel is a Gaussian with covariance
factor² × the sample covariance, full in every dimension, as scipy builds it. A kernel length scale that is
not a density estimate, such as a graphon's, is not covered. The owner is
`utils/numerical/particle/interpolation.py`: `estimate_kde_bandwidth` states the rules; `kde_factor`,
`kde_kernel_covariance` and `kde_kernel_widths` turn a bandwidth into a factor, a kernel and the kernel's per-axis
widths, refusing anything else by name; and `kde_sample_has_spread` decides, at the conditioning bound it states,
whether a sample has a kernel (#2556).
The library's own estimators read it there. The reflection estimator adds the sample's kernel at each image
of the evaluation point across a nearby wall, which is the method of images with each particle's image
carrying the reflected kernel; so its derivative normal to a wall vanishes on the wall for any sample covariance,
for a kernel narrower than the domain (one image per wall: a kernel that reaches the far wall would need the images'
images).
*Not yet met (#2556): public readers that take a number as an absolute width (`ParticleMeasure.to_density`,
`torch_utils.GaussianKDE`, `ParticleDensityQuery(bandwidth=)`, `beta_kde`, `create_ghost_particles`, and the
Wendland kernel of `gaussian_kde_gpu`), and `ParticleDensityQuery`, which states the rules itself, so
`FPParticleSolver`'s meshfree density query (a callable drift with no initial density) reads its Scott rule whatever
`kde_bandwidth` is.*

---

## 10. Geometry

### Signed distance — two polarities, and they are not interchangeable

| object | convention |
|---|---|
| `Domain.signed_distance`, including `DifferenceDomain` | $\varphi < 0$ **inside the navigable domain**, $\varphi > 0$ outside it |
| an **obstacle's** SDF — `obstacle_sdf` for stencil filtering, `obstacles_sdf` for cloud geodesics | $\mathrm{sd} \le 0$ **inside the obstacle**, $> 0$ navigable |

These are opposite, deliberately: a domain answers *are you in the region we solve on*; an obstacle
answers *are you in the thing we avoid*. `DifferenceDomain(box, obstacle)` is a **domain** and takes the
first convention — passing its `.signed_distance` where an obstacle SDF is expected blocks every
segment that samples the navigable region. For one obstacle, pass that obstacle's own
`.signed_distance`; for several, the SDF of their union (`UnionDomain(...).signed_distance`).

Consequences that are easy to invert:

- masking a field to the navigable region is `raw[sd <= 0] = 0` for an obstacle SDF and
  `raw[sd > 0] = 0` for a domain SDF;
- a soft wall that is 1 at and beyond the wall and decays into the navigable region is
  $\exp(\min(\varphi, 0)/\varepsilon)$ for a domain SDF and $\exp(-\max(\mathrm{sd}, 0)/\varepsilon)$ for
  an obstacle SDF;
- $-\varphi$ is the distance to the nearest boundary of any kind; an obstacle's $\mathrm{sd}$ is the
  distance to that obstacle only.

**A normal derived from an SDF points out of whatever that SDF is negative inside.** $\nabla\varphi$
is the outward normal of the domain; $\nabla\mathrm{sd}$ points out of the obstacle, into the
navigable region. Name which SDF a normal came from before using it.

### Normals on a boundary condition

A boundary condition written $\partial u/\partial n = g$ uses the **outward normal of the
computational domain**, at every wall, in every dimension.

### The graph Laplacian

**$L = D - A$, positive semidefinite, with $D$ the weighted degree** $D_{ii} = \sum_j w_{ij}$, the node
strength (ruled 2026-10-08). It enters diffusion with a minus sign:
$\partial_t m = -\tfrac{\sigma^2}{2} L m + \dots$, so the graph Laplace operator is $\Delta_G = -L$. The
owner is one function that forms $L = D - A$ from a weighted adjacency matrix (ruled 2026-10-08);
this file names it `graph_laplacian(adjacency)`, in `geometry/graph/laplacian.py`.
`SupportsGraphLaplacian` stays the interface consumers type against; it is not the owner. Every
implementer calls the owner and supplies only its adjacency. *Not yet met (#2429, #1951): the owner does
not exist yet, and the geometry getters, `NetworkData`, the network backends and `LaplacianCoupling` each
form $L$ without it (#2429 lists the sites); `node_degrees` returns the combinatorial degree on the igraph and
networkit backends and the strength on networkx; `LaplacianCoupling.compute_fp_source`
returns $+\kappa (Lm)_i$ as an FP source term, so it enters as $\partial_t m = +\kappa L m$, the
opposite sign; the network FP solver assembles $-\tfrac{\sigma^2}{2} L m$ inline from edge weights
rather than reading $L$, and its inline comment's sign contradicts its arithmetic; and the protocol's
docstring writes the diffusion as $-\sigma^2 L m$, without the $\tfrac12$.*

### Particles at a wall

**One owner applies a particle's wall rule for each boundary type** (reflect, absorb or wrap), and a
boundary helper never mutates the caller's array (ruled 2026-10-08). On a box, reflection is owned by
`reflect_positions` in `geometry/boundary/corner/position.py`, the module #521 consolidated; absorption by
`absorb_at_walls` beside it; and wrapping by `wrap_positions` in `geometry/boundary/periodic.py`. Each rule's
arithmetic on one axis is `corner.fold_axis` and `periodic.wrap_axis`, on NumPy or a backend's array module: the fold
lands in [lo, hi] and the wrap in [lo, hi), as the result's dtype holds them, and only a result rounding left outside
is moved, under a refusal bounded by the rule's own rounding (#2830). An absorbed
particle is removed and never moved, and the mass it carried leaves through the wall (§ 9). Its exit position is where
its reader last saw it: the contact for a reader of the path, the end for a reader of the end. `corner.absorb_positions`
clamps a particle to the wall and keeps it, so it is not an absorbing wall; it is deprecated (#2550). A structural
guard (`test_particle_wall_has_one_owner_2550.py`) fails if a function under `mfgarchon/` outside the owners applies a
box wall rule to positions in one of the shapes its docstring lists: a modulo, a mirror or a shift by the box, or a
clip to it. It allows by name the owners' per-axis rules (`fold_axis`, `wrap_axis`), the contact loop and the
deprecated `absorb_positions`, and code that is not a box wall rule for particles: the semi-Lagrangian
departure-point folds (#2815), a KDE's reflection ghosts, and clips with another purpose (a control's or a field's
bounds, a point read for its segment, the geometry API's projections).

**The particle solver reads a step by its path.** It ends each CPU step at its first contacts with the particles'
boundary (`reflect_at_first_contacts`, #2733): it reflects a particle specularly at a contact, absorbs one at a
contact on an exit, and at a contact on a periodic face goes on with the rest of the step from its image on the
opposite face (#2704). It reads the boundary condition only at a contact on the box. The contact loop is the path
rule's owner; on the box its reflections agree with `reflect_positions`'s fold to rounding, short of its cap on
reflections per step (`test_contact_loop_2733.py`).

**At a corner one of whose faces is periodic** (`applicator_particle.periodic_corner_actions`, ruled 2026-10-26), the
owner's corner order (priority, then DIRICHLET before a natural condition, then a placed segment before a global one,
then declaration order: `BoundaryConditions._corner_key`) decides only whether the particle is absorbed, the one bit a
corner cannot split per axis. If it is not absorbed, each face at the corner that the step crosses there, reaching its
plane within the tie band along the path, acts on its own axis by its own segment: a periodic face wraps its axis, and
any other face reflects the particle along it. A face the step moves along or away from, or toward without reaching
it within the band, is not crossed there and takes no action; the path meets it later or not at all. Where the rule
bites differs by reader. For a
path it decides a tie, a step through a corner within the contact tolerance, which has probability zero outside that
band, and a step that starts at a corner, as placed initial particles can; it is there so the answer is determined.
For a reader that has only a step's end, an end past two faces has positive probability, and the rule carries real
weight there (#2550).

**A reader that has only a step's end** (`ParticleApplicator.apply`, public and no longer on the solver's path) reads
the end at its nearest point of the box. An end at or past a face, within the reader's reach band, is at that face. The
segment its nearest point takes, on one face that face's and at a corner the owner's corner order, decides whether the
particle is absorbed. If it is not, each face the end is at or past acts on its own axis by its own segment: a periodic
face wraps the axis and any other folds it (`reflect_positions`). A face the nearest point is read on but the end has
not reached takes no action on its axis; at a corner its segment still takes part in the corner order, so it can decide
the absorption. This is the tie rule above, with "the step crosses it there" read as "the end is at or past it"
(#2550, #2803's shape).

**The bands**, in one place. An end reader's reach band is its own (`walls_reached`): `ParticleApplicator` takes an end
within `ONWALL_TOL` = 1e-10 of a face, absolute, as at it; `MeshfreeApplicator`'s absorbing mode takes a particle on or
past its geometry's box, band 0, or on its geometry's boundary at the geometry's own band (`is_on_boundary`; for an
`ImplicitDomain` `SDF_BOUNDARY_TOL` = 1e-8 on the signed distance). `ParticleApplicator` reads the segment at the
nearest point within `SDF_BOUNDARY_TOL` + |bound| · `BOUNDARY_REL_TOL` = 1e-8 + |bound| · 1e-12 of a face
(`_faces_at_point`), so it can read a face the end has not reached, which then takes no action on its axis. The path
reader's tolerance is `CONTACT_EPS` = 1e-12 of the box's diagonal (#2733 D4). A non-finite position is refused by
name, by the contact loop (`ImplicitDomain.first_contact`) and by the end readers' owner (`walls_reached`) alike.

**The 1-D helpers** (`utils/particle_utils.apply_boundary_conditions_numpy` / `_gpu`). The NumPy one folds and wraps
through the owners. The particle solver's 1-D GPU path calls the GPU one, which folds and wraps on the device through
the owners' per-axis rules (#2830); `test_particle_wall_owners_2550.py` pins it against the owners. Both refuse
"dirichlet" by name, since a function that returns positions cannot remove a particle and a clamp keeps the mass
that should leave; both refuse a non-finite position, as the owner's reach does. No helper measured mutates the caller's
array.

*Not yet met (#2693): on a geometry that is not a box, `MeshfreeApplicator`'s particle modes read its bounding box.*

### Pointwise and bulk must agree

Where a routine can produce a normal, a weight or a stencil either pointwise or in bulk, the two are
one implementation, and a neighbour set that changes size invalidates any matrix built from the
previous one. *Not yet met (#2429): SDF normals disagree between the pointwise and bulk paths at a
singular point, and three finite-difference SDF-gradient implementations coexist.*

### A shared Neumann condition

On a BC the HJB and the FP share, a NEUMANN face is the HJB's: $\partial u / \partial n = g$, the
boundary cost per unit of boundary local time. `NEUMANN(0)` is the reflecting pairing, and the FP reads it
as zero total flux, $J \cdot n = 0$ (`NO_FLUX`; ruled 2026-10-08). A `NEUMANN(g)` with $g \ne 0$ is
refused at the FP (user ruling 2026-10-09): $g$ says nothing about the agents' mass at the wall, and a
mass flux through the wall is not a Neumann condition. The two equations' BCs are then specified
separately: the shared BC keeps `NEUMANN(g)` for the HJB, and the FP solver is given
`boundary_conditions=no_flux_bc(...)` for reflected agents. A value that is not provably zero, such as a
callable, counts as $g \ne 0$. `BaseFPSolver` owns the reading, through `bc_utils.fp_view_of_shared_bc`, and
`get_boundary_conditions()` and each FP solver's own resolution end in it. A NEUMANN handed to an FP
solver explicitly would mean $\partial m / \partial n = g$, which no FP solver implements, so it is
refused, $g = 0$ included; a reflecting FP wall is `NO_FLUX` (#2512, row B3).

### A Robin condition at the FP

An FP Robin condition, for a constant $D$ (FP-FEM, its one consumer, refuses a varying one), is a condition on the
total flux $J = v m - D \nabla m$: `ROBIN(alpha, beta, g)`
means $J \cdot n = (D / \beta)(\alpha m - g)$, equivalently $\alpha m - (\beta / D)\, J \cdot n = g$ (user
ruling 2026-10-10). It is not $\alpha m + \beta\, \partial m / \partial n = g$ where the drift crosses the
wall: written as conditions equal to $g$, the two differ by $(\beta / D)(v \cdot n)\, m$. With $g = 0$ it is
an exit at a finite rate, $J \cdot n$ proportional to $m$. `FPFEMSolver` imposes it from its own
`boundary_conditions`, because its weak form leaves $J \cdot n$ on the boundary; no other FP solver accepts a
ROBIN.

On a BC the HJB and the FP share, a ROBIN face is the HJB's, $\alpha u + \beta\, \partial u / \partial n = g$,
and it is refused at the FP whatever its coefficients: they do not say what the FP's would be. As for a
shared Neumann value, the shared BC keeps the ROBIN for the HJB, and the FP solver is given its own
`boundary_conditions`. `bc_utils.fp_view_of_shared_bc` owns the refusal (#2512).

### A prescribed flux at the FP

`FLUX` prescribes the total flux, $J \cdot n = q$ with $J = v m - \nabla(D m)$ (§ 5) and $n$ the outward normal: $q$ is
mass per unit time per unit boundary measure, outward positive, and $q = 0$ is NO_FLUX (user ruling 2026-10-09,
B-1). It is a condition on the FP alone. A BC the HJB and the FP share carries none, and
`MFGProblem.get_boundary_conditions` refuses one there (B-2); an FP solver takes it as its own
`boundary_conditions`. FP-FDM's implicit step puts $-q/(\Delta x/2)$ on the half cell of each wall node per FLUX
face, and FP-FEM the load $-\int_\Gamma q\,\varphi\,dS$, each with its operator unchanged. So each step changes the
mass by exactly $-\Delta t \int_\Gamma q(t_{k+1})\,dS$, the integral on the scheme's own face quadrature (FP-FDM's
nodal values on the trapezoid face weights, FP-FEM's facet quadrature), on a step that conserves mass under NO_FLUX,
with no Dirichlet face (a Dirichlet row exchanges mass of its own),
and, on FP-FDM, where its positivity clip leaves the step alone.

B-3 refuses a step where the flux takes the density negative past the gate (`refuse_what_a_flux_takes_negative`, one
owner for both solvers; user rulings 2026-10-09 and 2026-10-30). First, where the negatives the flux's part makes, at
nodes the step without it leaves non-negative, are more than `MAX_CLIP_MASS_FABRICATION` of the step's mass. Otherwise,
where the step without it is within the gate and either the flux's part deepens that step's negatives past it, counted
without netting on the mass of the step that holds them, or the step with the flux is past the gate itself, as a flux
that drains the positive mass makes it; neither refuses a step with no negative. Any other step whose own negatives are
past the gate is the scheme's, the flux's share included: FP-FDM's guard refuses it, and FP-FEM keeps its negative
values and reports them, as on any step. Every other solver refuses FLUX by name (#2581).

### A capacity-limited exit at the FP

A `flux_capacity` $c$ on an exit caps its outflow per unit boundary measure: on the exit, $m \ge 0$,
$J \cdot n \le c$ and $m\,(c - J \cdot n) = 0$, a local complementarity, so the excess queues inside the domain (user
ruling 2026-10-09, A-1 and A-3). It is offered for a standalone FP solve only, and every coupled solver refuses an FP
solver whose BC carries one: paired with an HJB that reads $u = g$ at the exit it is not an equilibrium, since as
$c \to 0$ the FP side becomes a wall while the HJB side stays an exit (user rulings 2026-10-10 and 2026-10-31). The
exit a coupled MFG is to use is the finite-rate one, #2581's PR 3, not yet implemented. The refusal reads the FP solver
a coupled solver is handed: a user's object that wraps a capped FP-FDM and hides its BC is not seen, as a user's own
coupling loop around `solve_fp_system` is not (user ruling 2026-11-01). FP-FDM honours it on a `DIRICHLET` exit with value 0 of
its own `boundary_conditions`, on `divergence_upwind`, as an active set within each implicit step: every exit node
starts absorbing; a node whose outflow over the step would exceed $c\,|\text{face}|\,\Delta t$ takes the `FLUX(c)` row,
and the step is solved again until no absorbing node exceeds its cap. A node's outflow is read off its no-flux row, the
one the scheme already assembles at a no-flux wall, so the trapezoid mass plus what the exits let out is the mass at
the returned level 0 plus what a source added and a `FLUX` wall let in. (In 1-D a BC whose segments are all
`DIRICHLET` holds its uncapped walls at $g(0)$ at level 0, #2535; a capped wall keeps the caller's density there.)
A cap on the shared BC, on another type, on a `DIRICHLET(g \ne 0)`, on `divergence_centered` (where the
complementarity can have two roots at a node), on a face that meets a face not governed by a `NO_FLUX` segment, or on
any other solver is refused by name (#2581, #2575). That rule is read on the faces, so which segment governs a corner,
by priority or declaration order, does not decide it. A corner of the capped face that a `NO_FLUX` face governs is a
wall node: the cap then holds on the capped face less that corner's half cell, an $O(h)$ difference.

### What is not settled here

The remaining boundary-condition convention — ghost-cell placement on vertex- versus cell-centred grids —
is being reworked. Until that lands, the open boundary-condition issues (#1456,
#2005) are the record, not this file.

---

## 11. Component ownership

The problem decomposes into components, each parameter owned by exactly one:

| component | holds | parameters |
|---|---|---|
| `Model` | the game rules | `hamiltonian` or `lagrangian`, `volatility`, `volatility_kind`, `coupling_cost`, `terminal_coupling` |
| `Conditions` | the problem data | `u_terminal`, `m_initial`, `T` |
| domain | the spatial arena | the grid or geometry |
| `MFGProblem` | the assembly | `model` + domain + `conditions` + `Nt`, and the assembly-level `source_term_hjb` and `state_penalty` |

`Conditions` holds **callables**, not arrays, so the same conditions work at any resolution. $T$ is
physics and belongs to `Conditions`; $N_t$ is discretisation and belongs to the assembly.

Composition is by whole component — `with_model`, `with_domain`, `with_conditions` — and single fields
change through `dataclasses.replace`. There are no per-field shortcuts.

*Not yet met (#2429): `Model(lagrangian=...)` cannot yet be assembled; `coupling_cost` and
`terminal_coupling` bind only on a `TensorProductGrid` and through HJB-FEM P1 on a simplex mesh
(#1642); and `Model`
defaults the volatility to `0.1` where the legacy constructor gives `0.0`. There is no stationary (ergodic) problem: `ErgodicConditions`, which
declared one that no `MFGProblem` accepted, is removed (#2648).*

A **legacy constructor**, the components= form `MFGProblem(geometry=, components=, T=, Nt=, …)`, also exists. It
emits a `PendingDeprecationWarning`: the model/domain/conditions form is planned to replace it, and nothing needs
to change now (#2184). `volatility=`, `volatility_kind=` and `diffusion=` as `MFGProblem` keywords exist only there.

---

## 12. Names

`potential` is $V$, `coupling` is $f$, `source_term_hjb` is $S$, `volatility` is $\Sigma$,
`diffusion` is $A$, `control_cost` is the control-cost object and `lambda_` the scalar weight of any
control cost.

**The terminal cost $g$ has two canonical names**, and the pair is § *Solver method surface*'s case
rule rather than a duplication: `u_terminal` is the attribute, `U_terminal` the solver-method
parameter.

**Case tracks the owner, not the mathematical status.** Attributes of `Conditions` and `MFGProblem`
are lowercase whatever they hold — `conditions.u_terminal` is a callable and `problem.u_terminal` is
an array of the grid's shape, both lowercase, because the construction evaluates the callable once and
keeps both. Uppercase belongs to the solver methods' array parameters. Reading the case as "callable
versus array" is wrong in exactly the place it looks right.

**`control_cost` names two things.** On a Hamiltonian it is the `ControlCostBase` object. On a
control-cost constructor it is a scalar spelling of `lambda_` that is deprecated in documentation;
passing both scalars raises, and passing the object together with `lambda_` on its constructor is the
ordinary idiom. *Not yet met (#2429): the constructor keyword emits no warning, and three public
functions still take `control_cost` as a scalar.*

`obstacles` (plural) are geometric regions; the retired `obstacle` was a penalty, now `state_penalty`
or a `constraint=ObstacleConstraint(...)`.

### Symbols or words

A name is a mathematical symbol or a descriptive word according to what it stands for.

- **A symbol for the horizon and the discretisation**: `T`, `Nt`, `Nt_points`, `dt` and `tSpace` on the problem,
  `Nx` and `Nx_points` on the grid; and `lambda_`, the control-cost weight. Inside an algorithm, a
  local name may be the symbol of the formula being implemented (`p`, `dx`); a solver-method array
  keeps § *Solver method surface*'s uppercase (`U`, `M`).
- **A word for configuration and for the model's terms**: `max_iterations`, `tolerance`, `relaxation`;
  `potential`, `coupling`, `volatility`, `diffusion`. (A scalar coefficient: § *Not settled here*.)

The volatility shows where the line falls: a formula writes it $\Sigma$, or $\sigma$ when it is a
scalar (§ *Volatility and diffusion*), and the public API names it `volatility` (#2375 ruling 6).
*Not yet met (#2429): `VariationalMFGComponents` takes it as `noise_intensity`,
`MFGProblem.get_diffusion_coefficient_field` as `override`, and internal helpers as `tensor_field`,
`sigma_tensor`, `Sigma` or `override`.*

### Retired

**Refused**, with an error naming the replacement:

| retired | use |
|---|---|
| `sigma` (the agents' volatility, on every public API), `problem.sigma` | `volatility` |
| `sigma_kind` | `volatility_kind` |
| `sigma_at_n` | `volatility_at_n` |
| `sigma_function` | `volatility_function` |
| per-solve `volatility_field`, `volatility_matrix`, `diffusion_field`, `tensor_diffusion_field` | `volatility=` with `volatility_kind=` — the per-solve keyword always carried the volatility |
| `problem.volatility_field`, `problem.diffusion_field` | `problem.volatility`, `problem.diffusion` |
| `obstacle=` | `state_penalty`, or `constraint=ObstacleConstraint(...)` |
| `MFGComponents(potential_func=)` | the Hamiltonian's `potential=`, e.g. `SeparableHamiltonian(potential=...)`. It was stored and no solver added it to $H$ (#2554) |
| `MFGProblem(gamma=)` | the Hamiltonian's `coupling=` (with `coupling_dm=`). It was stored and nothing read it (#2554) |
| `damping_factor` on `FixedPointIterator` | `relaxation` |
| `OptimizationSense`, `sense`, `sense_sign` | nothing — § *The library minimises* |
| `AdjointConsistentProvider`, `NormalDriftProvider` | nothing — removed |
| `problem.spatial_discretization` | `problem.geometry.Nx` (intervals) or `problem.geometry.Nx_points` (points). It held one count or the other depending on the constructor (#1889); the `spatial_discretization=` argument stays and counts intervals |

**Removed**, with no named error: `problem.f_potential` and `problem.gamma`, which held the two inputs
retired above (#2554). Reading either raises a plain `AttributeError`.

**Deprecated**, still accepted with a warning: `num_points` → `Nx_points`; `damping_factor` →
`relaxation`, `damping_factor_M` → `relaxation_M` and `damping` → `relaxation` elsewhere; `u_final` →
`u_terminal`; `velocity_field` → `drift_field`; `m_initial_condition` and `m_initial` → `M_initial`;
`drift_field` → `potential_field` on the semi-Lagrangian pair, `FPNetworkSolver` and the weak-form
family; `ControlCostBase.control_cost` →
`lambda_`. When each goes is recorded in `docs/user/DEPRECATION_MODERNIZATION_GUIDE.md`, which is
generated from the code.

`sigma` remains correct as the parameter of another distribution or kernel: a stochastic process's
own volatility (`OrnsteinUhlenbeckProcess(sigma=...)` and its siblings), a Lévy jump width
(`GaussianJumps`), a Gaussian blur width. Some names still say `sigma` while carrying the diffusion or
covariance tensor (#2426).

---

## 13. Not settled here

Known to be unstated or open. A claim about these must not be read out of this file.

- Discounting: whether a discount factor multiplies the running cost, the value function, or enters
  the Hamiltonian as a zeroth-order term.
- The Lévy compensator convention for jump models.
- The optimal-transport ground cost and how it scales with the time horizon.
- Boundary data when $\Sigma = 0$, where the equation degenerates and the admissible number of
  boundary conditions drops.
- The sign and normalisation of the reported Nash gap.
- The remaining boundary-condition family (§ *Geometry*).
- Whether a scalar coefficient of the model or of a method takes its textbook symbol or a word. The
  library has both: `lambda_`, `GFDMConfig.delta` and `WENOConfig.epsilon` are symbols;
  `discount_rate` and `coupling_coefficient` are words.
