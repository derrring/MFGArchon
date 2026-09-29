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
    return -float(np.sum((np.asarray(x) - 0.5) ** 2))  # what the agents would maximise


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
- HJB semi-Lagrangian's update assembles control cost $+\,V + f$ from the same object, as
  $H(p) - 2H(0)$.
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
copy. Solvers that advect with $-c\,\nabla u$ read $c$ from `fp_drift_coefficient(problem)`; FVM, FEM
and meshless-Galerkin take $\alpha^*$ from `H.optimal_control` on their own basis.

`fp_drift_coefficient(problem)` resolves in this order:

1. A quadratic `SeparableHamiltonian` → $1/\lambda$, taken from its control cost. **This is the path a
   well-formed problem takes.**
2. A `SeparableHamiltonian` that is *not* quadratic → **raises**. The scalar form $-c\,\nabla u$ does
   not represent that optimal control.
3. Any Hamiltonian that is not a `SeparableHamiltonian`, or none at all → the problem's
   `coupling_coefficient`, a legacy scalar.
4. Neither of the above → **raises**.

`coupling_coefficient` is **not** an alternative spelling of the control cost and must never be set
to stand in for one.

### What the Fokker–Planck solver receives

Two channels, and each FP solver declares which one it reads (`_drift_convention`):

- **`potential_field` carries the value function $u$.** On the default path — a smooth separable
  $H$ — the coupling layer passes `potential_field=U` to every FP solver that has that parameter, and
  the solver forms $\alpha^* = -D_pH$ itself. *Not yet met (#2429): `MultiPopulationIterator` passes
  the value function to network populations as `drift_field=`, the deprecated alias.*
- **`drift_field` carries a velocity $\alpha^*$.** The coupling layer computes it only for a
  non-smooth or non-separable $H$, and passes it only to a solver whose convention is `VELOCITY`
  (FDM, FVM, GFDM).

A solver whose convention is `VALUE_FUNCTION` — the particle solver, the two semi-Lagrangian solvers,
the network solver and the weak-form family — cannot represent the drift of a non-smooth or
non-separable $H$, and the coupling layer refuses to route $u$ to one rather than let it advect with
$-c\,\nabla u$. `FPGFDMSolver` has no `potential_field`, and the coupling layer refuses to hand it $u$
as a velocity. *Not yet met (#2429): called directly, the semi-Lagrangian pair and the particle solver
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
- **HJB-GFDM** builds them only with `monotonicity_scheme="qp_m_matrix"` or `"joint_socp"`; its
  default batch path passes gradient arrays of shape `(N, d)`.
- **HJB-FDM** never builds one. In $d \ge 2$ it evaluates $H$ in one batch on gradient arrays; in 1-D
  its residual and Jacobian call `problem.H()` per point with the legacy dict `{(0,): u, (1,): p}`.
- **HJB semi-Lagrangian** passes arrays and does not use them. No FP solver uses them.

---

## 5. Volatility and diffusion

### Two quantities, never conflated

From the Itô dynamics $\mathrm{d}X_t = \alpha_t\,\mathrm{d}t + \Sigma\,\mathrm{d}W_t$:

- $\Sigma$ is the **volatility** — the SDE's noise matrix, and the primitive the user supplies.
- $A = \tfrac12 \Sigma\Sigma^{\mathsf{T}}$ is the **diffusion** — the PDE's second-order coefficient,
  appearing as $\operatorname{tr}(A\,D^2u)$.

In one dimension this is $D = \sigma^2/2$. That is not a separate rule; it is the scalar case.

$\Sigma\Sigma^{\mathsf{T}}$, **not** $\Sigma^{\mathsf{T}}\Sigma$. For a square $\Sigma$ the two
coincide exactly when $\Sigma$ is normal ($\Sigma\Sigma^{\mathsf{T}} = \Sigma^{\mathsf{T}}\Sigma$) —
every symmetric $\Sigma$ is — so a non-normal $\Sigma$ gives two different matrices.

`problem.volatility` holds $\Sigma$ **as supplied** — a scalar, an array with its kind, or a
callable. `problem.diffusion` is $A$, derived on access. There is no scalar view of a non-scalar
volatility: a consumer that needs one calls `scalar_volatility(volatility, consumer=...)`, which
refuses a field, a tensor or a callable by name rather than averaging it.

### Where the volatility is supplied

In the v1.0 API the volatility belongs to the model: `Model(volatility=..., volatility_kind=...)`.
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
| `source_term_hjb` | `(t, x, v, m)` |
| `potential` | `(t, x)` |
| `coupling` | `(m)` |
| `ControlCostBase.optimal_control` / `.lagrangian` | `(p)` / `(alpha)` |
| volatility callable and PDE coefficient callables | `(t, x, m)` |
| a measure field's `field_fn` / `gradient_fn` | `(t, x, mu)` |

The order is chosen so the derivative family can grow: $D^2u$ is already a first-class object one
layer down (§ *Derivatives*), and a signature with $m$ between $x$ and $p$ leaves nowhere to put it.
Time-first also agrees with § *Arrays, grids and time* and with `scipy.integrate.solve_ivp`'s
`(t, y)`.

- **On the operator family, `t` has no default** and is always passed.
- **A time-independent callable may omit `t`.** The library still supplies it, binding the callable
  by name.
- **Currying the measure needs a keyword**: `partial(H, m=mu)`.

*Not yet met (#2429): the per-point adapter `problem.H` / `problem.dH_dm` is not time-first and
defaults its time to 0, and some per-point solver paths call it without one.
`LagrangianBase.proximal(tau, z, *, t=0.0, x=None, m=None)` puts its step arguments first and defaults
its time to 0, and several private helpers put `t` last: the finite-difference ones and
`DualHamiltonian._find_optimal_alpha` take `(x, m, p, t)`, `DualLagrangian._find_optimal_p` takes
`(x, alpha, m, t)`, and `NetworkHamiltonian._default_hamiltonian` takes `(node, m, p, t)`.*

**`u_terminal` and `m_initial` are space-only or time-first.** `lambda x: ...` is read at each point;
`lambda t, x: ...` is read at $t = T$ for `u_terminal` and $t = 0$ for `m_initial`. *Not yet met
(#2429): `StochasticMFGProblem`'s conditional problems call `u_terminal(x)` directly, so a time-first
one is refused there.*

### The library refuses the old order rather than misreading it

A callable written `lambda x, t: ...` would still *run* under this order and return wrong numbers,
because both arguments are numeric. Three mechanisms stop it:

- **When a callable is accepted**, its parameter names are read and an out-of-order signature is
  refused, with a message naming the expected order. This covers the potential, `source_term_hjb` /
  `source_term_fp`, measure fields, Hamiltonian and feedback callables, network callables, the
  variational Lagrangian, and `u_terminal` and `m_initial`. A solver's own `source_term` is checked at
  its first evaluation.
- **When a class is created**, a Hamiltonian or Lagrangian subclass, or an `MFGProblem` subclass's
  `hamiltonian` / `running_cost`, is refused if its methods are out of order.
- **Invocation by keyword.** A callable bound by name is called by keyword, so a correctly named one is
  safe. A nameless callable, a solver `source_term` and a variational Lagrangian are called
  positionally in slot order, except a `u_terminal` or `m_initial`: one that names nothing is read
  only if `x` alone works, or, in 2-D and 3-D, as expanded coordinates if it is a bare `*args`.

*Not yet met (#2429): volatility, drift and coupling callables are never bound or inspected.*

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
`NetworkHJBSolver(tolerance=)` and `HJBHowardSolver(max_iter=, tol=)` are inner-loop settings.

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
`grid.Nx`, `.Nx_points`, `.coordinates` (a per-axis list of coordinate arrays),
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
- **The measure has one owner**, `mfgarchon.utils.numerical.quadrature`, exposed on
  `TensorProductGrid` as `integrate` and `quadrature_weights`, and correct on graded axes. A bare
  `np.sum(m) * dx` is a *different functional* even on a uniform grid: it gives each end node a full
  cell and over-counts by $\tfrac{dx}{2}(m_0 + m_N)$. No other geometry carries an `integrate`; a
  network is measured on its nodes, and other geometries fall back to a named `uniform-cell` or
  `point-average` measure. *Not yet met (#2429): live sites outside that fallback still measure with
  `np.sum(m) * dx`, mostly particle/KDE diagnostics and convergence metrics.*
- **Mass conservation needs both the boundary condition and a conservative discretisation.** Under
  no-flux and periodic boundaries the divergence-form schemes, FVM and the splatting semi-Lagrangian
  solver conserve mass to rounding; FP-FDM's `gradient_upwind` and `gradient_centered` schemes do not,
  and can lose most of it. Under absorbing boundaries the invariant is that the mass lost equals the
  accumulated boundary flux. Conservation and discrete adjointness are separate properties, tested
  separately.
- **Renormalisation is a per-solver property, not a convention.** The deprecated `FPSLJacobianSolver`
  rescales to the pre-step mass at every step; `FPSLSolver` does not, because its splatting conserves;
  `FPParticleSolver` pins every slice to the caller's mass under `kde_normalization="all"`. Comparing
  $\int m$ across solvers therefore measures the solver as well as the physics.

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

### Pointwise and bulk must agree

Where a routine can produce a normal, a weight or a stencil either pointwise or in bulk, the two are
one implementation, and a neighbour set that changes size invalidates any matrix built from the
previous one. *Not yet met (#2429): SDF normals disagree between the pointwise and bulk paths at a
singular point, and three finite-difference SDF-gradient implementations coexist.*

### What is not settled here

The remaining boundary-condition conventions — the relationship between `NO_FLUX` and a zero Neumann
condition, the Fokker–Planck flux form at a wall, and ghost-cell placement on vertex- versus
cell-centred grids — are being reworked. Until that lands, the open boundary-condition issues (#1456,
#2005) are the record, not this file.

---

## 11. Component ownership

The problem decomposes into components, each parameter owned by exactly one:

| component | holds | parameters |
|---|---|---|
| `Model` | the game rules | `hamiltonian` or `lagrangian`, `volatility`, `volatility_kind`, `drift_field`, `coupling_cost`, `terminal_coupling` |
| `Conditions` | the problem data | `u_terminal`, `m_initial`, `T` |
| domain | the spatial arena | the grid or geometry |
| `MFGProblem` | the assembly | `model` + domain + `conditions` + `Nt`, and the assembly-level `source_term_hjb` and `state_penalty` |

`Conditions` holds **callables**, not arrays, so the same conditions work at any resolution. $T$ is
physics and belongs to `Conditions`; $N_t$ is discretisation and belongs to the assembly.

Composition is by whole component — `with_model`, `with_domain`, `with_conditions` — and single fields
change through `dataclasses.replace`. There are no per-field shortcuts.

*Not yet met (#2429): `Model(lagrangian=...)` cannot yet be assembled; `coupling_cost` and
`terminal_coupling` are read by nothing; `ErgodicConditions` (`m_stationary_guess`, `discount_rate`)
is declared but no `MFGProblem` accepts it; and `Model` defaults the volatility to `0.1` where the
legacy constructor gives `0.0`.*

A **legacy constructor** — `MFGProblem(geometry=, components=, T=, Nt=, …)` — still exists and emits
a `DeprecationWarning`. It is the migration surface, and new code does not use it. `volatility=`,
`volatility_kind=` and `diffusion=` as `MFGProblem` keywords exist only there.

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
| `damping_factor` on `FixedPointIterator` | `relaxation` |
| `OptimizationSense`, `sense`, `sense_sign` | nothing — § *The library minimises* |
| `AdjointConsistentProvider`, `NormalDriftProvider` | nothing — removed |

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
