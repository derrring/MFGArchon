- **Both coupling iterators resolve a backend name, pass through an object, and refuse one that**
  **cannot carry a solve** (Issue #2250). `FixedPointIterator` and `FictitiousPlayIterator` annotated
  `backend` as `str | None` and documented it as a name, then used it as an *object* in exactly one
  place — the cold-start allocation `self.U = self.backend.zeros(...)`. Every non-`None` string raised
  `AttributeError: 'str' object has no attribute 'zeros'`, **including `"numpy"`**, the backend
  already in use. The `is not None` guard hid it: the default is `None`, which took the `np.zeros`
  branch, so the whole suite and every example ran the working path, and no test in the repository
  constructed either iterator with a non-`None` backend.

  **A blanket refusal of anything but `None`/`"numpy"` was measured and rejected**, not merely
  proposed: a `NumPyBackend` **object** solved correctly on the pre-fix tree, and
  `examples/basic/solvers/acceleration_comparison.py` assigns one post-construction, so refusing
  objects would have been a capability regression rather than a fix. `resolve_backend` (in
  `base_mfg.py`) resolves a string via `create_backend`, passes a `BaseBackend` instance through
  untouched, and raises `TypeError` on anything else — called once at construction (fail fast on an
  obviously bad value) and again inside `allocate_state_arrays` (catch a name or a bad type assigned
  to the public `self.backend` attribute *after* construction, which a constructor-only check cannot
  see).

  **Whether the array can be written is a separate question**, checked at the allocation, not the
  constructor: a JAX array is immutable and a torch tensor rejects a numpy right-hand side, so
  allocating from those backends produced arrays that failed later, deep in the solve, with a message
  about the array rather than the configuration. `allocate_state_arrays` probes one write with the
  value type the loop actually uses — assigning a numpy row, not `U[0] = U[0]` — and raises
  `NotImplementedError` naming #1922 (the capability: "selecting a backend is not an operation this
  package supports") if the write fails. The distinction matters: self-assignment is symmetric and
  cannot see an asymmetric defect. Measured on `(3, 4)`: assigning a numpy row gives
  `ok / TypeError / TypeError` for numpy / jax / torch, while self-assignment gives
  `ok / TypeError / **ok**` — a torch tensor accepts its own element back and rejects the numpy array
  the loop hands it, so a self-assignment probe would have passed torch straight into the same crash
  this issue exists to remove.

  One owner in `base_mfg.py` (`resolve_backend`, `allocate_state_arrays`), because both iterators held
  the same three lines and fixing one would have left the other. Reached from the config surface:
  `config.backend.type` began arriving through `config/translator.py` in `5610e1af`, which fixed
  #1284's *dropped* fields by delivering one to a consumer that could not accept its type.
