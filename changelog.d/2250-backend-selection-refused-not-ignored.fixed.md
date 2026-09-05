- **Both coupling iterators refuse a backend they cannot run on, instead of accepting and ignoring it**
  (Issue #2250). `FixedPointIterator` and `FictitiousPlayIterator` annotated `backend` as `str | None`
  and documented it as a name, then used it as an *object* in exactly one place — the cold-start
  allocation `self.U = self.backend.zeros(...)`. Every non-`None` value raised
  `AttributeError: 'str' object has no attribute 'zeros'`, **including `"numpy"`**, the backend
  already in use. The `is not None` guard hid it: the default is `None`, which took the `np.zeros`
  branch, so the whole suite and every example ran the working path, and no test in the repository
  constructed either iterator with a non-`None` backend.

  **Resolving the name was measured and rejected.** With `create_backend(backend)` substituted in, the
  allocation succeeds and the loop dies further in — `JAX arrays are immutable and do not support
  in-place item assignment`, and `can't assign a numpy.ndarray to a torch.DoubleTensor`. That
  allocation was the *only* use of `self.backend`, so the parameter bought one array of a type the
  rest of the loop cannot write into: wired and inert, not unwired. Wiring it further would have been
  a green diff that fixes nothing. #1922 records the missing capability — "selecting a backend is not
  an operation this package supports" — and names this same `__setitem__` wall.

  `None` and `"numpy"` both describe what actually runs, so both normalise to `None`; anything else
  raises `NotImplementedError` naming #1922. One owner in `base_mfg.resolve_supported_backend`,
  because both iterators held the same three lines and fixing one would have left the other. The
  backend-allocation fork is gone with it — this change made it unreachable.

  Reached from the config surface: `config.backend.type` began arriving through `config/translator.py`
  in `5610e1af`, which fixed #1284's *dropped* fields by delivering one to a consumer that could not
  accept its type.
