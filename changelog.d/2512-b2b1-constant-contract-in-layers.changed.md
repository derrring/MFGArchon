- **A boundary value's constant contract is read in layers, and a Robin segment's alpha and beta have one reader**
  (#2512, row B2's second half). One function, `_read_constant`, reads a constant, what a callable returns, and
  what a `BCValueProvider` computes. It works in three layers:
  1. **dtype.** An array, masked or not, must have an integer or a floating dtype. A boolean, complex, object,
     string, bytes, datetime or timedelta array is refused at the container, whatever it holds. A boolean or
     complex scalar is refused too.
  2. **One unwrap.** An array of one element, of any shape, masked or not, is that element. A return of more than
     one number is refused.
  3. **The value.** `None` is 0 for a constant, and refused for a return. A masked element, and whatever reads as
     zero, is 0. A finite real is its value. Anything else, and a non-finite number, is refused.

  The first half of the row read one object level and looked inside object arrays. These now behave differently:
  - **An object array is refused, whatever it holds.** That includes one holding `0.7`, which was read as 0.7. It
    also covers the corners the first half's reviews found: an object array holding `True` (read as 1.0 before the
    first half's fix), a doubly nested `False` or complex 0 (read as 0), and a doubly nested `0.7` or `None`.
  - **A fully masked one-element boolean or complex array** was read as 0. It is refused.
  - **A string or a timedelta array** was refused by type, or raised an unnamed `TypeError`. It is refused by
    dtype, by name.
  - **`np.matrix([[0.7]])`** was refused; it is read as 0.7, as any one-element array is.
  - **Exceptions to "a container reads as its element":** a fully masked numeric array is 0 (the mask means no
    datum), and a constant of more than one element is read whole: 0 if all zero, refused otherwise.
- **A provider's result is one number, read as a callable's return is**, by `boundary_value` with a `state`.
  - Before, the owner returned a provider's non-scalar result raw. It is now refused, naming #2600, the planned
    per-node capability.
  - **A regression of the first half, restored.** `resolve_bc` with `HJBResolver` resolved a provider on a Robin
    segment's value with the solver state; the first half's passthrough read it with no state and refused it. The
    passthrough now passes the state to the owner, for `HJBResolver` and `FPResolver`, and the resolver's own
    provider branch, which no longer ran, is gone. `FPResolver`, which has no such branch, read the same value as
    0.0 before the first half; it resolves it now.
  - **Two paths still read a provider their own way:** the HJB resolver's NO_FLUX/REFLECTING branch with its own
    `float()`, and `with_resolved_providers`, whose result a reader then reads as a constant.
- **A Robin segment's `alpha` and `beta` are read by one reader, `segment_robin_coefficients`.**
  - It replaces 24 reads, on 19 lines, in 11 functions. Each took its own `getattr` default or `float()`, or passed
    the raw field.
  - Each coefficient is read by the constant contract above, and a refusal advises a number, not a callable. A
    callable coefficient is refused, naming #1237; so is `None`, and a field from a provider is refused naming
    #2600.
  - Only a Robin segment's coefficients are read: the readers use them in their Robin arm alone. So a Dirichlet,
    Neumann or no-flux segment solves as before whatever its `alpha` and `beta` hold.
  - **These change for a Robin segment:**
    - FEM's Robin operator refused a NumPy scalar coefficient that is not a Python `float` (`np.float32(2.0)`); it
      reads it now.
    - FEM and the FDM per-face ghosts read `alpha=True` as 1.0; it is refused.
    - The uniform FDM ghosts raised on a one-element array coefficient; they read it.
  - The particle applicator's wall classifier reads `beta`'s sign itself, and is not one of these readers.
- **FEM's Robin and Neumann loads stay one constant per segment (#1237).** Their refusal of a callable `g` now
  names #1237.
