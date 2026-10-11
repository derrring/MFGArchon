- **A boundary value's constant contract is read in layers, at any depth, and a Robin segment's alpha and beta
  have one reader** (#2512, row B2's second half). One function, `_read_constant`, reads a constant, what a
  callable returns, and what a provider computes. It works in three layers:
  1. **dtype.** A boolean or a complex value anywhere inside the value is refused, masked or not, at any depth of
     object arrays.
  2. **One unwrap.** An array of one element, of any shape, masked or not, is that element, once. A value still
     nested in an array is refused. A return of more than one number is refused.
  3. **The value.** `None` is 0 for a constant, and refused for a return. A masked element, and whatever reads as
     zero, is 0. A finite real is its value. Anything else, and a non-finite number, is refused.

  The first half of the row read one object level. These now behave differently:
  - A fully masked one-element boolean or complex array was read as 0. It is refused.
  - A doubly nested object array holding `False` or a complex 0 was read as 0. It is refused.
  - A callable returning a doubly nested `0.7` was read as 0.7, and the same value given as a constant was
    refused. Both are refused now.
  - A doubly nested `None` was read as 0, both as a constant and as a return. It is refused.
- **What a `BCValueProvider` computes is read as a callable's return is: one number.** Before, the owner returned
  a provider's result raw when it was not a scalar. A value per node is a planned capability, #2600.
  `with_resolved_providers` still keeps a field resolved onto `alpha` or `beta`; every reader refuses it, naming
  #2600. The HJB resolver's Robin branch read a provider's value with its own `float()`; it now reads it through
  the owner.
- **A Robin segment's `alpha` and `beta` are read by one reader, `segment_robin_coefficients`.** It replaces 19 reads,
  in 11 functions, that each took their own `getattr` default or `float()`: in the FDM ghost layer and calculator
  route, the BC resolver, the meshfree applicator, the 1-D HJB read, HJB-GFDM, FEM's Robin operator and two
  `bc_utils`/`_compat` helpers. Each coefficient is read by the constant contract above.
  - A callable coefficient is refused, naming #1237.
  - `None` is refused.
  - A field from a provider is refused, naming #2600.
  - FEM's Robin operator refused a NumPy scalar coefficient that is not a Python `float`, such as
    `np.float32(2.0)`. It now reads it as 2.0.
- **FEM's Robin and Neumann loads stay one constant per segment (#1237).** Their refusal of a callable `g` now
  names #1237.
