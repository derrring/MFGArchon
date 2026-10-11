- **`mfgarchon.utils.deprecation.deprecated_value` deprecates an argument's form as well as its value** (#2597).
  - **A type key** matches by `isinstance`. It warns, naming the type and its replacement, and passes the value
    through for the function body to translate. A hashable key still remaps, as before; before this, an unhashable
    value such as a dict passed through with no warning.
  - **`deprecated_on` and `removal_blockers`** are recorded, as `deprecated_parameter` records them. The six-month
    arm of the removal policy then applies, and the deprecation ratchet reads a value's blockers instead of treating
    it as cleared.
  - **A `TypeError` raised by the decorated function** reaches the caller. On a remapped value it used to be caught,
    and the function ran a second time with the unremapped arguments.
  - **The warning is attributed past this module's own frames,** as `deprecated_parameter`'s is (#2579).
