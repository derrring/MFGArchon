Removed two deprecation shims that were overdue for deletion.

`mfgarchon.utils.numerical.anderson_acceleration` and
`mfgarchon.utils.numerical.gfdm_strategies` re-exported modules that live in `mfgarchon.alg`.
Both were deprecated in v0.18.0 and marked for removal in v0.21.0; the version is 0.22.0.dev0.

Import from the canonical locations instead:

    mfgarchon.alg.numerical.coupling.anderson_acceleration
    mfgarchon.alg.numerical.gfdm_components.gfdm_strategies

The deletion halves the module-level `utils -> alg` imports, from 4 to 2. That direction is an
inversion — `utils` sits below `alg` — and it is what makes the import graph unsplittable
(#1930).
