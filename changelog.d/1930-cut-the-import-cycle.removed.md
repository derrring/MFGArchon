`mfgarchon.utils.numerical` no longer re-exports the GFDM strategy classes.

`BoundaryHandler`, `DifferentialOperator`, `DirectCollocationHandler`, `GhostNodeHandler`,
`LocalRBFOperator`, `TaylorOperator`, `UpwindOperator`, `create_bc_handler` and
`create_operator` live in `mfgarchon.alg.numerical.gfdm_components.gfdm_strategies`; import
them from there. No code in the repository used the `utils` spelling.

The re-export was the back-edge of an import cycle — `mfgarchon.utils` sits below
`mfgarchon.alg`, and importing upward reached `fp_network`, which needs a name from
`utils.numerical` while that module is still executing. The cycle was invisible in normal use
because an eager import elsewhere completed `utils.numerical` by another route first, and only
surfaced when that eager import was made lazy. Removing it is what makes that deferral possible.
(#1930)
