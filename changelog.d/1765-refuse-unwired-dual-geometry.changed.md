`MFGProblem(hjb_geometry=..., fp_geometry=...)` now raises `NotImplementedError` when the two
geometries differ (#1765). They were accepted, a `GeometryProjector` was built, and the
distinction was then discarded — `self.geometry` is set from the HJB one and is the single
attribute every solver reads, so a 41-point HJB grid with an 11-point FP grid returned
`M.shape == (11, 41)` with the FP solver logging the HJB grid's `dx`. No error, no warning, and
every downstream number computed on a grid the caller did not choose. Identical geometries are
still accepted (equivalent to the unified path), and `GeometryProjector` is unchanged and still
usable directly. Wiring the projector into the coupling loop is the follow-up; refusing is the
honest state until then.
