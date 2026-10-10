- **BREAKING. Three geometry trait protocols are removed: `ConnectivityAware`, `StructureAware` and
  `TraitAwareGeometry`** (from `mfgarchon.geometry` and `mfgarchon.geometry.traits`). Nothing in this
  repository used them. `git grep -nw` over every tracked file finds each only in `traits.py` and
  `geometry/__init__.py`'s import and `__all__`. Out-of-repo use cannot be measured, which is why this is
  marked breaking. `BoundaryAware` stays: `get_applicator_for_geometry` reads it for MESHFREE
  discretisation, to choose `ImplicitApplicator` for an implicit boundary. The three enums
  (`ConnectivityType`, `StructureType`, `BoundaryDef`) stay as well. Their future belongs to the dispatch
  consolidation.
- **BREAKING. `Geometry.is_cartesian`, `is_network`, `is_mesh` and `is_implicit` are removed.** They had
  no reader in tracked files outside their own docstring examples. Use
  `geometry.geometry_type == GeometryType.CARTESIAN_GRID` (`NETWORK`, `UNSTRUCTURED_MESH`, `IMPLICIT`).
  Not to be confused with `MFGProblem.is_network`, `is_cartesian` and `is_implicit`, which read
  `problem.domain_type` and are unchanged: `MFGProblem` reads `is_network` itself to choose how mass is
  computed, and the geometry-first API guide teaches all three.
