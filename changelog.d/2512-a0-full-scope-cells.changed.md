- **`scripts/oracle_pins.txt` counts the particle FP's cells and FP-FEM's prescribed-density pair: 67 oracle cells, 53
  before** (#2512). (a) covers every shipped solver family × each BC it declares (user ruling 2026-10-25, comment
  6395444642); the manifest counts a cell for each type in `_SUPPORTED_BC_TYPES`, in 1-D and 2-D, on the default
  configuration.
  - The 14 new cells are `-`: the particle FP's 12 (the main pair's FP) and FP-FEM's two prescribed-density cells. No
    test changed, so the gate's oracle line went from 37/53 to 37/67.
  - An off-default family (HJB-SL, HJB-WENO, FP-FVM, FP-SL, meshless) gains its cells when its batch lands; a BC its
    batch refuses leaves `_SUPPORTED_BC_TYPES` and gains no cell (user ruling 2026-10-16, comment 6217184763).
    FP-GFDM is withdrawn (#2583) and declares no type until its rebuild (#2584); the network solvers declare none.
