- **`scripts/oracle_pins.txt` counts the particle FP's cells and FP-FEM's prescribed-density pair: 67 oracle cells, 53
  before** (#2512; user ruling 2026-10-25, comment 6395444642). (a) covers every shipped solver family, for each BC it
  declares in `_SUPPORTED_BC_TYPES`, in 1-D and 2-D, on its default configuration.
  - The 14 new cells are `-`: the particle FP's 12, the main pair's FP, and FP-FEM's two prescribed-density cells. No
    test changed, so the gate's oracle line goes from 37/53 to 37/67.
  - An off-default family (HJB-SL, HJB-WENO, FP-FVM, FP-GFDM, FP-SL, meshless, network) gains its cells when its batch
    lands, as an oracle or as a declaration change (user ruling 2026-10-16, comment 6217184763).
