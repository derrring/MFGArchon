- **`scripts/oracle_pins.txt` counts #2512 (a) at its full scope: 109 oracle cells, 53 before** (#2512; user ruling
  2026-10-25, confirming #2512 comment 6014400413). Every shipped solver family is counted for each BC it declares in
  `_SUPPORTED_BC_TYPES`, in 1-D and 2-D, on its default configuration; HJB-GFDM keeps a cell for each distinct
  discrete equation it ships.
  - The 56 new cells are `-`: HJB-SL, HJB-WENO, FP-FVM, FP-SL, the particle FP, meshless Galerkin FP and HJB, and
    FP-FEM's two prescribed-density cells. No test changed, so the gate's oracle line goes from 37/53 to 37/109.
  - Where a family refuses the prescribed-density route (the particle FP, #2519; meshless Galerkin, #2532), the
    refusal is that cell's content.
