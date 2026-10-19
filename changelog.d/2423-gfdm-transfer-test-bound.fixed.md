- **`test_gfdm_grid_collocation_transfer_2756.py`'s 2-D row holds on Linux** (#2423). It compares U from two HJB-GFDM
  solves whose collocation points are the grid's nodes in two orders. Its bound, 1e-10, was below the round-off between
  them on the Ubuntu runner, 2.05e-9 (7.6e-12 on macOS), so the weekly discrimination sweep failed in its baseline step
  on every shard before any mutation ran. The bound is now 1e-6. Reading the density by index, #2756's defect, still
  puts U 1.33 off.
