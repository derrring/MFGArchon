- **`test_gfdm_grid_collocation_transfer_2756.py`'s permutation rows hold on Linux** (#2423). Each compares U from
  two HJB-GFDM solves whose collocation points are the grid's nodes in two orders. They differ in round-off only, by an
  amount that depends on the order and the BLAS: the 2-D row gives 7.6e-12 on macOS and 2.05e-9 on the Ubuntu runner,
  so its bound of 1e-10 failed the weekly discrimination sweep in its baseline step, on every shard, before any
  mutation ran. The 1-D rows had the same bound and passed there by margin alone. Every row's bound is now 1e-6, and
  each states its own round-off and the gap that #2756's defect, reading the density by index, gives on it: 0.480,
  0.602 and 1.331.
