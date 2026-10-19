- **`test_gfdm_grid_collocation_transfer_2756.py`'s permutation rows hold on Linux** (#2423). Each compares U from
  two HJB-GFDM solves whose collocation points are the grid's nodes in two orders. HJB-GFDM's stencil search breaks an
  exact distance tie between neighbours by point order, and on a tensor grid such ties are common. So two orders can
  give a centre different stencils, and the platform decides which way some ties fall. The 2-D row's two solves then
  differed by 7.6e-12 on macOS and by 2.05e-9 on the Ubuntu runner. Its bound of 1e-10 failed the weekly
  discrimination sweep in its baseline step, on every shard, before any mutation ran. The 1-D rows had the same bound.
  Every row's bound is now 1e-6. Each row states its measured gap and the gap that #2756's defect, reading the density
  by index, gives on it: 0.480, 0.602 and 1.331.
