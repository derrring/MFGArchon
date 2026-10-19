- **`PotentialCost(V)` hands V a copy of every point it reads V at** (#1642 stage 1b-2). A `V` that wrote into its
  argument changed what the library read. At binding, the library's first call to it ran on a view of the first node,
  so that node's value was read shifted; the nodes themselves did not move. Off the nodes it moved the caller's own
  points, such as HJB-GFDM's collocation points, and with them the other parts of a sum.
