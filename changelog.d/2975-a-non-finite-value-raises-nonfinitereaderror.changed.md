- **A non-finite value at the solve's existing checks now raises `NonFiniteReadError`** (#2975), a `ValueError`, with
  the owner's message. These already refused it, under another type or message:
  - a boundary value that is NaN or infinite, read by a boundary segment, a `NodeBC`, the FDM ghost layer, the graph
    applicator (`apply_direct`, `create_graph_applicator`, `set_dirichlet_nodes`, `set_source_nodes`) or a Robin
    coefficient, raised `NotImplementedError` and now raises `NonFiniteReadError`. Code that caught
    `NotImplementedError` for it no longer does. A value of a type the reader cannot read, and an integer no float
    holds, keep `NotImplementedError`;
  - a callable volatility's or drift's NaN output, a volatility tensor, a `volatility=` override, the iterator's
    initial mass and the particle FP's `m_initial` raised `ValueError` with their own messages.

  Where the old message printed the value, NaN or infinity, the new one gives the count of non-finite entries.
