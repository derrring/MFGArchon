`ghost_cell_neumann`'s docstring stated `u_ghost = u_interior + 2*dx*g*sign`, the formula #1972
removed from its body, while an implementation comment eleven lines below already struck that text.
The body returns `interior_value + dx * flux_value`. Anyone reading `help()` saw the retired
formula presented as current, and the correction was visible only to a reader of the source. Fixed
here because #1936 makes it load-bearing: both retirement messages now send every caller to this
function.
