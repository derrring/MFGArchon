The per-face ghost path silently dropped a non-zero Neumann flux and handed unclaimed walls the
first segment instead of `default_bc`.

`PreallocatedGhostBuffer` dispatches on `bc.is_uniform`, so stating the same boundary condition as
one unrestricted segment or as one segment per face selected different code. `_apply_ghost_for_face`
mirrored the interior without adding `dx * value`, which `_apply_linear_reflection` has done since
#1262 — a per-face `du/dn = 2` was applied as `du/dn = 0`. Its fallback for a face no segment claimed
was `bc.segments[0]`, and segments sort priority-descending, so the highest-priority segment (usually
the exit) governed every wall it had not claimed; `default_bc` was never consulted. Resolution now
goes through `_resolve_default_bc`, which raises when no default is set (#1100) rather than supplying
one by accident of sort order.
