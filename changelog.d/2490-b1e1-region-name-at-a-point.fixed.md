- **A `region_name` segment is read at a point as it is read on a face** (#2490 stage B1e-1, #2512 row B1).
  - `conditions.region_name_holds_point` reads it in the face rule's order: the mask of a geometry that marks the
    region, then a face label, then a refusal with its remedies. `BCSegment.matches_point` asks it.
  - A label-named segment, `region_name="x_max"` with no geometry, governed its face for the face readers and was
    refused at a point of that face. It now holds that face's points for both.
  - The point rule read a point it could not map as outside the region, in two `except` clauses that returned False,
    and read a point off the marking grid at the node it clamped the point to. Such a point is now refused by name. A reader that resolves its points at
    construction, HJB-GFDM's classification, refuses there.
  - A point between the marking grid's nodes is read at the nearest node. Main read it at the node below it on each
    axis, truncating the index: on nodes y = 0.25 and 0.5, with the strip marked at y > 0.4, a point at y = 0.45
    took 0.25's value and lay outside the strip. It now takes 0.5's.
