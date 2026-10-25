- **A marked region on part of a face is read as a `region=` placement on it** (#2795). A `region_name` segment's face
  cover was read through a resolver that refused a partial cover of any face of the region, whichever face was asked.
  So:
  - every box corner was refused;
  - HJB-GFDM refused a cloud holding one;
  - the particle FP's array drift refused the BC, so a coupled HJB-GFDM + particle solve could not use an exit window
    marked with `mark_region`.

  The cover is now read on the region's grid mask, as a `region=` or `sdf_region` placement's is, so a partial marked
  window answers as `boundary=` + `region=` placing the same nodes does wherever a face's cover is read: the face
  readers, the corners, and `face_segments_at_point` there. A point read elsewhere on a face reads a marked region at
  its nearest grid node, half a grid spacing past the marked range, as it did before.
  - HJB-GFDM and the array drift read such a window, and the coupled solve runs. The array drift absorbs within the
    marked nodes' cells, as the callable drift did.
  - A face-level reader (the FDM family) still refuses a partial marked window whose condition differs from the rest of
    its face, now with the `region=` refusal, naming `region_name`. A window with the same condition is accepted, as
    a partial `region=` is.
  - A marked region covering no face's interior is decided by the BC's whole-BC check, as a `region=` covering none
    is: refused where its condition would be dropped, accepted where it is effect-neutral.
  - `faces_covered_by_region` and `region_name_governs_face`, private resolvers with no other caller, are removed.
