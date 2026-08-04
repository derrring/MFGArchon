- **The dead `if __name__ == "__main__":` block in `geometry/boundary/enforcement.py`**, which was
  the module's only test of any kind and **asserted the wrong periodic convention** (Issue #1820).
  Same shape as the block #1736 removed from `bc_utils.py`: it cannot run under either invocation,
  so nothing ever checked it, and it pinned `field[0] = field[-2]` as correct. Replaced by
  `tests/unit/test_geometry/test_enforcement.py`, which pins the convention rather than the
  formula -- a rewrite keeping the formula and changing the grid is equally wrong.
