- **The old argument order for `u_terminal` / `m_initial` is refused (#2431).** A callable whose parameter names put time after space, `lambda x, t` or `lambda x, t=0.0`, raises, naming the order `(t, x)`. The first used to be read as `f(x, t)`; the second was read at its default `t`. Also refused:
  - in 1-D, a two-argument callable whose names do not say which argument is time (`lambda a, b`);
  - an `np.vectorize` object, whose signature hides its names.

  In 2-D and 3-D, a callable that binds no time slot is read as the deprecated expanded coordinates `f(x, y)` / `f(x, y, z)`, with their warning. That includes an `np.vectorize` object. Before, such a callable was first probed as `f(x, t)` and `f(t, x)`. `validate_components` and `validate_u_terminal` take a `terminal_time` keyword.
