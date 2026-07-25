- **A fast test that catches a wrong Newton step LENGTH** (Refs #1660). Automatic Newton coverage
  already existed in `test_newton_picard_agreement.py`, which runs in every automatic tier and is
  strictly stronger — 5 Picard warmup iterations plus 3 Newton iterations with line search. Measured
  over 12 mutations, the one defect class it passes and this new test fails is a halved Newton step;
  it catches sign flips, a transposed FD Jacobian, a wrong FD epsilon and a discarded step already.
  0.25 s, in the fast tier.
