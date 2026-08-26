`examples/basic/three_mode_api_demo.py` selected the non-conservative `gradient_upwind` advection
scheme in its expert-mode demonstration and lost **98.17%** of its probability mass (#2008). It also
did not run to completion before or after that line: `compare_schemes()` selects
`FDM_CENTERED`, whose FP half goes negative at timestep 3, and the mass-fabrication gate stops the
solve rather than clipping it. The script exited 1 on `main` and exits 0 here.

The four advection schemes fail four different ways on this problem, which is why the fix names a
scheme rather than a family: `gradient_centered` and `divergence_centered` abort on a negative
density inside the first Picard iteration; `gradient_upwind` converges and loses the mass;
`divergence_upwind` converges with a mass-conservation error of `5.1e-15` and is what the demo now
uses.
