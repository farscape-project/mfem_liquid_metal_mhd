# Liquid Metal MHD in MFEM

Repository for a liquid metal MHD code in MFEM - deliverable for Fusion Computing Lab WP3.3.

## Electrostatics

The solver in `01_fluids_solver` contains a linearised form of the Navier-Stokes equations, implemented in `01_fluids_solver.cpp` [1].  Currently the solver just solves the linearised equation in a preconditioner class that is ready to be incorporated into a non-linear solver.  Next step is to implement the GMRES solve and incorporate the preconditioner.


References:

[1] Li, Lingxiao, Mingjiu Ni, and Weiying Zheng. "A charge-conservative finite element method for inductionless MHD equations. Part II: a robust solver." SIAM Journal on Scientific Computing 41.4 (2019): B816-B842.
