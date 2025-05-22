# Liquid Metal MHD in MFEM

Repository for an inductionless liquid metal MHD solver in MFEM, detailed in [1] - deliverable for Fusion Computing Lab WP3.3.

The solver in `01_fluids_solver` contains a number of in progress solvers.  Currently `01_fluids_solver.cpp` just solves the linearised equation in a preconditioner class that is ready to be incorporated into a non-linear solver.  The file `02_gmres_fluids_solver.cpp` is a first step towards a GMRES non-linear solver and `03_gmres_solver_restructure.cpp` is a refactored version of `02_gmres_fluids_solver.cpp` into multiple files to simplify working.  Next step is to continue to work on the GMRES solve and incorporate the preconditioner.


References:

[1] Li, Lingxiao, Mingjiu Ni, and Weiying Zheng. "A charge-conservative finite element method for inductionless MHD equations. Part II: a robust solver." SIAM Journal on Scientific Computing 41.4 (2019): B816-B842.
