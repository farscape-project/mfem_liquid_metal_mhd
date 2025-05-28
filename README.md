# Liquid Metal MHD in MFEM

Repository for an inductionless liquid metal MHD solver in MFEM, detailed in [1] - deliverable for Fusion Computing Lab WP3.3.

The solver in `fluids_solver` contains `fluids_solver.cpp` which houses the `main()` function for the (work-in-progress) fluids solver.  The solver is currently a first step towards a GMRES non-linear solver.  Currently not implemented, but included, is the linearised equation in a preconditioner class that is ready to be incorporated into the non-linear solver.  Next steps are to continue to work on the GMRES solver and incorporate the preconditioner.


References:

[1] Li, Lingxiao, Mingjiu Ni, and Weiying Zheng. "A charge-conservative finite element method for inductionless MHD equations. Part II: a robust solver." SIAM Journal on Scientific Computing 41.4 (2019): B816-B842.
