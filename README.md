# Liquid Metal MHD in MFEM

Repository for a liquid metal MHD code in MFEM - deliverable for Fusion Computing Lab WP3.3.

## Electrostatics

The solver in `01_fluids_solver` contains a linearised form of the Navier-Stokes equations, detailed in `01_fluids_solver.cpp`.  Currently the solver just solves the linearised equation in a preconditioner class that is ready to be incorporated into a non-linear solver.  Next step is to implement the GMRES solve and incorporate the preconditioner.
