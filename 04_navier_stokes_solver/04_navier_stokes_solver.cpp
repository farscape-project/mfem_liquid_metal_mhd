//
// Compile with: make 04_navier_stokes_solver
//
// Sample runs:  ./04_navier_stokes_solver
//
// Description: This example code demonstrates the most basic usage of MFEM to
//              define a simple finite element discretization of the Laplace
//              problem -Delta u = 1 with zero Dirichlet boundary conditions.
//              General 2D/3D mesh files and finite element polynomial degrees
//              can be specified by command line options.

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

void checkpoint(int num)
{
   cout << "**********************************************" << endl;
   cout << "**************** CHECKPOINT " << num << " ****************" << endl;
   cout << "**********************************************" << endl;
}

int main(int argc, char *argv[])
{

   // Initialize MPI and HYPRE.
   Mpi::Init(argc, argv);
   //int num_procs = Mpi::WorldSize();
   //int myid = Mpi::WorldRank();
   Hypre::Init();

   // Define constants.
   ConstantCoefficient zero(0.0);
   ConstantCoefficient one(1.0);


   // Note: alpha1 must be defined negative.
   real_t alpha1(-1.0);  // alpha1 = alpha + 1/Re.  alpha = 1 (default).
   // kappa = ...

   // Parse command line options.
   string mesh_file = "../mesh/square.msh";
   int order_pressure = 1;
   int order_velocity;
   order_velocity = order_pressure + 1;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
   args.ParseCheck();


   // Read the mesh from the given mesh file, and refine once uniformly.
   Mesh serial_mesh(mesh_file);
   ParMesh mesh(MPI_COMM_WORLD, serial_mesh);
   serial_mesh.Clear(); 
   mesh.UniformRefinement();

   // Define vector coefficient based on mesh dimension.
   Vector constantVector(mesh.Dimension());
   for (int i = 0; i < mesh.Dimension(); i++){constantVector(i) = 1.0;}
   VectorConstantCoefficient oneVector(constantVector);

   for (int i = 0; i < mesh.Dimension(); i++){constantVector(i) = 0.0;}
   VectorConstantCoefficient zeroVector(constantVector);

   // ----------------------------------------------------------------------------
   // Finite Element Spaces.
   // ----------------------------------------------------------------------------
   // Define finite element spaces on mesh. The order for pressure must be 1 less 
   // than that of velocity.
   // ----------------------------------------------------------------------------
   // H1 continuous Lagrange finite elements of given order for pressure.
   H1_FECollection pressure_fec(order_pressure, mesh.Dimension());
   ParFiniteElementSpace pressure_fespace(&mesh, &pressure_fec);

   // H1 continuous Lagrange finite elements of given order (order_pressure + 1)
   // for velocity.
   H1_FECollection velocity_fec(order_velocity, mesh.Dimension());
   ParFiniteElementSpace velocity_fespace(&mesh, &velocity_fec, mesh.Dimension());

   checkpoint(1);
   cout << "Pressure DoFs: " << pressure_fespace.GetTrueVSize() << endl;
   cout << "Velocity DoFs: " << velocity_fespace.GetTrueVSize() << endl;
   cout << endl;

   // ----------------------------------------------------------------------------
   // ----------------------------------------------------------------------------




   // ----------------------------------------------------------------------------
   // Define boundaries.
   // ----------------------------------------------------------------------------
   Array<int> boundary_marker_pressure, boundary_marker_velocity; 
   boundary_marker_pressure.SetSize(pressure_fespace.GetMesh()->bdr_attributes.Max());
   boundary_marker_pressure[0] = 0; 
   boundary_marker_pressure[1] = 0; 
   boundary_marker_pressure[2] = 1; 
   boundary_marker_pressure[3] = 0; 

   boundary_marker_velocity.SetSize(velocity_fespace.GetMesh()->bdr_attributes.Max());
   boundary_marker_velocity[0] = 1; 
   boundary_marker_velocity[1] = 0; 
   boundary_marker_velocity[2] = 0; 
   boundary_marker_velocity[3] = 0; 

   Array<int> pressure_ess_tdof, velocity_ess_tdof;
   pressure_fespace.GetEssentialTrueDofs(boundary_marker_pressure, pressure_ess_tdof);
   velocity_fespace.GetEssentialTrueDofs(boundary_marker_velocity, velocity_ess_tdof);

   checkpoint(2);
   cout << "No. of essential DoFs for pressure boundary conditions: " << pressure_ess_tdof.Size() << endl;
   cout << "No. of essential DoFs for velocity boundary conditions: " << velocity_ess_tdof.Size() << endl;
   cout << endl;

   // Extract the list of all the boundary DOFs. These will be marked as
   // Dirichlet in order to enforce zero boundary conditions.
   //Array<int> boundary_dofs;
   //pressure_fespace.GetBoundaryTrueDofs(boundary_dofs);
   //boundary_marker.SetSize(pressure_fespace.GetMesh()->bdr_attributes.Max());

   // Temporary location for velocity boundary conditions.
   //Array<int> ess_bdr(mesh.bdr_attributes.Max());
   //ess_bdr = 0;

   //velocity_fespace.GetEssentialTrueDofs(boundary_marker, test_ess_tdof);

   // ----------------------------------------------------------------------------
   // ----------------------------------------------------------------------------




   // ----------------------------------------------------------------------------
   // Initialise solutions.
   // ----------------------------------------------------------------------------
   // Define solutions as ParGridFunctions in relevant fespace and set initial 
   // guesses to zero.
   // ----------------------------------------------------------------------------
   // Define the solutions xi, nu, p as grid functions for pressure. 
   ParGridFunction xi(&pressure_fespace);
   ParGridFunction nu(&pressure_fespace);
   ParGridFunction yp(&pressure_fespace);

   // Define the solution for velocity.
   ParGridFunction yu(&velocity_fespace);
   // Solution for B^T * y_p.
   ParGridFunction bTyp(&velocity_fespace);

   // Set initial guesses to zero.  This also sets BCs.
   xi = 0.0;
   nu = 0.0;
   yp = 0.0;
   yu = 0.0;
   bTyp = 0.0;

   // ----------------------------------------------------------------------------
   // ----------------------------------------------------------------------------

   
   

   // ----------------------------------------------------------------------------
   // Linear forms for RHS.
   // ----------------------------------------------------------------------------
   // Set up the linear form b(.) corresponding to the right-hand side of 
   // each solve.
   // ----------------------------------------------------------------------------
   // Set up rhs for pressure solve.
   ParLinearForm rp(&pressure_fespace);
   rp.AddDomainIntegrator(new DomainLFIntegrator(zero));
   rp.Assemble();

   checkpoint(3);
   cout << "Pressure vector size (rp): " << rp.Size() << endl;

   // Set up rhs for velocity solve.
   ParLinearForm ru(&velocity_fespace);
   ru.AddDomainIntegrator(new VectorDomainLFIntegrator(zeroVector));
   ru.Assemble();

   cout << "Velocity vector size (ru): " << ru.Size() << endl;
   cout << endl;

   cout << "oneVector size: " << oneVector.GetVDim() << endl;
   cout << "Mesh dimension: " << mesh.Dimension() << endl;
   cout << endl;

   ParMixedBilinearForm b(&velocity_fespace,&pressure_fespace);
   b.AddDomainIntegrator(new VectorDivergenceIntegrator(one));
   b.Assemble();

   cout << "Pressure vector size (rp): " << rp.Size() << endl;
   cout << "Velocity vector size (ru): " << ru.Size() << endl;
   //cout << "Velocity vector size (b): " << b.Size() << endl;
   cout << endl;

   // ----------------------------------------------------------------------------
   // ----------------------------------------------------------------------------
   

   checkpoint(4);


   // ----------------------------------------------------------------------------
   // Bilinear/Nonlinear forms.
   // ----------------------------------------------------------------------------
   // Set up the bilinear and nonlinear forms corresponding to the operators for 
   // each solve.
   // ----------------------------------------------------------------------------
   // Bilinear forms for the two-part pressure solve.
   ParBilinearForm mp(&pressure_fespace);
   mp.AddDomainIntegrator(new MassIntegrator(one));
   mp.Assemble();

   ParBilinearForm sp(&pressure_fespace);
   sp.AddDomainIntegrator(new DiffusionIntegrator(one));
   sp.Assemble();

   checkpoint(5);

   // Bilinear forms for the velocity solve.
   ParNonlinearForm fk(&velocity_fespace);
   // Integrator for (v, v').
   fk.AddDomainIntegrator(new MassIntegrator(one));
   // Integrator for A_AL(v, v').
   fk.AddDomainIntegrator(new DiffusionIntegrator(one));
   // Integrator for O(u_n, v, v').
   fk.AddDomainIntegrator(new SkewSymmetricVectorConvectionNLFIntegrator(one));

   // Set up B^T yp part of velocity system.
   //ParBilinearForm b(&velocity_fespace);
   //b.AddDomainIntegrator(new MixedScalarDivergenceIntegrator(one));
   //b.Assemble();
   

   // ----------------------------------------------------------------------------
   // ----------------------------------------------------------------------------


   checkpoint(6);


   // ----------------------------------------------------------------------------
   // Form linear systems.
   // ----------------------------------------------------------------------------
   // ----------------------------------------------------------------------------
   // Form the linear systems for both 
   //       M_p xi = r_p, and
   //       S_p nu = r_p. 
   HypreParMatrix Mp;
   Vector Rp_m, Xi;
   mp.FormLinearSystem(boundary_marker_pressure, xi, rp, Mp, Xi, Rp_m);

   HypreParMatrix Sp;
   Vector Rp_s, Nu;
   sp.FormLinearSystem(boundary_marker_pressure, nu, rp, Sp, Nu, Rp_s);

   // Set up nonlinear system for F_k y_u = r_u.
   //Vector Yu(velocity_fespace.GetTrueVSize()), Ru(velocity_fespace.GetTrueVSize());
   //yu.GetTrueDofs(Yu);
   //ru.ParallelAssemble(Ru);
   //fk.SetEssentialBC(ess_bdr, &ru);
   
   // Set up linear calculation for B^T y_p.
   HypreParMatrix B;
   Vector Brhs(pressure_fespace.GetTrueVSize()), BTyp(velocity_fespace.GetTrueVSize());
   //b.FormLinearSystem(boundary_dofs, yp, b, B, BTyp, Brhs);
   b.FormRectangularSystemMatrix(pressure_ess_tdof, velocity_ess_tdof, B);

   B.MultTranspose(Brhs,BTyp);

   ru.Add(1.0,BTyp);

   fk.SetEssentialBC(velocity_ess_tdof, &ru);

   // ----------------------------------------------------------------------------
   // ----------------------------------------------------------------------------

   checkpoint(7);
   cout << "Bilinear form mp matrix size: " << Mp.Height() << " x " << Mp.Width() << endl;
   cout << "Bilinear form sp matrix size: " << Sp.Height() << " x " << Sp.Width() << endl;

   cout << "Pressure solution vector size: " << Xi.Size() << endl;
   cout << "Pressure rhs vector size: " << Rp_m.Size() << endl;


   // ----------------------------------------------------------------------------
   // Solving.
   // ----------------------------------------------------------------------------
   // ----------------------------------------------------------------------------
   // Solve the M_p xi = r_p system using PCG with Jacobi preconditioner.
   CGSolver M_solver(MPI_COMM_WORLD);
   HypreSmoother M_prec;

   M_solver.iterative_mode = false;
   M_solver.SetRelTol(1e-8);
   M_solver.SetAbsTol(0.0);
   M_solver.SetMaxIter(200);
   M_solver.SetPrintLevel(0);
   M_prec.SetType(HypreSmoother::Jacobi); // Works for now, but check if diagonal...
   M_solver.SetPreconditioner(M_prec);
   M_solver.SetOperator(Mp);

   M_solver.Mult(Rp_m,Xi);
   mp.RecoverFEMSolution(Xi, rp, xi);


   // Solve the S_p nu = r_p system using PCG with Jacobi preconditioner.
   HypreSolver *amg = new HypreBoomerAMG(Sp);
   HyprePCG *pcg = new HyprePCG(Sp);

   pcg->SetTol(1e-12);
   pcg->SetMaxIter(200);
   pcg->SetPrintLevel(2);
   pcg->SetPreconditioner(*amg);
   pcg->Mult(Rp_s, Nu);
   sp.RecoverFEMSolution(Nu, rp, nu);

   // Do the sum y_p = -alpha1 xi - nu.  Note: above alpha1 must be defined negative.
   yp.Add(alpha1,xi);
   yp.Add(-1.0,nu);
   

   // Do multiplication for B^T y_p.
   //const SparseMatrix* rm = velocity_fespace.GetRestrictionMatrix();
   //rm->MultTranspose(yp, ru);

   // Set up nonlinear system for F_k y_u = r_u (where r_u is B^T yp).
   Vector Yu(velocity_fespace.GetTrueVSize()), Ru(velocity_fespace.GetTrueVSize());
   yu.GetTrueDofs(Yu);
   ru.ParallelAssemble(Ru);
   fk.SetEssentialBC(velocity_ess_tdof, &ru);

   // Set up the solve for the nonlinear F_k y_u = r_u system.
   CGSolver F_solver(MPI_COMM_WORLD);
   NewtonSolver newton(MPI_COMM_WORLD);
   HypreSmoother F_prec;

   newton.SetOperator(fk);
   newton.SetSolver(F_solver);
   newton.SetPrintLevel(1);
   newton.SetRelTol(1e-10);
   newton.SetMaxIter(200);
   F_prec.SetType(HypreSmoother::Jacobi); // Schwarz preconditioner...?  ASM (PETSc)?
   newton.SetPreconditioner(F_prec);

   // Solve nonlinear system.
   cout << "F_k operator size: " << fk.Height() << " x " << fk.Width() << endl;
   cout << "Ru size: " << Ru.Size() << endl;
   cout << "Yu size: " << Yu.Size() << endl;
   newton.Mult(Ru, Yu);
   cout << "Mult completed." << endl;

   yu.Distribute(Yu);

   // ----------------------------------------------------------------------------
   // ----------------------------------------------------------------------------



   // ----------------------------------------------------------------------------
   // Exporting.
   // ----------------------------------------------------------------------------
   // Save data in the ParaView format
   // ----------------------------------------------------------------------------
   ParaViewDataCollection paraview_dc("navier_stokes", &mesh);
   paraview_dc.SetPrefixPath("data");
   paraview_dc.SetLevelsOfDetail(order_pressure);
   paraview_dc.SetCycle(0);
   paraview_dc.SetDataFormat(VTKFormat::BINARY);
   paraview_dc.SetHighOrderOutput(true);
   paraview_dc.SetTime(0.0); // set the time

   // Export pressure data.
   paraview_dc.RegisterField("xi",&xi);
   paraview_dc.RegisterField("nu",&nu);
   paraview_dc.RegisterField("p",&yp);

   // Export velocity data.
   paraview_dc.RegisterField("u",&yu);

   paraview_dc.Save();


   // ----------------------------------------------------------------------------
   // ----------------------------------------------------------------------------

   return 0;
}
