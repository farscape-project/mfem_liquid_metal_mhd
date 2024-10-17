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

int main(int argc, char *argv[])
{

   // Initialize MPI and HYPRE.
   Mpi::Init(argc, argv);
   int num_procs = Mpi::WorldSize();
   int myid = Mpi::WorldRank();
   Hypre::Init();

   // Define constants.
   ConstantCoefficient zero(0.0);
   ConstantCoefficient one(1.0);

   // Note: alpha1 must be defined negative.
   real_t alpha1(-1.0);


   // Parse command line options.
   string mesh_file = "../mesh/square.msh";
   int order_pressure = 1;
   int order_velocity;
   order_velocity = order_pressure + 1;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
   args.AddOption(&order_pressure, "-o", "--order", "Finite element polynomial degree");
   args.ParseCheck();


   // Read the mesh from the given mesh file, and refine once uniformly.
   Mesh serial_mesh(mesh_file);
   ParMesh mesh(MPI_COMM_WORLD, serial_mesh);
   serial_mesh.Clear(); 
   mesh.UniformRefinement();



   // ----------------------------------------------------------------------------
   // Finite Element Spaces.
   // ----------------------------------------------------------------------------
   // Define finite element spaces on mesh. The order for pressure must be 1 less 
   // than that of velocity.
   // ----------------------------------------------------------------------------
   // H1 continuous Lagrange finite elements of given order for pressure.
   H1_FECollection pressure_fec(order_pressure, mesh.Dimension());
   ParFiniteElementSpace pressure_fespace(&mesh, &pressure_fec);
   cout << "Number of unknowns: " << pressure_fespace.GetTrueVSize() << endl;

   // H1 continuous Lagrange finite elements of given order (order_pressure + 1)
   // for velocity.
   H1_FECollection velocity_fec(order_velocity, mesh.Dimension());
   ParFiniteElementSpace velocity_fespace(&mesh, &velocity_fec);
   cout << "Number of unknowns: " << velocity_fespace.GetTrueVSize() << endl;
   // ----------------------------------------------------------------------------
   // ----------------------------------------------------------------------------



   // Extract the list of all the boundary DOFs. These will be marked as
   // Dirichlet in order to enforce zero boundary conditions.
   Array<int> boundary_dofs;
   pressure_fespace.GetBoundaryTrueDofs(boundary_dofs);



   // ----------------------------------------------------------------------------
   // Initialise solutions.
   // ----------------------------------------------------------------------------
   // Define solutions as ParGridFunctions in relevant fespace and set initial 
   // guesses to zero.
   // ----------------------------------------------------------------------------
   // Define the solutions xi, nu, p as grid functions for pressure. 
   ParGridFunction xi(&pressure_fespace);
   ParGridFunction nu(&pressure_fespace);
   ParGridFunction p(&pressure_fespace);

   // Define the solution for velocity.
   ParGridFunction u(&velocity_fespace);

   // Set initial guesses to zero.  This also sets BCs.
   xi = 0.0;
   nu = 0.0;
   p = 0.0;
   u = 0.0;
   // ----------------------------------------------------------------------------
   // ----------------------------------------------------------------------------

   

   // ----------------------------------------------------------------------------
   // Linear forms for RHS.
   // ----------------------------------------------------------------------------
   // Set up the linear form b(.) corresponding to the right-hand side of 
   // each solve.
   // ----------------------------------------------------------------------------
   // Set up rhs for pressure solve.
   ParLinearForm bp(&pressure_fespace);
   bp.AddDomainIntegrator(new DomainLFIntegrator(one));
   bp.Assemble();

   // Set up rhs for velocity solve.
   ParLinearForm bu(&pressure_fespace);
   bu.AddDomainIntegrator(new DomainLFIntegrator(one)); //FIXME
   bu.Assemble();
   // ----------------------------------------------------------------------------
   // ----------------------------------------------------------------------------

   

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

   // Bilinear forms for the velocity solve.
   ParBilinearForm fk(&velocity_fespace);
   fk.AddDomainIntegrator(new MassIntegrator(one));  //FIXME
   fk.Assemble();

   ParBilinearForm b(&velocity_fespace);
   b.AddDomainIntegrator(new MassIntegrator(one));  //FIXME
   b.Assemble();
   // ----------------------------------------------------------------------------
   // ----------------------------------------------------------------------------



   // ----------------------------------------------------------------------------
   // Form linear systems.
   // ----------------------------------------------------------------------------
   // ----------------------------------------------------------------------------
   // Form the linear systems for both 
   //       M_p xi = r_p,
   //       S_p nu = r_p. 
   HypreParMatrix Mp;
   Vector Bmp, Xi;
   mp.FormLinearSystem(boundary_dofs, xi, bp, Mp, Xi, Bmp);

   HypreParMatrix Sp;
   Vector Bsp, Nu;
   sp.FormLinearSystem(boundary_dofs, nu, bp, Sp, Nu, Bsp);

   // How to set up linear system for velocity...

   // ----------------------------------------------------------------------------
   // ----------------------------------------------------------------------------



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
   M_solver.SetMaxIter(10);
   M_solver.SetPrintLevel(0);
   M_prec.SetType(HypreSmoother::Jacobi); // Works for now, but check if diagonal...
   M_solver.SetPreconditioner(M_prec);
   M_solver.SetOperator(Mp);

   M_solver.Mult(Bmp,Xi);
   mp.RecoverFEMSolution(Xi, bp, xi);


   // Solve the S_p nu = r_p system using PCG with Jacobi preconditioner.
   HypreSolver *amg = new HypreBoomerAMG(Sp);
   HyprePCG *pcg = new HyprePCG(Sp);
   pcg->SetTol(1e-12);
   pcg->SetMaxIter(200);
   pcg->SetPrintLevel(2);
   pcg->SetPreconditioner(*amg);
   pcg->Mult(Bsp, Nu);
   sp.RecoverFEMSolution(Nu, bp, nu);

   // Do the sum y_p = -alpha1 xi - nu.  Note: above alpha1 must be defined negative.
   p.Add(alpha1,xi);
   p.Add(-1.0,nu);
   
   // ----------------------------------------------------------------------------
   // ----------------------------------------------------------------------------



   // ----------------------------------------------------------------------------
   // Exporting.
   // ----------------------------------------------------------------------------
   // Save data in the ParaView format
   // ----------------------------------------------------------------------------
   // Export pressure data.
   ParaViewDataCollection paraview_dc("navier_stokes", &mesh);
   paraview_dc.SetPrefixPath("data");
   paraview_dc.SetLevelsOfDetail(order_pressure);
   paraview_dc.SetCycle(0);
   paraview_dc.SetDataFormat(VTKFormat::BINARY);
   paraview_dc.SetHighOrderOutput(true);
   paraview_dc.SetTime(0.0); // set the time
   paraview_dc.RegisterField("xi",&xi);
   paraview_dc.RegisterField("nu",&nu);
   paraview_dc.RegisterField("p",&p);
   paraview_dc.Save();

   // ----------------------------------------------------------------------------
   // ----------------------------------------------------------------------------

   return 0;
}
