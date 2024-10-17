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
   // Define constants.
   ConstantCoefficient zero(0.0);
   ConstantCoefficient one(1.0);

   real_t alpha1(-1.0);

   // 1. Parse command line options.
   string mesh_file = "../mesh/square.msh";
   int order_pressure = 1;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
   args.AddOption(&order_pressure, "-o", "--order", "Finite element polynomial degree");
   args.ParseCheck();

   // 2. Read the mesh from the given mesh file, and refine once uniformly.
   Mesh mesh(mesh_file);
   mesh.UniformRefinement();

   // Define finite element space on mesh. Specify H1 continuous Lagrange 
   // finite elements of given order.  The order for pressure must be 1 less 
   // than that of velocity.
   H1_FECollection fec(order_pressure, mesh.Dimension());
   FiniteElementSpace fespace(&mesh, &fec);
   cout << "Number of unknowns: " << fespace.GetTrueVSize() << endl;

   // 4. Extract the list of all the boundary DOFs. These will be marked as
   //    Dirichlet in order to enforce zero boundary conditions.
   Array<int> boundary_dofs;
   fespace.GetBoundaryTrueDofs(boundary_dofs);

   // 5. Define the solution x as a finite element grid function in fespace. Set
   //    the initial guess to zero, which also sets the boundary conditions.
   GridFunction xi(&fespace);
   GridFunction nu(&fespace);
   GridFunction p(&fespace);

   xi = 0.0;
   nu = 0.0;
   p = 0.0;

   // 6. Set up the linear form b(.) corresponding to the right-hand side.
   LinearForm b(&fespace);
   b.AddDomainIntegrator(new DomainLFIntegrator(one));
   b.Assemble();

   // Set up the bilinear form corresponding to the operators for pressure.
   // ******************************************************
   // ******************************************************
   // This needs separating out into xi and nu solves.
   // ******************************************************
   // ******************************************************
   BilinearForm mp(&fespace);
   mp.AddDomainIntegrator(new MassIntegrator(one));
   mp.Assemble();

   BilinearForm sp(&fespace);
   sp.AddDomainIntegrator(new DiffusionIntegrator(one));
   sp.Assemble();

   // 8. Form the linear system A X = B. This includes eliminating boundary
   //    conditions, applying AMR constraints, and other transformations.
   SparseMatrix Mp;
   Vector Bmp, Xi;
   mp.FormLinearSystem(boundary_dofs, xi, b, Mp, Xi, Bmp);

   HypreParMatrix Sp;
   Vector Bsp, Nu;
   sp.FormLinearSystem(boundary_dofs, nu, b, Sp, Nu, Bsp);

   // Solve the system using PCG with symmetric Gauss-Seidel preconditioner.
   //GSSmoother MatMp(Mp);
   //PCG(Mp, MatMp, Bmp, Xi, 1, 10, 1e-12, 0.0);

   CGSolver M_solver;
   DSmoother M_prec; // Diagonal preconditioner.  Jacobi from PetsC would be 
                     // go to, but this works for now.

   M_solver.iterative_mode = false;
   M_solver.SetRelTol(1e-8);
   M_solver.SetAbsTol(0.0);
   M_solver.SetMaxIter(10);
   M_solver.SetPrintLevel(0);
   M_solver.SetPreconditioner(M_prec);
   M_solver.SetOperator(Mp);

   M_solver.Mult(b,xi);

   // How to solve this with AMG Solver?
   //GSSmoother MatSp(Sp);
   //PCG(Sp, MatSp, Bsp, Nu, 1, 200, 1e-12, 0.0);

   CGSolver S_solver;
   HypreBoomerAMG S_prec;

   HypreParMatrix *hSp = Sp.As<HypreParMatrix>();

   S_solver.iterative_mode = false;
   S_solver.SetRelTol(1e-8);
   S_solver.SetAbsTol(0.0);
   S_solver.SetMaxIter(10);
   S_solver.SetPrintLevel(0);
   S_solver.SetPreconditioner(S_prec);
   //S_solver.SetPreconditioner(*amg);
   S_solver.SetOperator(hSp);

   S_solver.Mult(b,nu);


   p.Add(alpha1,xi);
   p.Add(-1.0,nu);
   

   // 10. Save data in the ParaView format
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

   return 0;
}
