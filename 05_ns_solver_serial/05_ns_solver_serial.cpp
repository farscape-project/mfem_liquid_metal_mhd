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
   cout << endl;
}

void u_exact(const mfem::Vector & x, mfem::Vector & f)
{
  double u_max(1.0);
  double y_max(1.0);
  double z_max(1.0);
  double y(x(1));
  double z(x(2));
  f(0) = (9.0 / 4.0) * u_max * (1 - (y * y) / (y_max * y_max)) * (1 - (z * z) / (z_max * z_max));
  f(1) = 0.0;
  f(2) = 0.0;
}

int main(int argc, char *argv[])
{

   // Define constants.
   ConstantCoefficient zero(0.0);
   ConstantCoefficient one(1.0);
   ConstantCoefficient reciprocal_Re(1 / 100.0);


   // Note: alpha1 must be defined negative.
   real_t alpha1(-1.0);  // alpha1 = alpha + 1/Re.  alpha = 1 (default).
   // kappa = ...

   // Set fe_space orders.
   int order_pressure = 1;
   int order_velocity;
   order_velocity = order_pressure + 1;

   //Mesh mesh = Mesh::MakeCartesian2D(10, 10, mfem::Element::Type::QUADRILATERAL, true, 1.0, 1.0);
   Mesh mesh = Mesh::MakeCartesian3D(10, 10, 10, mfem::Element::Type::QUADRILATERAL, true, 1.0, 1.0, 1.0);
   int dim = mesh.Dimension();

   // Define vector coefficient based on mesh dimension.
   Vector one_vector(3);
   one_vector = 0.0;
   VectorConstantCoefficient one_vector_coef(one_vector);

   Vector zero_vector(3);
   zero_vector = 1.0;
   VectorConstantCoefficient zero_vector_coef(zero_vector);

   // ----------------------------------------------------------------------------
   // Finite Element Spaces.
   // ----------------------------------------------------------------------------
   // Define finite element spaces on mesh. The order for pressure must be 1 less 
   // than that of velocity.
   // ----------------------------------------------------------------------------
   // H1 continuous Lagrange finite elements of given order for pressure.
   H1_FECollection pressure_fec(order_pressure, mesh.Dimension());
   FiniteElementSpace pressure_fespace(&mesh, &pressure_fec);

   // H1 continuous Lagrange finite elements of given order (order_pressure + 1)
   // for velocity.
   H1_FECollection velocity_fec(order_velocity, mesh.Dimension());
   FiniteElementSpace velocity_fespace(&mesh, &velocity_fec, 3);

   checkpoint(1);
   cout << "Mesh dimension: " << mesh.Dimension() << endl;
   cout << "Pressure DoFs: " << pressure_fespace.GetTrueVSize() << endl;
   cout << "Velocity DoFs: " << velocity_fespace.GetTrueVSize() << endl;
   cout << endl;

   // ----------------------------------------------------------------------------
   // ----------------------------------------------------------------------------


   mfem::GridFunction ustar_n(&velocity_fespace); // 0.5(3u_{n-1} - u_{n-2})
   //ustar_n = 1.0;
   mfem::VectorGridFunctionCoefficient ustar_coef(&ustar_n);
   mfem::VectorFunctionCoefficient ucoef(3, u_exact);
   ustar_n.ProjectCoefficient(ucoef);

   //mfem::VectorFunctionCoefficient ucoef(3, u_exact);

   //ustar_n.ProjectCoefficient(ucoef);

   // ----------------------------------------------------------------------------
   // Define boundaries.
   // ----------------------------------------------------------------------------
   Array<int> boundary_marker_pressure, boundary_marker_velocity; 
   boundary_marker_pressure.SetSize(pressure_fespace.GetMesh()->bdr_attributes.Max());
   boundary_marker_pressure = 0; 
   boundary_marker_pressure[0] = 0; 
   boundary_marker_pressure[1] = 0; 
   boundary_marker_pressure[2] = 0; 
   boundary_marker_pressure[3] = 0; 

   boundary_marker_velocity.SetSize(velocity_fespace.GetMesh()->bdr_attributes.Max());
   boundary_marker_velocity = 0; 
   boundary_marker_velocity[0] = 0; 
   boundary_marker_velocity[1] = 0; 
   boundary_marker_velocity[2] = 0; 
   boundary_marker_velocity[3] = 0; 

   Array<int> pressure_ess_tdof, velocity_ess_tdof;
   pressure_fespace.GetEssentialTrueDofs(boundary_marker_pressure, pressure_ess_tdof);
   velocity_fespace.GetEssentialTrueDofs(boundary_marker_velocity, velocity_ess_tdof);

   checkpoint(2);
   const Array<int> &bdr_attr = velocity_fespace.GetMesh()->bdr_attributes;
   cout << "Boundary attributes: ";
   for (int i = 0; i < bdr_attr.Size(); i++) {
      cout << bdr_attr[i] << " ";
   }
   cout << endl; 
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
   GridFunction xi(&pressure_fespace);
   GridFunction nu(&pressure_fespace);
   GridFunction yp(&pressure_fespace);

   // Define the solution for velocity.
   GridFunction yu(&velocity_fespace);
   // Solution for B^T * y_p.
   GridFunction bTyp(&velocity_fespace);

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
   LinearForm rp(&pressure_fespace);
   rp.AddDomainIntegrator(new DomainLFIntegrator(zero));
   rp.Assemble();

   checkpoint(3);
   cout << "Pressure vector size (rp): " << rp.Size() << endl;

   // Set up rhs for velocity solve.
   LinearForm ru(&velocity_fespace);
   ru.AddDomainIntegrator(new VectorDomainLFIntegrator(zero_vector_coef));
   ru.Assemble();

   cout << "Velocity vector size (ru): " << ru.Size() << endl;
   cout << endl;

   cout << "one_vector_coef size: " << one_vector_coef.GetVDim() << endl;
   cout << "Mesh dimension: " << mesh.Dimension() << endl;
   cout << endl;

   MixedBilinearForm b(&velocity_fespace,&pressure_fespace);
   b.AddDomainIntegrator(new VectorDivergenceIntegrator(one));
   b.Assemble();
   b.Finalize();

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
   BilinearForm mp(&pressure_fespace);
   mp.AddDomainIntegrator(new MassIntegrator(one));
   mp.Assemble();

   BilinearForm sp(&pressure_fespace);
   sp.AddDomainIntegrator(new DiffusionIntegrator(one));
   sp.Assemble();

   checkpoint(5);

   // Bilinear forms for the velocity solve.
   //ParNonlinearForm fk(&velocity_fespace);
   //*********** ParBilinearForm...
   BilinearForm fk(&velocity_fespace);

   // Integrator for (v, v').
   fk.AddDomainIntegrator(new mfem::VectorMassIntegrator());
   // Integrator for A_AL(v, v').
   fk.AddDomainIntegrator(new mfem::VectorDiffusionIntegrator(reciprocal_Re)); // 1/Re (one may cause it to fail due to cancelling).
   
   // Old Integrator for O(u_n, v, v').
   //fk.AddDomainIntegrator(new SkewSymmetricVectorConvectionNLFIntegrator(one));

   // Integrator for O(u_n, v, v').
   //***********  They take a bilinear form in paper...
   fk.AddDomainIntegrator(new mfem::ConvectionIntegrator(ustar_coef, 0.5));
   fk.AddDomainIntegrator(new mfem::ConservativeConvectionIntegrator(ustar_coef, 0.5));

   checkpoint(6);
   fk.Assemble();
   
   

   // ----------------------------------------------------------------------------
   // ----------------------------------------------------------------------------


   checkpoint(7);


   // ----------------------------------------------------------------------------
   // Form linear systems.
   // ----------------------------------------------------------------------------
   // ----------------------------------------------------------------------------
   // Form the linear systems for both 
   //       M_p xi = r_p, and
   //       S_p nu = r_p. 
   SparseMatrix Mp;
   Vector Rp_m, Xi;
   mp.FormLinearSystem(boundary_marker_pressure, xi, rp, Mp, Xi, Rp_m);

   checkpoint(7);

   SparseMatrix Sp;
   Vector Rp_s, Nu;
   sp.FormLinearSystem(boundary_marker_pressure, nu, rp, Sp, Nu, Rp_s);

   checkpoint(8);

   // Set up nonlinear system for F_k y_u = r_u.
   //Vector Yu(velocity_fespace.GetTrueVSize()), Ru(velocity_fespace.GetTrueVSize());
   //yu.GetTrueDofs(Yu);
   //ru.ParallelAssemble(Ru);
   //fk.SetEssentialBC(ess_bdr, &ru);
   
   // Set up linear calculation for B^T y_p.
   SparseMatrix B;
   //OperatorPtr opB;
   Vector Brhs(pressure_fespace.GetTrueVSize()), BTyp(velocity_fespace.GetTrueVSize());
   //b.FormLinearSystem(boundary_dofs, yp, b, B, BTyp, Brhs);

   checkpoint(9);

   b.FormRectangularSystemMatrix(pressure_ess_tdof, velocity_ess_tdof, B);
   //************ Alex doesn't do this

   //TransposeOperator *B = NULL;
   //B = new TransposeOperator(opB.Ptr());

   checkpoint(10);

   //if (B.NumRows() == 0 || B.NumCols() == 0) {
   //   std::cerr << "Error: B was not properly initialized!" << std::endl;
   //   return -1;
   //}

   //cout << "B Rows: " << B.NumRows() << ", B Cols: " << B.NumCols() << endl;

   //cout << "Brhs Size: " << Brhs.Size() << endl;
   //cout << "BTyp Size: " << BTyp.Size() << endl;


   B.MultTranspose(Brhs,BTyp);

   checkpoint(11);

   ru.Add(1.0,BTyp);

   checkpoint(12);

   //fk.SetEssentialBC(velocity_ess_tdof, &ru);

   // ----------------------------------------------------------------------------
   // ----------------------------------------------------------------------------

   checkpoint(13);
   cout << "Bilinear form mp matrix size: " << Mp.Height() << " x " << Mp.Width() << endl;
   cout << "Bilinear form sp matrix size: " << Sp.Height() << " x " << Sp.Width() << endl;

   cout << "Pressure solution vector size: " << Xi.Size() << endl;
   cout << "Pressure rhs vector size: " << Rp_m.Size() << endl;


   // ----------------------------------------------------------------------------
   // Solving.
   // ----------------------------------------------------------------------------
   // ----------------------------------------------------------------------------
   // Solve the M_p xi = r_p system using PCG with Jacobi preconditioner.
   CGSolver M_solver;
   DSmoother M_prec;

   M_solver.iterative_mode = false;
   M_solver.SetRelTol(1e-8);
   M_solver.SetAbsTol(0.0);
   M_solver.SetMaxIter(200);
   M_solver.SetPrintLevel(0);
   //M_prec.SetType(DSmoother::Jacobi); // Works for now, but check if diagonal...
   M_solver.SetPreconditioner(M_prec);
   M_solver.SetOperator(Mp);

   M_solver.Mult(Rp_m,Xi);
   mp.RecoverFEMSolution(Xi, rp, xi);


   // Solve the S_p nu = r_p system using PCG with Jacobi preconditioner.
   CGSolver S_solver;
   DSmoother S_prec;

   S_solver.iterative_mode = false;
   S_solver.SetRelTol(1e-8);
   S_solver.SetAbsTol(0.0);
   S_solver.SetMaxIter(200);
   S_solver.SetPrintLevel(0);
   //S_prec.SetType(DSmoother::Jacobi); // Works for now, but check if diagonal...
   S_solver.SetPreconditioner(S_prec);
   S_solver.SetOperator(Sp);

   S_solver.Mult(Rp_s,Nu);
   mp.RecoverFEMSolution(Nu, rp, nu);

   // Do the sum y_p = -alpha1 xi - nu.  Note: above alpha1 must be defined negative.
   yp.Add(alpha1,xi);
   yp.Add(-1.0,nu);
   

   // Do multiplication for B^T y_p.
   //const SparseMatrix* rm = velocity_fespace.GetRestrictionMatrix();
   //rm->MultTranspose(yp, ru);

   // Set up nonlinear system for F_k y_u = r_u (where r_u is B^T yp).
   Vector Yu(velocity_fespace.GetTrueVSize()), Ru(velocity_fespace.GetTrueVSize());
   // Initialise Ru, Yu
   Ru = 0.0;
   //ru.GetTrueDofs(Ru);
   yu.GetTrueDofs(Yu);
   //ru.ParallelAssemble(Ru);
   //yu.ParallelProject(Yu);    // https://github.com/mfem/mfem/issues/2791 comment Feb 17, 2022
   //fk.SetEssentialBC(velocity_ess_tdof, &ru);

   cout << "f_k operator size (height x width): " << fk.Height() << " x " << fk.Width() << endl;
   cout << "Ru size: " << Ru.Size() << endl;
   cout << "Yu size: " << Yu.Size() << endl;

   // Set up the solve for the nonlinear F_k y_u = r_u system.
   CGSolver F_solver;
   OperatorJacobiSmoother F_prec;

   F_solver.SetOperator(fk);
   F_solver.SetPrintLevel(1);
   F_solver.SetRelTol(1e-10);
   F_solver.SetMaxIter(200);
   //F_prec.SetType(SparseSmoother::Jacobi); // Schwarz preconditioner...?  ASM (PETSc)?
   F_solver.SetPreconditioner(F_prec);
   F_solver.SetPrintLevel(3);

   // Solve nonlinear system.
   
   cout << "Starting Mult..." << endl;
   F_solver.Mult(Ru, Yu);
   cout << "Mult completed." << endl;

   //yu.Distribute(Yu);

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
