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

void checkpoint(int num);
real_t velocity_dbc(const Vector & x);
real_t pressure_dbc(const Vector & x);
real_t zero_dbc(const Vector & x);
void velocity_dbc_vec_func(const Vector & x, Vector & f);
void u_exact(const Vector & x, Vector & f);
double pFun_ex(const Vector & x);
void fFun(const Vector & x, Vector & f);
double gFun(const Vector & x);
double f_natural(const Vector & x);

int main(int argc, char *argv[])
{

   // Define constants.
   real_t Re(100.0);
   real_t reciprocal_Re(1 / Re);
   real_t alpha(1.0); // alpha = 1 (default).
   real_t alpha1, neg_alpha1;  
   alpha1 = alpha + reciprocal_Re;
   neg_alpha1 = -alpha1;

   // Define ConstantCoefficients.
   ConstantCoefficient zero(0.0);
   ConstantCoefficient one(1.0);
   ConstantCoefficient neg_one(-1.0);
   ConstantCoefficient reciprocal_Re_coef(reciprocal_Re);

   // kappa = ...

   // Define VectorConstantCoefficients.
   Vector zero_vector(3), one_vector(3);
   zero_vector = 1.0;
   VectorConstantCoefficient zero_vector_coef(zero_vector);
   one_vector = 0.0;
   VectorConstantCoefficient one_vector_coef(one_vector);
   

   // Set fe_space orders.
   int order_pressure = 1;
   int order_velocity;
   order_velocity = order_pressure + 1;

   //Mesh mesh = Mesh::MakeCartesian2D(10, 4, mfem::Element::Type::QUADRILATERAL, true, 1.0, 0.2);
   Mesh mesh = Mesh::MakeCartesian2D(30, 10, mfem::Element::Type::QUADRILATERAL, true, 1.0, 0.2);
   //Mesh mesh = Mesh::MakeCartesian3D(10, 10, 10, mfem::Element::Type::QUADRILATERAL, true, 1.0, 1.0, 1.0);
   int dim = mesh.Dimension();   

   // ----------------------------------------------------------------------------
   // Finite Element Spaces.
   // ----------------------------------------------------------------------------
   // Define finite element spaces on mesh. The order for pressure must be 1 less 
   // than that of velocity.
   // ----------------------------------------------------------------------------
   // H1 continuous Lagrange finite elements of given order for pressure.
   H1_FECollection pressure_fec(order_pressure, dim);
   FiniteElementSpace pressure_fespace(&mesh, &pressure_fec);

   // H1 continuous Lagrange finite elements of given order (order_pressure + 1)
   // for velocity.
   H1_FECollection velocity_fec(order_velocity, dim);
   FiniteElementSpace velocity_fespace(&mesh, &velocity_fec, dim);

   checkpoint(1);
   cout << "Mesh dimension: " << dim << endl;
   cout << "Pressure DoFs: " << pressure_fespace.GetTrueVSize() << endl;
   cout << "Velocity DoFs: " << velocity_fespace.GetTrueVSize() << endl;
   cout << endl;

   // ----------------------------------------------------------------------------
   // ----------------------------------------------------------------------------

   GridFunction ustar_n(&velocity_fespace); // 0.5(3u_{n-1} - u_{n-2})
   VectorFunctionCoefficient ucoef(3, u_exact);
   ustar_n.ProjectCoefficient(ucoef);
   VectorGridFunctionCoefficient ustar_coef(&ustar_n);

   // ----------------------------------------------------------------------------
   // Define boundaries.
   // ----------------------------------------------------------------------------

   VectorFunctionCoefficient velocity_DBC(dim, velocity_dbc_vec_func);
   FunctionCoefficient pressure_DBC(pressure_dbc);
   FunctionCoefficient zero_DBC(zero_dbc);

   // Essential (Dirichlet) boundary conditions for pressure.
   Array<int> ess_boundary_marker_pressure, ess_boundary_marker_velocity; 
   ess_boundary_marker_pressure.SetSize(pressure_fespace.GetMesh()->bdr_attributes.Max());
   ess_boundary_marker_pressure = 0; 
   // Dirichlet pressure at outlet.
   ess_boundary_marker_pressure[1] = 1; 
   // Dirichlet pressure at inlet.
   ess_boundary_marker_pressure[3] = 1; 

   // Natural (Neumann) boundary conditions for pressure.
   Array<int> nat_boundary_marker_pressure, nat_boundary_marker_velocity; 
   nat_boundary_marker_pressure.SetSize(pressure_fespace.GetMesh()->bdr_attributes.Max());
   nat_boundary_marker_pressure = 1;
   // Turn off natural boundary condition for pressure at outlet.
   nat_boundary_marker_pressure[1] = 0;
   // and inlet.
   nat_boundary_marker_pressure[3] = 0;


   // Essential (Dirichlet) boundary conditions for velocity.
   ess_boundary_marker_velocity.SetSize(velocity_fespace.GetMesh()->bdr_attributes.Max());
   ess_boundary_marker_velocity = 0;
   // Dirichlet velocity at inlet (non-zero).
   ess_boundary_marker_velocity[3] = 1;
   // Dirichlet velocity at "top"/"bottom" (zero).
   ess_boundary_marker_velocity[0] = 1;
   ess_boundary_marker_velocity[2] = 1;

   // Natural (Neumann) boundary conditions for velocity.
   nat_boundary_marker_velocity.SetSize(velocity_fespace.GetMesh()->bdr_attributes.Max());
   nat_boundary_marker_velocity = 1; 
   // Turn off natural boundary condition for velocity at inlet/"top"/"bottom".
   nat_boundary_marker_velocity[0] = 0; 
   nat_boundary_marker_velocity[2] = 0; 
   nat_boundary_marker_velocity[3] = 0; 


   Array<int> pressure_ess_tdof, velocity_ess_tdof;
   pressure_fespace.GetEssentialTrueDofs(ess_boundary_marker_pressure, pressure_ess_tdof);
   velocity_fespace.GetEssentialTrueDofs(ess_boundary_marker_velocity, velocity_ess_tdof);




   // ----------------------------------------------------------------------------
   // Initialise solutions.
   // ----------------------------------------------------------------------------
   // Define solutions as ParGridFunctions in relevant fespace and set initial 
   // guesses to zero.
   // ----------------------------------------------------------------------------
   // Define the solutions xi, eta, p as grid functions for pressure. 
   GridFunction xi(&pressure_fespace);
   GridFunction eta(&pressure_fespace);
   GridFunction yp(&pressure_fespace);

   // Define the solution for velocity.
   GridFunction yu(&velocity_fespace);
   // Solution for B^T * y_p.
   GridFunction bTyp(&velocity_fespace);

   // Set initial guesses to zero.  This also sets BCs.
   xi = 0.0;
   eta = 0.0;
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

   FunctionCoefficient velocity_dbc_coeff(velocity_dbc);

   // Set up rhs for velocity solve.
   LinearForm ru(&velocity_fespace);
   ru.AddDomainIntegrator(new VectorDomainLFIntegrator(zero_vector_coef));

   // Natural boundary condition.
   ru.AddBoundaryIntegrator(new VectorBoundaryFluxLFIntegrator(velocity_dbc_coeff), nat_boundary_marker_velocity);

   ru.Assemble();

   cout << "Velocity vector size (ru): " << ru.Size() << endl;
   cout << endl;

   cout << "one_vector_coef size: " << one_vector_coef.GetVDim() << endl;
   cout << "Mesh dimension: " << mesh.Dimension() << endl;
   cout << endl;

   MixedBilinearForm b(&velocity_fespace,&pressure_fespace);
   b.AddDomainIntegrator(new VectorDivergenceIntegrator(neg_one));
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
   fk.AddDomainIntegrator(new mfem::VectorDiffusionIntegrator(reciprocal_Re_coef)); // 1/Re (one may cause it to fail due to cancelling).
   
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

   //for(int ti = 0; ti < 10; ti++)
   //{

   // ----------------------------------------------------------------------------
   // Form linear systems.
   // ----------------------------------------------------------------------------
   // ----------------------------------------------------------------------------
   // Form the linear systems for both 
   //       M_p xi = r_p, and
   //       S_p eta = r_p. 
   SparseMatrix Mp;
   Vector Rp_m, Xi;

   checkpoint(7);

   SparseMatrix Sp;
   Vector Rp_s, Eta;
   

   checkpoint(8);
   
   // Set up linear calculation for B^T y_p.
   SparseMatrix B;
   //OperatorPtr opB;
   Vector Brhs(pressure_fespace.GetTrueVSize()), BTyp(velocity_fespace.GetTrueVSize());
   //b.FormLinearSystem(boundary_dofs, yp, b, B, BTyp, Brhs);

   checkpoint(9);


   // ----------------------------------------------------------------------------
   // ----------------------------------------------------------------------------

   checkpoint(11);


   // Project Dirichlet boundary values for pressure.
   xi.ProjectBdrCoefficient(zero_DBC,ess_boundary_marker_pressure);
   eta.ProjectBdrCoefficient(pressure_DBC,ess_boundary_marker_pressure);

   // Form linear systems for Mp and Sp.
   mp.FormLinearSystem(pressure_ess_tdof, xi, rp, Mp, Xi, Rp_m);
   sp.FormLinearSystem(pressure_ess_tdof, eta, rp, Sp, Eta, Rp_s);
   
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
   //Solver M_prec = NULL;
   //M_prec = new OperatorJacobiSmoother(mp, pressure_ess_tdof);
   //OperatorJacobiSmoother M_prec;
   //M_prec.SetOperator(Mp);

   M_solver.iterative_mode = false;
   M_solver.SetRelTol(1e-8);
   M_solver.SetAbsTol(0.0);
   M_solver.SetMaxIter(10);
   M_solver.SetPrintLevel(1);
   //M_prec.SetType(DSmoother::Jacobi); // Works for now, but check if diagonal...
   M_solver.SetPreconditioner(M_prec);
   M_solver.SetOperator(Mp);

   M_solver.Mult(Rp_m,Xi);
   mp.RecoverFEMSolution(Xi, rp, xi);


   // Solve the S_p eta = r_p system using PCG with Jacobi preconditioner.
   CGSolver S_solver;
   DSmoother S_prec;

   S_solver.iterative_mode = false;
   S_solver.SetRelTol(1e-8);
   S_solver.SetAbsTol(0.0);
   S_solver.SetMaxIter(200);
   S_solver.SetPrintLevel(1);
   //S_prec.SetType(DSmoother::Jacobi); 
   S_solver.SetPreconditioner(S_prec);
   S_solver.SetOperator(Sp);

   S_solver.Mult(Rp_s,Eta);
   sp.RecoverFEMSolution(Eta, rp, eta);

   // Do the sum y_p = -alpha1 xi - eta.  Note: above alpha1 must be defined negative.
   yp.Add(neg_alpha1,xi);
   yp.Add(-1.0,eta);
   
   // Calculate B^T y_p and add B^T y_p to r_u for the velocity solve.
   Vector Yp(pressure_fespace.GetTrueVSize());;
   yp.GetTrueDofs(Yp);
   //Yp = yp;
   // Calculate B^T y_p.
   b.MultTranspose(Yp,BTyp);

   // Recover GridFunction from Vector.
   bTyp.SetFromTrueDofs(BTyp);

   // Add B^T y_p to r_u for the velocity solve.
   ru.Add(-1.0,bTyp);
   
   Vector Yu(velocity_fespace.GetTrueVSize()), Ru(velocity_fespace.GetTrueVSize());
   // Initialise Ru, Yu
   Ru = 0.0;
   
   //ru.ParallelAssemble(Ru);
   //yu.ParallelProject(Yu);    // https://github.com/mfem/mfem/issues/2791 comment Feb 17, 2022
   //fk.SetEssentialBC(velocity_ess_tdof, &ru);


   // Project Dirichlet boundary values for velocity.
   yu.ProjectBdrCoefficient(velocity_DBC,ess_boundary_marker_velocity);

   yu.GetTrueDofs(Yu);

   SparseMatrix Fk;
   // Set up system for F_k y_u = r_u (where r_u is r_u - B^T yp).
   fk.FormLinearSystem(velocity_ess_tdof, yu, ru, Fk, Yu, Ru);

   cout << "f_k operator size (height x width): " << fk.Height() << " x " << fk.Width() << endl;
   cout << "Ru size: " << Ru.Size() << endl;
   cout << "Yu size: " << Yu.Size() << endl;

   // Set up the solve for the nonlinear F_k y_u = r_u system.
   GMRESSolver F_solver;
   //CGSolver F_solver;
   OperatorJacobiSmoother F_prec;

   //F_solver.SetOperator(fk);
   F_solver.SetOperator(Fk);
   F_solver.SetPrintLevel(1);
   F_solver.SetRelTol(1e-10);
   F_solver.SetMaxIter(2000);
   //F_prec.SetType(SparseSmoother::Jacobi); // Schwarz preconditioner...?  ASM (PETSc)?
   F_solver.SetPreconditioner(F_prec);
   F_solver.SetPrintLevel(3);

   // Solve nonlinear system.
   
   cout << "Starting Mult..." << endl;
   F_solver.Mult(Ru, Yu);
   cout << "Mult completed." << endl;

   //Fk.Print();

   fk.RecoverFEMSolution(Yu, ru, yu);

   cout << "Yu vector size: " << Yu.Size() << endl;
   for (int i = 0; i < Yu.Size(); i++)
   {
      cout << Yu[i] << endl;
   }


   //}

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
   paraview_dc.RegisterField("p",&yp);

   // Export velocity data.
   paraview_dc.RegisterField("u",&yu);

   // Temporary intermediate values for debugging.
   paraview_dc.RegisterField("bTyp",&bTyp);
   paraview_dc.RegisterField("xi",&xi);
   paraview_dc.RegisterField("eta",&eta);

   paraview_dc.Save();


   // ----------------------------------------------------------------------------
   // ----------------------------------------------------------------------------

   return 0;
}

real_t velocity_dbc(const Vector & x)
{
   return 1.0;
}

real_t pressure_dbc(const Vector & x)
{
   if (x(0) > 0.0)
   { // Value at outlet.
      return 2.0;
   }
   else
   { // Value at inlet.
      return 1.0;
   }
}

real_t zero_dbc(const Vector & x)
{
   return 0.0;
}

void velocity_dbc_vec_func(const Vector & x, Vector & f)
{
   
   if (x(0) > 0.0)
   { // Zero on top and bottom boundaries.
      f(0) = 0.0;
      f(1) = 0.0;

      if (x.Size() == 3)
      {
         f(2) = 0.0;
      }
   } 
   else 
   { // One in x-direction at inlet.
      f(0) = 1.0;
      f(1) = 0.0;

      if (x.Size() == 3)
      {
         f(2) = 0.0;
      }
   }
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

double pFun_ex(const Vector & x)
{
   double xi(x(0));
   double yi(x(1));
   double zi(0.0);

   if (x.Size() == 3)
   {
      zi = x(2);
   }

   return exp(xi)*sin(yi)*cos(zi);
}

void fFun(const Vector & x, Vector & f)
{
   f = 0.0;
}

double gFun(const Vector & x)
{
   if (x.Size() == 3)
   {
      return -pFun_ex(x);
   }
   else
   {
      return 0.0;
   }
}

double f_natural(const Vector & x)
{
   return (-pFun_ex(x));
}

void checkpoint(int num)
{
   cout << "**********************************************" << endl;
   cout << "**************** CHECKPOINT " << num << " ****************" << endl;
   cout << "**********************************************" << endl;
   cout << endl;
}
