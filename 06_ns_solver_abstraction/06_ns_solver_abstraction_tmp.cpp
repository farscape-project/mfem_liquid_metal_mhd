#include "mfem.hpp"
#include <memory>
#include <iostream>
#include <fstream>

using namespace std;
using namespace mfem;

// Custom preconditioner class for solving Py = r.
class PPreconditioner : public Solver
{
protected:
   Array<FiniteElementSpace *> spaces;

   SparseMatrix *Mp, *Sp;

   mutable DSmoother M_prec, S_prec;
   mutable CGSolver M_solver, S_solver;
   mutable GMRESSolver F_solver;

   // Block offsets for variable access
   Array<int> &block_trueOffsets;

public:
   PPreconditioner(Array<FiniteElementSpace *> &spaces, Array<int> &offsets, SparseMatrix &Mp, SparseMatrix &Sp, SparseMatrix &Fk);

   virtual void Mult(const Vector &x, Vector &y) const;
   virtual void SetOperator(const Operator &op);

   virtual ~PPreconditioner();
};


void visualize(ParaViewDataCollection &paraview_dc, int order, GridFunction *field, 
   const char *field_name = NULL);

void checkpoint(int num);
real_t velocity_nbc(const Vector & x);
real_t pressure_nbc(const Vector & x);
real_t pressure_dbc(const Vector & x);
real_t zero_dbc(const Vector & x);
real_t outflow_term_func(const Vector & x);
void velocity_dbc_vec_func(const Vector & x, Vector & f);
void u_exact(const Vector & x, Vector & f);

// Define constants.
real_t Re(100.0);
real_t reciprocal_Re(1 / Re);
real_t alpha(1.0); // alpha = 1 (default).
real_t alpha1, neg_alpha1;  
//alpha1 = alpha + reciprocal_Re;
//neg_alpha1 = -alpha1;
real_t tau(0.1);

// Define ConstantCoefficients.
ConstantCoefficient zero(0.0);
ConstantCoefficient one(1.0);
ConstantCoefficient half(0.5);
ConstantCoefficient neg_one(-1.0);
ConstantCoefficient tmp_const(1.0);
ConstantCoefficient reciprocal_Re_coef(reciprocal_Re);
ConstantCoefficient vectorMassCoef(2.0 / tau);

int main(int argc, char *argv[])
{
   
   // Define VectorConstantCoefficients.
   Vector zero_vector(3), one_vector(3);
   zero_vector = 0.0;
   VectorConstantCoefficient zero_vector_coef(zero_vector);
   one_vector = 1.0;
   VectorConstantCoefficient one_vector_coef(one_vector);

   // Set fe_space orders.
   int order_pressure = 1;
   int order_velocity;
   order_velocity = order_pressure + 1;

   // Generate mesh.
   Mesh mesh = Mesh::MakeCartesian2D(30, 10, mfem::Element::Type::QUADRILATERAL, true, 1.0, 0.2);
   int dim = mesh.Dimension();   

   // ----------------------------------------------------------------------------
   // Finite Element Spaces.
   // ----------------------------------------------------------------------------
   // H1 continuous Lagrange finite elements of given order for pressure.
   H1_FECollection pressure_fec(order_pressure, dim);
   FiniteElementSpace pressure_fespace(&mesh, &pressure_fec);

   // H1 continuous Lagrange finite elements of given order (order_pressure + 1)
   // for velocity.
   H1_FECollection velocity_fec(order_velocity, dim);
   FiniteElementSpace velocity_fespace(&mesh, &velocity_fec, dim);

   Array<FiniteElementSpace *> spaces(2);
   spaces[0] = &velocity_fespace;
   spaces[1] = &pressure_fespace;

   // ----------------------------------------------------------------------------
   // Define boundaries.kmkml
   // ----------------------------------------------------------------------------

   VectorFunctionCoefficient velocity_DBC(dim, velocity_dbc_vec_func);
   FunctionCoefficient pressure_DBC(pressure_dbc);
   FunctionCoefficient zero_DBC(zero_dbc);

   // Essential (Dirichlet) boundary conditions for pressure.
   Array<int> ess_boundary_marker_pressure, ess_boundary_marker_velocity; 
   ess_boundary_marker_pressure.SetSize(pressure_fespace.GetMesh()->bdr_attributes.Max());
   ess_boundary_marker_pressure = 0; 
   // Dirichlet boundary condition for pressure.
   ess_boundary_marker_pressure[0] = 0; // Top
   ess_boundary_marker_pressure[1] = 1; // Outlet
   ess_boundary_marker_pressure[2] = 0; // Bottom
   ess_boundary_marker_pressure[3] = 1; // Inlet

   // Essential (Dirichlet) boundary conditions for velocity.
   ess_boundary_marker_velocity.SetSize(velocity_fespace.GetMesh()->bdr_attributes.Max());
   ess_boundary_marker_velocity = 0;
   // Dirichlet boundary conditions for velocity.
   ess_boundary_marker_velocity[0] = 1; // Top
   ess_boundary_marker_velocity[1] = 0; // Outlet
   ess_boundary_marker_velocity[2] = 1; // Bottom
   ess_boundary_marker_velocity[3] = 0; // Inlet

   Array<Array<int> *> ess_bdr(2);
   ess_bdr[0] = &ess_boundary_marker_velocity;
   ess_bdr[1] = &ess_boundary_marker_pressure;

   // Natural (Neumann) boundary conditions for pressure.
   Array<int> nat_boundary_marker_pressure, nat_boundary_marker_velocity; 
   nat_boundary_marker_pressure.SetSize(pressure_fespace.GetMesh()->bdr_attributes.Max());
   nat_boundary_marker_pressure = 0;
   // Set natural bcs for pressure.
   nat_boundary_marker_pressure[0] = 1; // Top
   nat_boundary_marker_pressure[1] = 0; // Outlet
   nat_boundary_marker_pressure[2] = 1; // Bottom
   nat_boundary_marker_pressure[3] = 0; // Inlet

   // Natural (Neumann) boundary conditions for velocity.
   nat_boundary_marker_velocity.SetSize(velocity_fespace.GetMesh()->bdr_attributes.Max());
   nat_boundary_marker_velocity = 0;
   // Set natural bcs for velocity.
   nat_boundary_marker_velocity[0] = 0; // Top
   nat_boundary_marker_velocity[1] = 1; // Outlet
   nat_boundary_marker_velocity[2] = 0; // Bottom
   nat_boundary_marker_velocity[3] = 1; // Inlet

   Array<int> pressure_ess_tdof, velocity_ess_tdof;
   pressure_fespace.GetEssentialTrueDofs(ess_boundary_marker_pressure, pressure_ess_tdof);
   velocity_fespace.GetEssentialTrueDofs(ess_boundary_marker_velocity, velocity_ess_tdof);


   // Define block structure of the solution vector (u then p).
   Array<int> block_trueOffsets(3);
   block_trueOffsets[0] = 0;
   block_trueOffsets[1] = pressure_fespace.GetTrueVSize();
   block_trueOffsets[2] = pressure_fespace.GetTrueVSize();
   block_trueOffsets.PartialSum();

   BlockVector X(block_trueOffsets);
   BlockVector Y(block_trueOffsets);

   // ----------------------------------------------------------------------------
   // Initialise solutions as GridFunctions.
   // ----------------------------------------------------------------------------
   // Define the solutions xi, eta, p as grid functions for pressure. 
   GridFunction xi(&pressure_fespace);
   GridFunction eta(&pressure_fespace);
   GridFunction yp(&pressure_fespace);

   // Define the solution for velocity.
   GridFunction yu(&velocity_fespace);
   // Solution for B^T * y_p.
   GridFunction bTyp(&velocity_fespace);

   xi.MakeTRef(&pressure_fespace, Y.GetBlock(0), 0);
   eta.MakeTRef(&pressure_fespace, Y.GetBlock(1), 0);

   // Set initial guesses to zero.  This also sets BCs.
   xi = 0.0;
   eta = 0.0;
   yp = 0.0;
   yu = 0.0;
   bTyp = 0.0;


   FunctionCoefficient velocity_nbc_coeff(velocity_nbc);
   FunctionCoefficient pressure_nbc_coeff(pressure_nbc);

   // Set up rhs for pressure solve.
   LinearForm rp(&pressure_fespace);
   rp.AddDomainIntegrator(new DomainLFIntegrator(zero));
   // Natural boundary condition.
   rp.AddBoundaryIntegrator(new BoundaryLFIntegrator(pressure_nbc_coeff), nat_boundary_marker_pressure);
   rp.Assemble();

   // Set up rhs for velocity solve.
   LinearForm ru(&velocity_fespace);
   ru.AddDomainIntegrator(new VectorDomainLFIntegrator(zero_vector_coef));
   // Natural boundary condition.
   ru.AddBoundaryIntegrator(new VectorBoundaryFluxLFIntegrator(velocity_nbc_coeff), nat_boundary_marker_velocity);
   ru.Assemble();

   MixedBilinearForm b(&velocity_fespace,&pressure_fespace);
   b.AddDomainIntegrator(new VectorDivergenceIntegrator(neg_one));
   b.Assemble();
   b.Finalize();


   // Bilinear forms for the two-part pressure solve.
   BilinearForm mp(&pressure_fespace);
   mp.AddDomainIntegrator(new MassIntegrator(one));
   mp.Assemble();

   BilinearForm sp(&pressure_fespace);
   sp.AddDomainIntegrator(new DiffusionIntegrator(one));
   sp.Assemble();


   // Form the linear systems for both 
   //       M_p xi = r_p, and
   //       S_p eta = r_p. 
   SparseMatrix Mp, Sp;
   Vector Xi, Eta, Rp_m, Rp_s;
      

   SparseMatrix Fk;

   // Project Dirichlet boundary values for pressure.
   xi.ProjectBdrCoefficient(zero_DBC,pressure_ess_tdof);
   eta.ProjectBdrCoefficient(pressure_DBC,pressure_ess_tdof);

   // Form linear system for Mp.
   mp.FormLinearSystem(pressure_ess_tdof, xi, rp, Mp, Xi, Rp_m);

   // Form linear system for Sp.
   sp.FormLinearSystem(pressure_ess_tdof, eta, rp, Sp, Eta, Rp_s);

   xi.SetTrueVector();
   eta.SetTrueVector();


   PPreconditioner precond(spaces, block_trueOffsets, Mp, Sp, Fk);
   precond.Mult(X, Y);

   xi.SetFromTrueVector();
   eta.SetFromTrueVector();


   mp.RecoverFEMSolution(Xi, rp, xi);
   sp.RecoverFEMSolution(Eta, rp, eta);

   // Set up visualisation in Paraview.
   ParaViewDataCollection paraview_dc("navier_stokes", &mesh);
   paraview_dc.SetPrefixPath("data");
   visualize(paraview_dc, order_pressure, &xi, "xi");
   visualize(paraview_dc, order_pressure, &eta, "Eta");


   return 0;
}


PPreconditioner::PPreconditioner(Array<FiniteElementSpace *> &spaces, Array<int> &offsets, SparseMatrix &Mp, SparseMatrix &Sp, SparseMatrix &Fk)
: block_trueOffsets(offsets)
{

   M_solver.iterative_mode = false;
   M_solver.SetRelTol(1e-8);
   M_solver.SetAbsTol(0.0);
   M_solver.SetMaxIter(10);
   M_solver.SetPrintLevel(1);
   //M_prec.SetType(DSmoother::Jacobi); // Works for now, but check if diagonal...
   M_solver.SetPreconditioner(M_prec);
   M_solver.SetOperator(Mp);

   S_solver.iterative_mode = false;
   S_solver.SetRelTol(1e-8);
   S_solver.SetAbsTol(0.0);
   S_solver.SetMaxIter(200);
   S_solver.SetPrintLevel(1);
   //S_prec.SetType(DSmoother::Jacobi); 
   S_solver.SetPreconditioner(S_prec);
   S_solver.SetOperator(Sp);

   /*F_solver.SetOperator(Fk);
   F_solver.SetRelTol(1e-10);
   F_solver.SetMaxIter(2000);
   F_solver.SetPrintLevel(3);*/
}

void PPreconditioner::Mult(const Vector &x, Vector &y) const
{
   /*
   Vector Xi( y.GetData() + block_trueOffsets[0], block_trueOffsets[1] - block_trueOffsets[0]);
   Vector Eta(y.GetData() + block_trueOffsets[1], block_trueOffsets[2] - block_trueOffsets[1]);
   

   // Apply M_solver and S_solver to get results
   Vector Rp_m, Rp_s;  // Result of M_solver and S_solver combined
   M_solver.Mult(Rp_m, Xi);
   S_solver.Mult(Rp_s, Eta);

   // Combine results into output vector y
   for (int i = 0; i < y.Size() / 2; i++) {
      y(i) = Xi(i);
      y(i + y.Size() / 2) = Eta(i);
   }*/

   Vector Rp_m(x.GetData() + block_trueOffsets[0], block_trueOffsets[1] - block_trueOffsets[0]);
   Vector Rp_s(x.GetData() + block_trueOffsets[1], block_trueOffsets[2] - block_trueOffsets[1]);

   Vector Xi( y.GetData() + block_trueOffsets[0], block_trueOffsets[1] - block_trueOffsets[0]);
   Vector Eta(y.GetData() + block_trueOffsets[1], block_trueOffsets[2] - block_trueOffsets[1]);
   
   // Apply M_solver and S_solver to get results
   M_solver.Mult(Rp_m, Xi);
   S_solver.Mult(Rp_s, Eta);
   
}

void PPreconditioner::SetOperator(const Operator &op)
{

}

PPreconditioner::~PPreconditioner()
{

}


real_t velocity_nbc(const Vector & x)
{
   return 0.0;
}

real_t pressure_nbc(const Vector & x)
{
   return 0.0;
}

real_t pressure_dbc(const Vector & x)
{
   if (x(0) > 0.0)
   { // Value at outlet.
      return 0.0;
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

real_t outflow_term_func(const Vector & x)
{

   real_t val = 1.0;

   if (x(0) < 0.99999)
   { // Value at inlet.
      return 0.0;
   }
   else
   { // Value at outlet.
      return 0.5 * val;
   }
}


void velocity_dbc_vec_func(const Vector & x, Vector & f)
{
   real_t pi = 3.14159;

   real_t r_max = 0.1;
   real_t u_avg = 1.0;
   
   if (x(0) > 0.00005)
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
      //f(0) = sin(5 * pi * x(1));
      f(0) = u_avg * (1. - ((x(1)-r_max)*(x(1)-r_max)) / (r_max*r_max));
      //f(0) = 1.0;
      f(1) = 0.0;

      if (x.Size() == 3)
      {
         f(2) = 0.0;
      }
   }
}

void u_exact(const mfem::Vector & x, mfem::Vector & f)
{

   real_t r_max = 0.1;
   real_t u_avg = 1.0;

   f(0) = u_avg * (1. - ((x(1)-r_max)*(x(1)-r_max)) / (r_max*r_max));
   f(1) = 0.0;
   if (x.Size() == 3)
   {
      f(2) = 0.0;
   }
   
}


void checkpoint(int num)
{
   cout << "**********************************************" << endl;
   cout << "**************** CHECKPOINT " << num << " ****************" << endl;
   cout << "**********************************************" << endl;
   cout << endl;
}

void initial_velocity(const Vector & x, Vector & f)
{
   real_t r_max = 0.1;
   real_t u_avg = 1.0;

   f(0) = u_avg * (1. - ((x(1)-r_max)*(x(1)-r_max)) / (r_max*r_max));
}

// Inline visualization
void visualize(ParaViewDataCollection &paraview_dc, int order, GridFunction *field, const char *field_name)
{

   paraview_dc.SetLevelsOfDetail(order);
   paraview_dc.SetCycle(0);
   paraview_dc.SetDataFormat(VTKFormat::BINARY);
   paraview_dc.SetHighOrderOutput(true);
   paraview_dc.SetTime(0.0); // set the time

   // Export field data.
   paraview_dc.RegisterField(field_name,field);

   paraview_dc.Save();
}
