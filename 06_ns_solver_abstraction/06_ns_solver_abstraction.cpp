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
   // Finite element spaces for setting up preconditioner blocks
   Array<FiniteElementSpace *> spaces;

   // Offsets for extracting block vector segments
   Array<int> &block_trueOffsets;

   // Jacobian for block access
   BlockOperator *jacobian;

   // Scaling factor for the pressure mass matrix in the block preconditioner
   real_t gamma;

   // Objects for the block preconditioner application
   SparseMatrix *pressure_mass;
   Solver *mass_pcg;
   Solver *mass_prec;
   Solver *stiff_pcg;
   Solver *stiff_prec;

public:
   PPreconditioner(Array<FiniteElementSpace *> &fes,
                          SparseMatrix &mass, Array<int> &offsets);

   virtual void Mult(const Vector &k, Vector &y) const;
   virtual void SetOperator(const Operator &op);

   virtual ~PPreconditioner();
};

// Operator for solving Ax = b.
class AOperator : public Operator
{
protected:
   // Finite element spaces
   Array<FiniteElementSpace *> spaces;

   // Block nonlinear form
   BlockNonlinearForm *Hform;

   // Pressure mass matrix for the preconditioner
   SparseMatrix *pressure_mass;

   // Newton solver for the hyperelastic operator
   NewtonSolver newton_solver;

   // Solver for the Jacobian solve in the Newton method
   Solver *j_solver;

   // Preconditioner
   Solver *j_prec;

   // Define relevant coefficients
   //Coefficient &mu;

   // Block offsets for variable access
   Array<int> &block_trueOffsets;

public:
   AOperator(Array<FiniteElementSpace *> &fes, Array<Array<int> *>&ess_bdr,
                  Array<int> &block_trueOffsets);

   // Required to use the native newton solver
   virtual Operator &GetGradient(const Vector &xp) const;
   virtual void Mult(const Vector &k, Vector &y) const;

   // Driver for the newton solver
   void Solve(Vector &xp) const;

   virtual ~AOperator();
};


int main(int argc, char *argv[])
{
   // Define constants.
   real_t Re(100.0);
   real_t reciprocal_Re(1 / Re);
   real_t alpha(1.0); // alpha = 1 (default).
   real_t alpha1, neg_alpha1;  
   alpha1 = alpha + reciprocal_Re;
   neg_alpha1 = -alpha1;
   real_t tau(0.1);

   // Define ConstantCoefficients.
   ConstantCoefficient zero(0.0);
   ConstantCoefficient one(1.0);
   ConstantCoefficient half(0.5);
   ConstantCoefficient neg_one(-1.0);
   ConstantCoefficient tmp_const(1.0);
   ConstantCoefficient reciprocal_Re_coef(reciprocal_Re);
   ConstantCoefficient vectorMassCoef(2.0 / tau);

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
   // Define boundaries.
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
   block_trueOffsets[1] = velocity_fespace.GetTrueVSize();
   block_trueOffsets[2] = pressure_fespace.GetTrueVSize();
   block_trueOffsets.PartialSum();

   BlockVector y(block_trueOffsets);

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

   // Set initial guesses to zero.  This also sets BCs.
   xi = 0.0;
   eta = 0.0;
   yp = 0.0;
   yu = 0.0;
   bTyp = 0.0;

   yu.MakeTRef(&velocity_fespace, y.GetBlock(0), 0);
   yp.MakeTRef(&pressure_fespace, y.GetBlock(1), 0);

   //x_gf.ProjectCoefficient(deform);
   //x_ref.ProjectCoefficient(refconfig);
   //p_gf = 0.0;

   //x_gf.SetTrueVector();
   //p_gf.SetTrueVector();

   // Initialize operator for Ax = b solve.  Arguments need updating.
   AOperator oper(spaces, ess_bdr, block_trueOffsets);

   // Do the solve.
   oper.Solve(y);

   // Visualize results.

   return 0;
}


PPreconditioner::PPreconditioner(Array<FiniteElementSpace *> &fes,
                                               SparseMatrix &mass,
                                               Array<int> &offsets)
   : Solver(offsets[2]), block_trueOffsets(offsets), pressure_mass(&mass)
{
   /*fes.Copy(spaces);

   gamma = 0.00001;

   // Define things here that do not change during Newton iterations.  
   // Things that do need to be updated go in SetOperator.
   GSSmoother *mass_prec_gs = new GSSmoother(*pressure_mass);

   mass_prec = mass_prec_gs;

   CGSolver *mass_pcg_iter = new CGSolver();
   mass_pcg_iter->SetRelTol(1e-12);
   mass_pcg_iter->SetAbsTol(1e-12);
   mass_pcg_iter->SetMaxIter(200);
   mass_pcg_iter->SetPrintLevel(0);
   mass_pcg_iter->SetPreconditioner(*mass_prec);
   mass_pcg_iter->SetOperator(*pressure_mass);
   mass_pcg_iter->iterative_mode = false;

   mass_pcg = mass_pcg_iter;

   // The stiffness matrix does change every Newton cycle, so we will define it
   // during SetOperator
   stiff_pcg = NULL;
   stiff_prec = NULL;*/
}

void PPreconditioner::Mult(const Vector &k, Vector &y) const
{
   // Extract the blocks from the input and output vectors
   /*Vector disp_in(k.GetData() + block_trueOffsets[0],
                  block_trueOffsets[1]-block_trueOffsets[0]);
   Vector pres_in(k.GetData() + block_trueOffsets[1],
                  block_trueOffsets[2]-block_trueOffsets[1]);

   Vector disp_out(y.GetData() + block_trueOffsets[0],
                   block_trueOffsets[1]-block_trueOffsets[0]);
   Vector pres_out(y.GetData() + block_trueOffsets[1],
                   block_trueOffsets[2]-block_trueOffsets[1]);

   Vector temp(block_trueOffsets[1]-block_trueOffsets[0]);
   Vector temp2(block_trueOffsets[1]-block_trueOffsets[0]);

   // Perform the block elimination for the preconditioner
   mass_pcg->Mult(pres_in, pres_out);
   pres_out *= -gamma;

   jacobian->GetBlock(0,1).Mult(pres_out, temp);
   subtract(disp_in, temp, temp2);

   stiff_pcg->Mult(temp2, disp_out);*/
}

void PPreconditioner::SetOperator(const Operator &op)
{
   /*jacobian = (BlockOperator *) &op;

   // Initialize the stiffness preconditioner and solver
   if (stiff_prec == NULL)
   {
      GSSmoother *stiff_prec_gs = new GSSmoother();

      stiff_prec = stiff_prec_gs;

      GMRESSolver *stiff_pcg_iter = new GMRESSolver();
      stiff_pcg_iter->SetRelTol(1e-8);
      stiff_pcg_iter->SetAbsTol(1e-8);
      stiff_pcg_iter->SetMaxIter(200);
      stiff_pcg_iter->SetPrintLevel(0);
      stiff_pcg_iter->SetPreconditioner(*stiff_prec);
      stiff_pcg_iter->iterative_mode = false;

      stiff_pcg = stiff_pcg_iter;
   }

   // At each Newton cycle, compute the new stiffness preconditioner by updating
   // the iterative solver which, in turn, updates its preconditioner
   stiff_pcg->SetOperator(jacobian->GetBlock(0,0));*/
}

PPreconditioner::~PPreconditioner()
{
   delete mass_pcg;
   delete mass_prec;
   delete stiff_prec;
   delete stiff_pcg;
}


AOperator::AOperator(Array<FiniteElementSpace *> &fes,
                               Array<Array<int> *> &ess_bdr,
                               Array<int> &offsets)
   : Operator(fes[0]->GetTrueVSize() + fes[1]->GetTrueVSize()),
     newton_solver(), block_trueOffsets(offsets)
{
   Array<Vector *> rhs(2);
   rhs = NULL; // Set all entries in the array

   fes.Copy(spaces);

   // Define the block nonlinear form
   Hform = new BlockNonlinearForm(spaces);

   // Add the incompressible neo-Hookean integrator
   //Hform->AddDomainIntegrator(new IncompressibleNeoHookeanIntegrator(mu));

   // Set the essential boundary conditions
   Hform->SetEssentialBC(ess_bdr, rhs);

   // Compute the pressure mass stiffness matrix
   BilinearForm *a = new BilinearForm(spaces[1]);
   ConstantCoefficient one(1.0);
   a->AddDomainIntegrator(new MassIntegrator(one));
   a->Assemble();
   a->Finalize();

   OperatorPtr op;
   Array<int> p_ess_tdofs;
   a->FormSystemMatrix(p_ess_tdofs, op);
   pressure_mass = a->LoseMat();
   delete a;

   // Initialize the Jacobian preconditioner
   PPreconditioner *jac_prec =
      new PPreconditioner(fes, *pressure_mass, block_trueOffsets);
   j_prec = jac_prec;

   // Set up the Jacobian solver
   GMRESSolver *j_gmres = new GMRESSolver();
   j_gmres->iterative_mode = false;
   j_gmres->SetRelTol(1e-12);
   j_gmres->SetAbsTol(1e-12);
   j_gmres->SetMaxIter(300);
   j_gmres->SetPrintLevel(-1);
   j_gmres->SetPreconditioner(*j_prec);
   j_solver = j_gmres;

   real_t newton_rel_tol = 1e-4;
   real_t newton_abs_tol = 1e-6;
   int newton_iter = 500;

   // Set the newton solve parameters
   newton_solver.iterative_mode = true;
   newton_solver.SetSolver(*j_solver);
   newton_solver.SetOperator(*this);
   newton_solver.SetPrintLevel(-1);
   newton_solver.SetRelTol(newton_rel_tol);
   newton_solver.SetAbsTol(newton_abs_tol);
   newton_solver.SetMaxIter(newton_iter);
}

// Solve the Newton system
void AOperator::Solve(Vector &xp) const
{
   Vector zero;
   newton_solver.Mult(zero, xp);
   MFEM_VERIFY(newton_solver.GetConverged(),
               "Newton Solver did not converge.");
}

// compute: y = H(x,p)
void AOperator::Mult(const Vector &k, Vector &y) const
{
   Hform->Mult(k, y);
}

// Compute the Jacobian from the nonlinear form
Operator &AOperator::GetGradient(const Vector &xp) const
{
   return Hform->GetGradient(xp);
}

AOperator::~AOperator()
{
   delete Hform;
   delete pressure_mass;
   delete j_solver;
   delete j_prec;
}


// Inline visualization
void visualize(ostream &os, Mesh *mesh, GridFunction *deformed_nodes,
               GridFunction *field, const char *field_name, bool init_vis)
{
   if (!os)
   {
      return;
   }

   GridFunction *nodes = deformed_nodes;
   int owns_nodes = 0;

   mesh->SwapNodes(nodes, owns_nodes);

   os << "solution\n" << *mesh << *field;

   mesh->SwapNodes(nodes, owns_nodes);

   if (init_vis)
   {
      os << "window_size 800 800\n";
      os << "window_title '" << field_name << "'\n";
      if (mesh->SpaceDimension() == 2)
      {
         os << "view 0 0\n"; // view from top
         // turn off perspective and light, +anti-aliasing
         os << "keys jlA\n";
      }
      os << "keys cmA\n"; // show colorbar and mesh, +anti-aliasing
      // update value-range; keep mesh-extents fixed
      os << "autoscale value\n";
   }
   os << flush;
}

void ReferenceConfiguration(const Vector &x, Vector &y)
{
   // Set the reference, stress free, configuration
   y = x;
}

void InitialDeformation(const Vector &x, Vector &y)
{
   // Set the initial configuration. Having this different from the reference
   // configuration can help convergence
   y = x;
   y[1] = x[1] + 0.25*x[0];
}
