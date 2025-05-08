#include "mfem.hpp"
#include <memory>
#include <iostream>
#include <fstream>
#include "custom_integrators.hpp"

using namespace std;
using namespace mfem;

// Custom preconditioner class for solving Py = r.
class PPreconditioner : public Solver
{
protected:
   Array<FiniteElementSpace *> spaces;

   SparseMatrix &Mp, &Sp, &Fk;

   mutable CGSolver M_solver, S_solver;
   mutable DSmoother M_prec, S_prec;
   mutable GMRESSolver F_solver;

   // Block offsets for variable access 
   Array<int> &block_trueOffsets;

   GridFunction &xi, &eta, &yu, &yp;
   //GridFunction bTyp;

   LinearForm &rp, &ru;

   BilinearForm &mp, &sp, &fk;

   MixedBilinearForm &b;

   const Array<int> &pressure_ess_tdof, &velocity_ess_tdof;
   FunctionCoefficient *pressure_DBC, *zero_DBC;
   VectorFunctionCoefficient *velocity_DBC;

public:
   PPreconditioner(Array<FiniteElementSpace *> &fes, Array<int> &offsets, SparseMatrix &Mp, SparseMatrix &Sp, SparseMatrix &Fk, 
      GridFunction &yu, GridFunction &yp, GridFunction &xi, GridFunction &eta, BilinearForm &mp, BilinearForm &sp, BilinearForm &fk, LinearForm &rp, 
      LinearForm &ru, MixedBilinearForm &b, const Array<int> &pressure_ess_tdof, const Array<int> &velocity_ess_tdof, 
      VectorFunctionCoefficient *velocity_DBC_, FunctionCoefficient *pressure_DBC_, FunctionCoefficient *zero_DBC_);

    void SetFunctionCoefficients(VectorFunctionCoefficient *velocity_DBC_, FunctionCoefficient *pressure_DBC_, FunctionCoefficient *zero_DBC_)
    {
      velocity_DBC = velocity_DBC_;
      pressure_DBC = pressure_DBC_;
      zero_DBC = zero_DBC_;
    }

   virtual void Mult(const Vector &x, Vector &y) const;
   virtual void SetOperator(const Operator &op);

   virtual ~PPreconditioner();
};

// Operator for solving Ax = b.
/*class AOperator : public Operator
{
protected:
   // Finite element spaces
   Array<FiniteElementSpace *> spaces;

   // Block nonlinear form
   BlockLinearForm *Hform;

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
};*/


void CheckConvectionIntegrals(const DenseMatrix &elmat, const DenseMatrix &elmat_conservative, const FiniteElement &el) {
   int num_dofs = el.GetDof();
   
   for (int i = 0; i < num_dofs; i++) {
      for (int j = 0; j < num_dofs; j++) {
         real_t diff = elmat(i, j) + elmat_conservative(i, j); // Should ideally be 0 if they are transposed.
         
         if (std::abs(diff) > 1e-6) {
            std::cerr << "Error: Convection terms do not cancel out. Difference at (" 
                      << i << ", " << j << "): " << diff << std::endl;
         }
      }
   }
}


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
real_t alpha1(alpha + reciprocal_Re);
real_t neg_alpha1(-alpha1);
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
   Mesh mesh = Mesh::MakeCartesian2D(10, 4, mfem::Element::Type::QUADRILATERAL, true, 1.0, 0.2);
   //Mesh mesh = Mesh::MakeCartesian2D(30, 10, mfem::Element::Type::QUADRILATERAL, true, 1.0, 0.2);
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


   // Define block structure of the solution vector (p (twice) then u).
   Array<int> block_trueOffsets(4);
   block_trueOffsets[0] = 0;
   block_trueOffsets[1] = pressure_fespace.GetTrueVSize();
   block_trueOffsets[2] = pressure_fespace.GetTrueVSize();
   block_trueOffsets[3] = velocity_fespace.GetTrueVSize();
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
   //GridFunction bTyp(&velocity_fespace);

   xi.MakeTRef(&pressure_fespace, Y.GetBlock(0), 0);
   eta.MakeTRef(&pressure_fespace, Y.GetBlock(1), 0);
   yu.MakeTRef(&velocity_fespace, Y.GetBlock(2), 0);

   // Set initial guesses to zero.  This also sets BCs.
   xi = 0.0;
   eta = 0.0;
   yp = 0.0;
   yu = 0.0;
   //bTyp = 0.0;

   //yp.MakeTRef(&pressure_fespace, Y.GetBlock(0), 0);
   //yp.MakeTRef(&pressure_fespace, Y.GetBlock(1), 0);

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

   FunctionCoefficient outflow_term(outflow_term_func);

   GridFunction ustar_n(&velocity_fespace); // 0.5(3u_{n-1} - u_{n-2})
   VectorFunctionCoefficient ucoef(dim, u_exact);

   std::cout << "ustar_n.VectorDim() = " << ustar_n.VectorDim() << std::endl;
   std::cout << "ucoef.GetVDim() = " << ucoef.GetVDim() << std::endl;

   ustar_n.ProjectCoefficient(ucoef);
   VectorGridFunctionCoefficient ustar_coef(&ustar_n);

   
   Vector simpleVec(2);
   simpleVec = 1.0;
   VectorConstantCoefficient simpleCoeff(simpleVec);

   std::cout << "simpleVec.Size() = " << simpleVec.Size() << std::endl;
   for (int i = 0; i < simpleVec.Size(); i++) {
      cout << simpleVec[i] << endl;
   }

   std::cout << "simpleCoeff dimension: " << simpleCoeff.GetVDim() << std::endl;
   std::cout << "Mesh dimension: " << velocity_fespace.GetMesh()->Dimension() << std::endl;

   // Bilinear form for the velocity solve.
   BilinearForm fk(&velocity_fespace);
   // Integrator for (v, v').
   fk.AddDomainIntegrator(new VectorMassIntegrator(vectorMassCoef));
   // Integrator for A_AL(v, v').
   fk.AddDomainIntegrator(new VectorDiffusionIntegrator(reciprocal_Re_coef));

   std::cout << "ustar_coef.GetVDim() = " << ustar_coef.GetVDim() << std::endl;
   std::cout << "velocity_fespace.GetVDim() = " << velocity_fespace.GetVDim() << std::endl;

   // Integrator for O(u_n, v, v').
   //fk.AddDomainIntegrator(new ConvectionIntegrator(simpleCoeff, 0.5));
   //fk.AddDomainIntegrator(new ConservativeConvectionIntegrator(simpleCoeff, 0.5));
   
   std::cout << "fk Height: " << fk.Height() << ", Width: " << fk.Width() << std::endl;   

   
   fk.AddDomainIntegrator(new VectorConvectionIntegrator(ustar_coef,0.5));
   fk.AddDomainIntegrator(new ConservativeVectorConvectionIntegrator(ustar_coef,0.5));

   //fk.AddDomainIntegrator(new ConvectionIntegrator(ustar_coef, 0.5));
   //fk.AddDomainIntegrator(new ConservativeConvectionIntegrator(ustar_coef, 0.5));

   // Outflow boundary term.
   fk.AddBoundaryIntegrator(new VectorMassIntegrator(outflow_term));

   fk.Assemble(); // Crashing here!  Issue with ConvectionIntegrators.
   fk.Finalize();

   // Test...
   const FiniteElement &el = *velocity_fespace.GetFE(0); // get FE for element 0
   ElementTransformation &Trans = *velocity_fespace.GetElementTransformation(0);

   DenseMatrix elmat_std, elmat_cons;
   VectorConvectionIntegrator standard_integrator(ustar_coef,0.5);
   ConservativeVectorConvectionIntegrator conservative_integrator(ustar_coef,0.5);

   standard_integrator.AssembleElementMatrix(el, Trans, elmat_std);
   conservative_integrator.AssembleElementMatrix(el, Trans, elmat_cons);

   CheckConvectionIntegrals(elmat_std, elmat_cons, el);

   // Form the linear systems for both 
   //       M_p xi = r_p, and
   //       S_p eta = r_p. 
   SparseMatrix Mp, Sp;
   Vector Xi(pressure_fespace.GetTrueVSize()), Eta(pressure_fespace.GetTrueVSize());
   Vector Rp_m(pressure_fespace.GetTrueVSize()), Rp_s(pressure_fespace.GetTrueVSize());
   Vector Yu(velocity_fespace.GetTrueVSize()), Ru(velocity_fespace.GetTrueVSize());

      
   // Set up linear calculation for B^T y_p.
   //SparseMatrix B;
   //Vector Brhs(pressure_fespace.GetTrueVSize()), BTyp(velocity_fespace.GetTrueVSize());

   SparseMatrix Fk;

   xi.SetTrueVector();
   eta.SetTrueVector();
   yu.SetTrueVector();

   // Copy the right-hand-side elements into X
   for (int i = 0; i < Xi.Size(); i++) {
      X(i) = Rp_m(i); 
   }
   for (int i = 0; i < Eta.Size(); i++) {
      X(i + Xi.Size()) = Rp_s(i);  
   }
   for (int i = 0; i < Yu.Size(); i++) {
      X(i + Xi.Size() + Eta.Size()) = Ru(i);
   }

   //PPreconditioner precond(spaces, block_trueOffsets, Mp, Sp, Fk);
   PPreconditioner precond(spaces, block_trueOffsets, Mp, Sp, Fk, yu, yp, xi, eta, mp, sp, fk, rp, ru, b, pressure_ess_tdof, velocity_ess_tdof, 
      &velocity_DBC, &pressure_DBC, &zero_DBC);
   precond.SetFunctionCoefficients(&velocity_DBC, &pressure_DBC, &zero_DBC);

   precond.Mult(X, Y);
   
   //xi.SetFromTrueVector();
   //eta.SetFromTrueVector();

   //yp.Add(neg_alpha1,xi);
   //yp.Add(-1.0,eta);


   /*cout << "Grid function values for yp, xi, eta:" << endl;
   for (int i = 0; i < pressure_fespace.GetTrueVSize(); i++)
   {
      cout << i << " " << yp[i] << " " << xi[i] << " " << eta[i] << endl;
   }*/

   // Set up visualisation in Paraview.
   ParaViewDataCollection paraview_dc("navier_stokes", &mesh);
   paraview_dc.SetPrefixPath("data");
   visualize(paraview_dc, order_pressure, &xi, "xi");
   visualize(paraview_dc, order_pressure, &eta, "eta");
   visualize(paraview_dc, order_pressure, &yp, "pressure");
   visualize(paraview_dc, order_velocity, &yu, "velocity");


   // Initialize operator for Ax = b solve.  Arguments need updating.
   //AOperator oper(spaces, ess_bdr, block_trueOffsets);

   // Do the solve.
   //oper.Solve(y);

   return 0;
}

PPreconditioner::PPreconditioner(Array<FiniteElementSpace *> &fes, Array<int> &offsets, SparseMatrix &Mp_, SparseMatrix &Sp_, SparseMatrix &Fk_, 
   GridFunction &yu_, GridFunction &yp_, GridFunction &xi_, GridFunction &eta_, BilinearForm &mp_, BilinearForm &sp_, BilinearForm &fk_, LinearForm &rp_, 
   LinearForm &ru_, MixedBilinearForm &b_,const Array<int> &pressure_ess_tdof_, const Array<int> &velocity_ess_tdof_, 
   VectorFunctionCoefficient *velocity_DBC_, FunctionCoefficient *pressure_DBC_, FunctionCoefficient *zero_DBC_)
   : Mp(Mp_), Sp(Sp_), Fk(Fk_), block_trueOffsets(offsets), xi(xi_), eta(eta_), yu(yu_), yp(yp_), rp(rp_), ru(ru_), mp(mp_), sp(sp_), fk(fk_), b(b_), 
   velocity_ess_tdof(velocity_ess_tdof_), pressure_ess_tdof(pressure_ess_tdof_), velocity_DBC(velocity_DBC_), pressure_DBC(pressure_DBC_), zero_DBC(zero_DBC_)
   {

   fes.Copy(spaces);

   M_solver.iterative_mode = false;
   M_solver.SetRelTol(1e-8);
   M_solver.SetAbsTol(0.0);
   M_solver.SetMaxIter(10);
   M_solver.SetPrintLevel(1);
   //M_prec.SetType(DSmoother::Jacobi); // Works for now, but check if diagonal...
   M_solver.SetPreconditioner(M_prec);

   S_solver.iterative_mode = false;
   S_solver.SetRelTol(1e-8);
   S_solver.SetAbsTol(0.0);
   S_solver.SetMaxIter(200);
   S_solver.SetPrintLevel(1);
   //S_prec.SetType(DSmoother::Jacobi); 
   S_solver.SetPreconditioner(S_prec);

   
   F_solver.SetRelTol(1e-10);
   F_solver.SetMaxIter(2000);
   F_solver.SetPrintLevel(3);

}

void PPreconditioner::Mult(const Vector &x, Vector &y) const
{

   int idx(0);
   Vector Rp_m(x.GetData() + block_trueOffsets[idx], block_trueOffsets[idx+1] - block_trueOffsets[idx]);
   Vector Xi(  y.GetData() + block_trueOffsets[idx], block_trueOffsets[idx+1] - block_trueOffsets[idx]);

   idx = 1;
   Vector Rp_s(x.GetData() + block_trueOffsets[idx], block_trueOffsets[idx+1] - block_trueOffsets[idx]);
   Vector Eta( y.GetData() + block_trueOffsets[idx], block_trueOffsets[idx+1] - block_trueOffsets[idx]);

   idx = 2;
   Vector Ru(x.GetData() + block_trueOffsets[idx], block_trueOffsets[idx+1] - block_trueOffsets[idx]);
   Vector Yu(y.GetData() + block_trueOffsets[idx], block_trueOffsets[idx+1] - block_trueOffsets[idx]);

   // Project zero Dirichlet boundary values and form linear system.
   xi.ProjectBdrCoefficient(*zero_DBC,pressure_ess_tdof);
   mp.FormLinearSystem(pressure_ess_tdof, xi, rp, Mp, Xi, Rp_m);

   // Apply M_solver.
   M_solver.SetOperator(Mp);
   M_solver.Mult(Rp_m, Xi);

   mp.RecoverFEMSolution(Xi, rp, xi);

   // Project Dirichlet boundary values for pressure and form linear system.
   eta.ProjectBdrCoefficient(*pressure_DBC,pressure_ess_tdof);
   sp.FormLinearSystem(pressure_ess_tdof, eta, rp, Sp, Eta, Rp_s);

   

   // Apply S_solver.
   S_solver.SetOperator(Sp);
   S_solver.Mult(Rp_s, Eta);

   sp.RecoverFEMSolution(Eta, rp, eta);

   xi.SetFromTrueVector();
   eta.SetFromTrueVector();

   

   // Do the sum y_p = -alpha1 xi - eta.
   yp.Add(neg_alpha1,xi);
   yp.Add(-1.0,eta);

   // Calculate B^T y_p and add B^T y_p to r_u for the velocity solve.
   Vector Yp(spaces[0]->GetTrueVSize());
   yp.GetTrueDofs(Yp);

   Vector Brhs(spaces[1]->GetTrueVSize()), BTyp(spaces[0]->GetTrueVSize());

   // Calculate B^T y_p.
   b.MultTranspose(Yp,BTyp);

   GridFunction bTyp(spaces[0]);
   bTyp = 0.0;

   // Recover GridFunction from Vector.
   bTyp.SetFromTrueDofs(BTyp);

   // Add B^T y_p to r_u for the velocity solve.
   ru.Add(-1.0,bTyp);
   
   // Project Dirichlet boundary values for velocity.
   Vector Yu_tmp(spaces[1]->GetTrueVSize());


   yu.ProjectBdrCoefficient(*velocity_DBC,velocity_ess_tdof);
   yu.GetTrueDofs(Yu_tmp);

   fk.FormLinearSystem(velocity_ess_tdof, yu, ru, Fk, Yu_tmp, Ru);

   // Operator set here to prevent crashes for now.
   F_solver.SetOperator(Fk);
   F_solver.Mult(Ru, Yu_tmp);

   fk.RecoverFEMSolution(Yu, ru, yu);

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
   //delete mass_pcg;
   //delete mass_prec;
   //delete stiff_prec;
   //delete stiff_pcg;
}

/*
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
*/

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
   //real_t pi = 3.14159;

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
