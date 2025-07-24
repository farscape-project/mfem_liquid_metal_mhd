#include "mfem.hpp"
#include <memory>
#include <iostream>
#include <fstream>

#include "FluidsOperator.hpp"
#include "constants.hpp"

using namespace std;
using namespace mfem;

/*
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

*/

int main(int argc, char *argv[])
{

   // Set timestepping parameters.
   double t_final = 0.5;
   double dt = 5e-3;
   int vis_steps = 10; // Visualisation every 10 steps.

   int ode_solver_type = 14;

   // Define the ODE solver used for time integration.
   ODESolver *ode_solver;
   switch (ode_solver_type)
   {
      // Implicit L-stable methods
      case 1:  ode_solver = new BackwardEulerSolver; break;
      case 2:  ode_solver = new SDIRK23Solver(2); break;
      case 3:  ode_solver = new SDIRK33Solver; break;
      // Explicit methods
      case 11: ode_solver = new ForwardEulerSolver; break;
      case 12: ode_solver = new RK2Solver(0.5); break; // midpoint method
      case 13: ode_solver = new RK3SSPSolver; break;
      case 14: ode_solver = new RK4Solver; break;
      case 15: ode_solver = new GeneralizedAlphaSolver(0.5); break;
      // Implicit A-stable methods (not L-stable)
      case 22: ode_solver = new ImplicitMidpointSolver; break;
      case 23: ode_solver = new SDIRK23Solver; break;
      case 24: ode_solver = new SDIRK34Solver; break;
      default:
      cout << "Unknown ODE solver type: " << ode_solver_type << '\n';
      return 1;
   }

   // Set fe_space orders.
   int order_pressure = 1;
   int order_velocity;
   order_velocity = order_pressure + 1;

   // Generate mesh.
   //Mesh mesh = Mesh::MakeCartesian2D(10, 4, mfem::Element::Type::QUADRILATERAL, true, 1.0, 0.2);
   Mesh mesh = Mesh::MakeCartesian2D(20, 8, mfem::Element::Type::QUADRILATERAL, true, 5.0, 1.0);
   //Mesh mesh = Mesh::MakeCartesian3D(20, 8, 8, mfem::Element::Type::QUADRILATERAL, true, 1.0, 0.2, 0.2);
   //Mesh mesh = Mesh::MakeCartesian2D(40, 16, mfem::Element::Type::QUADRILATERAL, true, 1.0, 0.2);
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

   int v_space_size = velocity_fespace.GetTrueVSize();
   int p_space_size = pressure_fespace.GetTrueVSize();

   // ----------------------------------------------------------------------------
   // Define boundaries.
   // ----------------------------------------------------------------------------
   
   //FunctionCoefficient zero_DBC(zero_dbc);

   // Essential (Dirichlet) boundary conditions for pressure.
   Array<int> ess_boundary_marker_pressure, ess_boundary_marker_velocity; 
   ess_boundary_marker_pressure.SetSize(pressure_fespace.GetMesh()->bdr_attributes.Max());
   ess_boundary_marker_pressure = 0; 
   // Dirichlet boundary condition for pressure.
   ess_boundary_marker_pressure[0] = 0; // Top
   ess_boundary_marker_pressure[1] = 1; // Outlet
   ess_boundary_marker_pressure[2] = 0; // Bottom
   ess_boundary_marker_pressure[3] = 0; // Inlet

   // Essential (Dirichlet) boundary conditions for velocity.
   ess_boundary_marker_velocity.SetSize(velocity_fespace.GetMesh()->bdr_attributes.Max());
   ess_boundary_marker_velocity = 0;
   // Dirichlet boundary conditions for velocity.
   ess_boundary_marker_velocity[0] = 1; // Top
   ess_boundary_marker_velocity[1] = 0; // Outlet
   ess_boundary_marker_velocity[2] = 1; // Bottom
   ess_boundary_marker_velocity[3] = 1; // Inlet

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
   nat_boundary_marker_pressure[3] = 1; // Inlet

   // Natural (Neumann) boundary conditions for velocity.
   nat_boundary_marker_velocity.SetSize(velocity_fespace.GetMesh()->bdr_attributes.Max());
   nat_boundary_marker_velocity = 0;
   // Set natural bcs for velocity.
   nat_boundary_marker_velocity[0] = 0; // Top
   nat_boundary_marker_velocity[1] = 1; // Outlet
   nat_boundary_marker_velocity[2] = 0; // Bottom
   nat_boundary_marker_velocity[3] = 0; // Inlet

   Array<Array<int> *> nat_bdr(2);
   nat_bdr[0] = &nat_boundary_marker_velocity;
   nat_bdr[1] = &nat_boundary_marker_pressure;

   Array<int> pressure_ess_tdof, velocity_ess_tdof;
   pressure_fespace.GetEssentialTrueDofs(ess_boundary_marker_pressure, pressure_ess_tdof);
   velocity_fespace.GetEssentialTrueDofs(ess_boundary_marker_velocity, velocity_ess_tdof);

   // Print mesh statistics.
   std::cout << "***********************************************************\n";
   std::cout << "dim(u) = " << v_space_size << "\n";
   std::cout << "dim(p) = " << p_space_size << "\n";
   std::cout << "dim(u+p) = " << v_space_size + p_space_size << "\n";
   std::cout << "***********************************************************\n";

   GridFunction xu_gf(&velocity_fespace);
   GridFunction xp_gf(&pressure_fespace);
   
   // Define block structure of the solution vector (u then p).
   Array<int> block_trueOffsets(3);
   block_trueOffsets[0] = 0;
   block_trueOffsets[1] = v_space_size;
   block_trueOffsets[2] = p_space_size;
   block_trueOffsets.PartialSum();

   BlockVector X(block_trueOffsets);
   X = 0;
   BlockVector Xn_1(block_trueOffsets), Xn_2(block_trueOffsets);
   Xn_1 = 0; Xn_2 = 0;
   Vector u_star_vec(v_space_size);
   GridFunction u_star(&velocity_fespace);

   // Set up visualisation in Paraview.
   ParaViewDataCollection paraview_dc("navier_stokes", &mesh);
   paraview_dc.SetPrefixPath("data");

   // Initialise time-loop details.
   double t = 0.0;
   int n_steps = int(t_final / dt);
   int ti_out = 0; // Time step output index.

   std::cout << "dt (main) = " << dt << std::endl;


   // Initialise fluids operator.
   FluidsOperator oper(spaces, ess_bdr, nat_bdr, block_trueOffsets, dim, dt);

   ode_solver->Init(oper);

   for (int ti = 0; ti < n_steps; ti++)
   {
      t += dt;
      std::cout << "Time step " << ti + 1 << ", time = " << t << std::endl;

      // Set history and u_star.
      if (ti == 0)
      {
         // Set u_star = i.c. for first timestep.
         u_star.SetFromTrueDofs(X.GetBlock(0)); 
      }
      else if (ti == 1)
      {
         // Set history from previous time step.
         Xn_1 = X;

         // Set u_star = u_{n-1} for second time step.
         u_star.SetFromTrueDofs(Xn_1.GetBlock(0)); 
      }
      else
      {
         // Set history from previous time steps.
         Xn_2 = Xn_1;
         Xn_1 = X;

         // Calculate u_star = (3 * u_{n-1} - u_{n-2})/2 for the convection term.
         u_star_vec = Xn_1.GetBlock(0);  
         u_star_vec *= 3.0;               
         u_star_vec -= Xn_2.GetBlock(0); 
         u_star_vec *= 0.5;
         u_star.SetFromTrueDofs(u_star_vec); 
      }
      
      // Solve problem.
      //oper.Solve(X, u_star);
      oper.Set_ustar(&u_star);
      oper.Update(X);
      ode_solver->Step(X, t, dt);

      // Grid functions for visualisation.
      xu_gf.SetFromTrueDofs(X.GetBlock(0));
      xp_gf.SetFromTrueDofs(X.GetBlock(1));

      // Visualisation in Paraview.
      if (ti % vis_steps == 0 || ti == n_steps - 1)
      {
         std::cout << "Time step " << ti + 1 << ", time = " << t << ", dt = " << dt << ", vis_steps = " << vis_steps << std::endl;
         visualize(paraview_dc, order_velocity, &xu_gf, "velocity", ti_out, t);
         visualize(paraview_dc, order_pressure, &xp_gf, "pressure", ti_out, t);
         ti_out += 1;
      }
   }

   return 0;
}

/*
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

   // Apply F solver for velocity.
   F_solver.SetOperator(Fk);
   F_solver.Mult(Ru, Yu_tmp);

   fk.RecoverFEMSolution(Yu, ru, yu);

}

void PPreconditioner::SetOperator(const Operator &op)
{
   jacobian = (BlockOperator *) &op;

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
   stiff_pcg->SetOperator(jacobian->GetBlock(0,0));
}

PPreconditioner::~PPreconditioner()
{
   //delete mass_pcg;
   //delete mass_prec;
   //delete stiff_prec;
   //delete stiff_pcg;
}

*/