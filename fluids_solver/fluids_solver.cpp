#include "mfem.hpp"
#include <memory>
#include <iostream>
#include <fstream>

#include "FluidsOperator.hpp"
#include "constants.hpp"

using namespace std;
using namespace mfem;

int main(int argc, char *argv[])
{

   // Initialize MPI and HYPRE.
   Mpi::Init(argc, argv);
   int num_procs = Mpi::WorldSize();
   int myid = Mpi::WorldRank();
   Hypre::Init();

   // Set timestepping parameters.
   double t_final = 0.5;
   double dt = 5e-3;
   int vis_steps = 5;

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
   
   ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, mesh);

   
   // ----------------------------------------------------------------------------
   // Finite Element Spaces.
   // ----------------------------------------------------------------------------
   // H1 continuous Lagrange finite elements of given order for pressure.
   H1_FECollection pressure_fec(order_pressure, dim);
   ParFiniteElementSpace pressure_fespace(pmesh, &pressure_fec);

   // H1 continuous Lagrange finite elements of given order (order_pressure + 1)
   // for velocity.
   H1_FECollection velocity_fec(order_velocity, dim);
   ParFiniteElementSpace velocity_fespace(pmesh, &velocity_fec, dim);

   Array<ParFiniteElementSpace *> spaces(2);
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

   ParGridFunction xu_gf(&velocity_fespace);
   ParGridFunction xp_gf(&pressure_fespace);
   
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
   ParGridFunction u_star(&velocity_fespace);

   // Set up visualisation in Paraview.
   ParaViewDataCollection paraview_dc("navier_stokes", &mesh);
   paraview_dc.SetPrefixPath("data");

   // Initialise time-loop details.
   double t = 0.0;
   int n_steps = int(t_final / dt);
   int ti_out = 0; // Time step output index.

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
