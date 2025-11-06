#include "mfem.hpp"
#include <memory>
#include <iostream>
#include <fstream>
//#include "linalg/petsc.hpp"

#include "LmmhdOperator.hpp"
#include "constants.hpp"
#include "InputParser.hpp"
#include "tools.hpp"

using namespace std;
using namespace mfem;


int main(int argc, char *argv[])
{
   // Initialize MPI and HYPRE.
   Mpi::Init(argc, argv);
   int num_procs = Mpi::WorldSize();
   int myid = Mpi::WorldRank();
   Hypre::Init();

   mfem::tic();

   int ode_solver_type = 1;

   // Read in parameters from file.
   InputParser input("params.in");

   // Mesh parameters
   int nx = input.GetInt("nx");
   int ny = input.GetInt("ny");
   int nz = input.GetInt("nz");

   real_t Lx = input.GetReal("Lx");
   real_t Ly = input.GetReal("Ly");
   real_t Lz = input.GetReal("Lz");

   real_t clusterY = input.GetReal("clusterY");
   real_t clusterZ = input.GetReal("clusterZ");

   // Timestepping
   real_t t_final = input.GetReal("t_final");
   real_t dt = input.GetReal("dt");
   int vis_steps = input.GetInt("vis_steps");

   // Dimensionless parameters
   Re = input.GetReal("Re");
   Ha = input.GetReal("Ha");

   // Update constants.
   reciprocal_Re = 1.0 / Re;
   kappa_val = Ha * Ha / Re;
   neg_kappa_val = -kappa_val;
   alpha1 = alpha + reciprocal_Re;
   neg_alpha1 = -alpha1;


   if (Mpi::Root()) PrintParams(nx, ny, nz, Lx, Ly, Lz, clusterY, clusterZ, t_final, dt, vis_steps, Re, Ha);

   // Command line options.
   OptionsParser args(argc, argv);
   args.AddOption(&t_final, "-tf", "--t-final",
                  "Final time; start time is 0.");
   args.AddOption(&dt, "-dt", "--time-step",
                  "Time step.");
   args.AddOption(&vis_steps, "-vs", "--visualization-steps",
                  "Visualize every n-th timestep.");
   args.AddOption(&nx, "-nx", "--num-elements-x",
                  "Number of elements in the x direction.");
   args.AddOption(&ny, "-ny", "--num-elements-y",
                  "Number of elements in the y direction.");
   args.AddOption(&nz, "-nz", "--num-elements-z",
                  "Number of elements in the z direction.");
   args.AddOption(&Lx, "-lx", "--length-x",
                  "Length of the domain in the x direction.");
   args.AddOption(&Ly, "-ly", "--length-y",
                  "Length of the domain in the y direction.");
   args.AddOption(&Lz, "-lz", "--length-z",
                  "Length of the domain in the z direction.");
   args.AddOption(&clusterY, "-cy", "--cluster-y", "Clustering intensity in y direction.");
   args.AddOption(&clusterZ, "-cz", "--cluster-z", "Clustering intensity in z direction.");
   
   args.Parse();

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
      case 12: ode_solver = new RK2Solver(0.5); break;
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
   int order = 1;

   // Generate mesh.
   Mesh mesh = Mesh::MakeCartesian3D(nx, ny, nz, mfem::Element::Type::HEXAHEDRON, Lx, Ly, Lz);
   //const char *mesh_file = "./mesh/cuboid_clustered.msh";
   //Mesh mesh = Mesh(mesh_file, 1, 1);
   int dim = mesh.Dimension();
   
   
   // Cluster vertices in y and z.
   int numVertices = mesh.GetNV();
   for (int i = 0; i < numVertices; i++)
   {
      real_t *v = mesh.GetVertex(i);
      real_t yi = v[1] / Ly; // Normalize y
      real_t zi = v[2] / Lz; // Normalize z

      v[1] = cluster_symmetric(yi, clusterY) * Ly;
      v[2] = cluster_symmetric(zi, clusterZ) * Lz;
   }

   ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, mesh);

   
   // ----------------------------------------------------------------------------
   // Finite Element Spaces.
   // ----------------------------------------------------------------------------
   // HDiv finite elements for current density.
   RT_FECollection currentD_fec(order, dim);
   ParFiniteElementSpace currentD_fespace(pmesh, &currentD_fec);

   // H1 continuous Lagrange finite elements of given order for electric scalar potential.
   H1_FECollection electPot_fec(order_pressure, dim);
   ParFiniteElementSpace electPot_fespace(pmesh, &electPot_fec);

   // H1 continuous Lagrange finite elements of given order (order_pressure + 1)
   // for velocity.
   H1_FECollection velocity_fec(order_velocity, dim);
   ParFiniteElementSpace velocity_fespace(pmesh, &velocity_fec, dim);

   // H1 continuous Lagrange finite elements of given order for pressure.
   H1_FECollection pressure_fec(order_pressure, dim);
   ParFiniteElementSpace pressure_fespace(pmesh, &pressure_fec);


   Array<ParFiniteElementSpace *> spaces(4);
   spaces[0] = &currentD_fespace;
   spaces[1] = &electPot_fespace;
   spaces[2] = &velocity_fespace;
   spaces[3] = &pressure_fespace;

   int j_space_size = currentD_fespace.GetTrueVSize();
   int phi_space_size = electPot_fespace.GetTrueVSize();
   int v_space_size = velocity_fespace.GetTrueVSize();
   int p_space_size = pressure_fespace.GetTrueVSize();

   // ----------------------------------------------------------------------------
   // Define boundaries.
   // ----------------------------------------------------------------------------
   
   // Essential (Dirichlet) boundary conditions.
   //                                     {z1, y0, x1 (outlet), y1, x0 (inlet), z0}
   Array<int> ess_boundary_marker_currentD{1,  1,  1,           1,  1,          1};
   Array<int> ess_boundary_marker_electPot{0,  0,  0,           0,  0,          0};
   Array<int> ess_boundary_marker_pressure{0,  0,  1,           0,  0,          0};
   Array<int> ess_boundary_marker_velocity{1,  1,  0,           1,  1,          1};

   Array<Array<int> *> ess_bdr(4);
   ess_bdr[0] = &ess_boundary_marker_currentD;
   ess_bdr[1] = &ess_boundary_marker_electPot;
   ess_bdr[2] = &ess_boundary_marker_velocity;
   ess_bdr[3] = &ess_boundary_marker_pressure;

   Array<int> pressure_ess_tdof, velocity_ess_tdof, currentD_ess_tdof, electPot_ess_tdof;
   currentD_fespace.GetEssentialTrueDofs(ess_boundary_marker_currentD, currentD_ess_tdof);
   electPot_fespace.GetEssentialTrueDofs(ess_boundary_marker_electPot, electPot_ess_tdof);
   pressure_fespace.GetEssentialTrueDofs(ess_boundary_marker_pressure, pressure_ess_tdof);
   velocity_fespace.GetEssentialTrueDofs(ess_boundary_marker_velocity, velocity_ess_tdof);


   // Print mesh statistics.
   if (Mpi::Root()) PrintFESpaces(j_space_size, phi_space_size, v_space_size, p_space_size);

   ParGridFunction j_gf(&currentD_fespace);
   ParGridFunction phi_gf(&electPot_fespace);
   ParGridFunction u_gf(&velocity_fespace);
   ParGridFunction p_gf(&pressure_fespace);

   // Define block structure of the solution vector (u then p).
   Array<int> block_trueOffsets(5);
   block_trueOffsets[0] = 0;
   block_trueOffsets[1] = j_space_size;
   block_trueOffsets[2] = phi_space_size;
   block_trueOffsets[3] = v_space_size;
   block_trueOffsets[4] = p_space_size;
   block_trueOffsets.PartialSum();

   BlockVector X(block_trueOffsets);
   X = 0;
   BlockVector Xn_1(block_trueOffsets), Xn_2(block_trueOffsets);
   Xn_1 = 0; Xn_2 = 0;
   Vector u_star_vec(v_space_size);
   ParGridFunction u_star(&velocity_fespace);

   // Set up visualisation in Paraview.
   ParaViewDataCollection paraview_dc("lmmhd", pmesh);
   paraview_dc.SetPrefixPath("data");

   // Initialise time-loop details.
   real_t t = 0.0;
   int n_steps = int(t_final / dt);
   int ti_out = 0; // Time step output index.
   int ti = 0;

   // Initialise liquid-metal MHD operator.
   LmmhdOperator oper(spaces, ess_bdr, block_trueOffsets, dim, dt);

   ode_solver->Init(oper);

   //for (int ti = 0; ti < n_steps; ti++)
   while (t < t_final)
   {
      if (Mpi::Root()) { std::cout << "Time step " << ti << ", time = " << t << ", time elapsed = " << mfem::toc() << std::endl; }

      // Set history and u_star.
      if (ti == 0)
      {
         // Set u_star = i.c. for first timestep.
         u_star.SetFromTrueDofs(X.GetBlock(2)); 
      }
      else if (ti == 1)
      {
         // Set history from previous time step.
         Xn_1 = X;

         // Set u_star = u_{n-1} for second time step.
         u_star.SetFromTrueDofs(Xn_1.GetBlock(2)); 
      }
      else
      {
         // Set history from previous time steps.
         Xn_2 = Xn_1;
         Xn_1 = X;

         // Calculate u_star = (3 * u_{n-1} - u_{n-2})/2 for the convection term.
         u_star_vec = 0.0;
         u_star_vec = Xn_1.GetBlock(2);  
         u_star_vec *= 3.0;               
         u_star_vec -= Xn_2.GetBlock(2); 
         u_star_vec *= 0.5;
         u_star.SetFromTrueDofs(u_star_vec); 
      }
      
      // Solve problem.
      oper.Set_ustar(&u_star);
      oper.Update(X);
      ode_solver->Step(X, t, dt);
      
      // Grid functions for visualisation.
      j_gf.SetFromTrueDofs(X.GetBlock(0));
      phi_gf.SetFromTrueDofs(X.GetBlock(1));
      u_gf.SetFromTrueDofs(X.GetBlock(2));
      p_gf.SetFromTrueDofs(X.GetBlock(3));


      // Visualisation in Paraview.
      if (ti % vis_steps == 0 || ti == n_steps - 1)
      {
         if (Mpi::Root()) { std::cout << "Time step " << ti << ", time = " << t-dt << ", dt = " << dt << ". Print step " << ti_out << std::endl; }
         visualise(paraview_dc, order, &j_gf, "current density", ti_out, t);
         visualise(paraview_dc, order, &phi_gf, "electric potential", ti_out, t);
         visualise(paraview_dc, order_velocity, &u_gf, "velocity", ti_out, t);
         visualise(paraview_dc, order_pressure, &p_gf, "pressure", ti_out, t);
         ti_out += 1;
      }

      ti += 1;
   }

   if (Mpi::Root()) { std::cout << "Total simulation time: " << mfem::toc() << std::endl; }
   return 0;
}
